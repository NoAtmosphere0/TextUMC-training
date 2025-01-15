import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
import numpy as np
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
import logging
import torch.cuda
from torch.utils.data import DataLoader, Dataset
import json
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import math
from sklearn.metrics import pairwise_distances
import os
from datetime import datetime
from typing import Any

# Add imports at top
from rich.console import Console
from rich.traceback import install
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
from rich import print as rprint
import sys
from sklearn.manifold import TSNE

# Install rich traceback
install(show_locals=False)

# Initialize rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)


@dataclass
class Evidence:
    """Represents a piece of evidence associated with a claim."""

    evidence_id: str
    content: str
    embedding: Optional[np.ndarray] = None


@dataclass
class Claim:
    """Represents a claim with its associated metadata and evidences."""

    claim_id: str
    content: str
    label: int
    explanation: str
    evidences: List[Evidence] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    clustered_evidences: Optional[Dict[int, List[Evidence]]] = None


class TextDataset(Dataset):
    """Custom dataset for text data"""

    def __init__(self, texts: List[str]):
        if not texts:
            raise ValueError("Empty text list provided")
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        if not 0 <= idx < len(self.texts):
            raise IndexError("Dataset index out of range")
        return self.texts[idx]


class TextUMC(nn.Module):
    def __init__(
        self,
        bert_model_name="bert-base-uncased",
        embedding_dim=768,
        hidden_dim=256,
        reduced_dim=128,
    ):
        super(TextUMC, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # BERT components
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_encoder = BertModel.from_pretrained(bert_model_name)

        # Dimensions
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.reduced_dim = reduced_dim

        # Layers
        self.projection = nn.Linear(self.embedding_dim, self.hidden_dim)
        self.reduction = nn.Linear(self.hidden_dim, self.reduced_dim)

        # Augmentation layers
        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # Memory optimization
        self.bert_encoder.config.gradient_checkpointing = True
        torch.backends.cudnn.benchmark = True
        self.optimizer = None
        self.console = Console()

        self.to(self.device)

    def debug_tensor(self, tensor: torch.Tensor, name: str):
        """Debug tensor properties"""
        table = Table(title=f"Tensor Debug: {name}")
        table.add_column("Property")
        table.add_column("Value")
        table.add_row("Shape", str(tensor.shape))
        table.add_row("Device", str(tensor.device))
        table.add_row("Type", str(tensor.dtype))
        table.add_row("Mean", f"{tensor.mean().item():.4f}")
        table.add_row("Std", f"{tensor.std().item():.4f}")
        self.console.print(table)

    @torch.amp.autocast(device_type="cuda")
    def forward(self, texts: List[str]) -> torch.Tensor:
        # Add input validation
        if not texts:
            raise ValueError("Empty text input")

        # Process through BERT
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.bert_encoder(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Project through MLP
        x = self.projection(embeddings)
        x = F.relu(x)
        x = self.reduction(x)

        # Check for NaN values
        if torch.isnan(x).any():
            raise ValueError("NaN values detected in embeddings")

        return x

    def augment_views(self, z_t):
        # Add validation
        if torch.isnan(z_t).any():
            raise ValueError("NaN values in input to augment_views")

        # Simple dropout-based augmentation
        z1 = self.dropout1(z_t)
        z2 = self.dropout2(z_t)

        return z1, z2


def unsupervised_contrastive_loss(embeddings, temp=0.07):
    """SimCLR-style unsupervised contrastive loss"""
    batch_size = embeddings.size(0) // 2
    device = embeddings.device

    # Normalize embeddings
    norm_emb = F.normalize(embeddings, dim=1)

    # Create instance mask
    mask = torch.ones((batch_size * 2, batch_size * 2), device=device)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    mask = mask.bool()

    # Compute similarities
    sim = torch.matmul(norm_emb, norm_emb.T) / temp

    # Extract positive pairs
    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    # Organize positive and negative samples
    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(batch_size * 2, 1)
    negative_samples = sim[mask].reshape(batch_size * 2, -1)

    # Compute logits and loss
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    labels = torch.zeros(batch_size * 2, device=device).long()

    # Cross entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss


def supervised_contrastive_loss(embeddings, labels, temp=0.07):
    """Supervised contrastive loss with label information"""
    batch_size = embeddings.size(0)
    device = embeddings.device

    # Normalize embeddings
    norm_emb = F.normalize(embeddings, dim=1)

    # Create label mask
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()

    # Compute similarities
    sim_matrix = torch.matmul(norm_emb, norm_emb.T)

    # For numerical stability
    sim_matrix = torch.div(sim_matrix, temp)
    sim_matrix_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_matrix_max.detach()

    # Mask out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0
    )

    mask = mask * logits_mask

    # Compute log probabilities
    exp_sim_matrix = torch.exp(sim_matrix) * logits_mask
    log_prob = sim_matrix - torch.log(exp_sim_matrix.sum(1, keepdim=True))

    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # Loss
    loss = -mean_log_prob_pos.mean()

    return loss


def calculate_densities_and_optimal_knear(
    embeddings,
    cluster_indices,
    num_candidates=10,
    lower_bound=0.1,
    interval=0.02,
    device="cpu",
):
    """Enhanced density calculation with optimal k-near selection"""
    num_embeddings = len(embeddings)

    # Generate k-near candidates adaptively
    k_near_candidates = [
        max(1, math.floor(num_embeddings * (lower_bound + interval * q)))
        for q in range(num_candidates)
    ]
    k_near_candidates = sorted(set(k_near_candidates))

    # Calculate pairwise distances once
    distances = pairwise_distances(embeddings.detach().cpu().numpy())

    best_score = float("-inf")
    optimal_knear = k_near_candidates[0]
    final_densities = None

    # Find optimal k-near
    for k_near in k_near_candidates:
        densities = []
        for i in range(num_embeddings):
            # Get k nearest neighbors
            nn_indices = np.argsort(distances[i])[1 : k_near + 1]
            avg_dist = np.mean(distances[i, nn_indices])
            density = 1 / (avg_dist + 1e-8)  # Avoid division by zero
            densities.append(density)

        # Calculate cluster cohesion score
        cohesion_score = calculate_cluster_cohesion(
            embeddings, densities, k_near, cluster_indices, device
        )

        if cohesion_score > best_score:
            best_score = cohesion_score
            optimal_knear = k_near
            final_densities = np.array(densities)

    return final_densities, optimal_knear


def select_diverse_samples(embeddings, densities, optimal_knear, num_samples):
    """Select diverse high-density samples"""
    selected = []
    remaining = list(range(len(embeddings)))
    distances = pairwise_distances(embeddings.detach().cpu().numpy())

    # Select first sample with highest density
    first_idx = remaining[np.argmax(densities)]
    selected.append(first_idx)
    remaining.remove(first_idx)

    # Select remaining samples
    while len(selected) < num_samples:
        max_min_dist = float("-inf")
        next_idx = None

        # Find point with maximum minimum distance to selected points
        for idx in remaining:
            min_dist = min(distances[idx][selected])
            if min_dist > max_min_dist and densities[idx] > np.median(densities):
                max_min_dist = min_dist
                next_idx = idx

        if next_idx is None:
            break
        selected.append(next_idx)
        remaining.remove(next_idx)

    return selected


def cluster_and_select_samples(embeddings, num_clusters, t, current_centroids, device):
    # Add validation
    if torch.isnan(embeddings).any():
        raise ValueError("NaN values in embeddings before clustering")

    # Convert to numpy safely
    embeddings_np = embeddings.detach().cpu().numpy()

    # Handle potential NaN values
    if np.isnan(embeddings_np).any():
        logging.warning("Cleaning NaN values before clustering")
        embeddings_np = np.nan_to_num(embeddings_np, 0)

    kmeans = KMeans(
        n_clusters=num_clusters,
        init=current_centroids if current_centroids is not None else "k-means++",
        n_init=5,
        max_iter=200,
    )

    kmeans.fit(embeddings_np)
    return (
        kmeans.labels_,
        torch.tensor(kmeans.labels_).to(device),
        kmeans.cluster_centers_,
    )


def calculate_cluster_cohesion(embeddings, densities, k_near, cluster_indices, device):
    """Calculates the cohesion of a cluster using density values.

    Args:
        embeddings (torch.Tensor): Embeddings of a cluster.
        densities (np.ndarray): List of density values.
        k_near (int): Optimal k near value.
        cluster_indices (np.ndarray): List of indices belonging to the current cluster.
        device (str): Device to perform calculations on.

    Returns:
        float: Cohesion value of the cluster.
    """
    num_embeddings = len(embeddings)
    if num_embeddings <= 1:
        return 0  # handle the cases where the cluster has only one or no embedding.

    distances = pairwise_distances(embeddings.detach().cpu().numpy())
    total_cohesion = 0

    for i in range(num_embeddings):
        nearest_neighbors_indices = np.argsort(distances[i])[
            1 : k_near + 1
        ]  # Exclude self, take top k

        if nearest_neighbors_indices.size == 0:
            continue  # Handles the case where there is no neighbors

        total_cohesion += np.mean(distances[i, nearest_neighbors_indices])

    cohesion_score = total_cohesion / num_embeddings if num_embeddings > 0 else 0
    return -cohesion_score  # Return negative to maximize through argmax


def umc_train(
    model,
    evidences: List[Evidence],
    num_clusters: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: int = 100,
    temp_1: float = 0.2,  # Temperature for unsupervised loss
    temp_2: float = 0.2,  # Temperature for supervised loss
) -> Tuple[List[Evidence], dict]:
    """Train model on evidence texts using combined UMC loss"""
    # Get evidence texts
    evidence_texts = [ev.content for ev in evidences]

    # Adjust parameters based on evidence count
    batch_size = min(batch_size, len(evidence_texts))
    num_clusters = min(num_clusters, len(evidence_texts))

    if num_clusters < 2:
        num_clusters = 2
        logging.warning(f"Adjusted clusters to minimum: {num_clusters}")

    # Initialize optimizer
    if model.optimizer is None:
        model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    metrics = {"epoch_losses": [], "unsup_losses": [], "sup_losses": []}
    current_centroids = None

    # Training loop
    try:
        for epoch in track(range(num_epochs), description="Training"):
            epoch_loss = 0
            for i in range(0, len(evidence_texts), batch_size):
                batch_texts = evidence_texts[i : i + batch_size]
                embeddings = model(batch_texts)

                # Get augmented views
                z1, z2 = model.augment_views(embeddings)
                aug_embeddings = torch.cat([z1, z2], dim=0)

                # Unsupervised contrastive loss
                unsup_loss = unsupervised_contrastive_loss(aug_embeddings, temp=temp_1)

                # Get cluster assignments for supervised loss
                _, pseudo_labels, current_centroids = cluster_and_select_samples(
                    embeddings, num_clusters, 0.5, current_centroids, model.device
                )

                # Supervised contrastive loss with pseudo-labels
                sup_loss = supervised_contrastive_loss(
                    embeddings, pseudo_labels, temp=temp_2
                )

                # Combine losses
                loss = unsup_loss + sup_loss

                model.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                model.optimizer.step()

                epoch_loss += loss.item()

                # Add NaN checks
                if torch.isnan(embeddings).any():
                    raise ValueError("NaN embeddings detected")

                metrics["unsup_losses"].append(unsup_loss.item())
                metrics["sup_losses"].append(sup_loss.item())

            avg_loss = epoch_loss / (len(evidence_texts) / batch_size)
            metrics["epoch_losses"].append(avg_loss)

            # Add debug info
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch} loss: {avg_loss:.4f}")

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

    return evidences, metrics


@torch.no_grad()
def _save_embeddings(model, claims):
    """Save embeddings for all claims"""
    for claim in claims:
        claim_text = [claim.content]
        embedding = (
            model(claim_text).detach().cpu().numpy()
        )  # Extract the final embeddings.
        claim.embedding = embedding.flatten()


def visualize_clusters_pca(embeddings, labels, title, save_path):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    return plot_clusters(reduced, labels, f"{title} (PCA)", save_path)


def visualize_clusters_tsne(embeddings, labels, title, save_path):
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000)
    reduced = tsne.fit_transform(embeddings)
    return plot_clusters(reduced, labels, f"{title} (t-SNE)", save_path)


def plot_clusters(reduced_embeddings, labels, title, save_path):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis"
    )
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(save_path)
    plt.close()


def visualize_clusters(
    embeddings: np.ndarray, labels: np.ndarray, title: str, save_path: str
):
    """Plot clusters using both PCA and t-SNE side by side"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # PCA reduction
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)

    # t-SNE reduction
    tsne = TSNE(n_components=2, perplexity=10, max_iter=1000)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Plot PCA
    scatter1 = ax1.scatter(
        pca_embeddings[:, 0], pca_embeddings[:, 1], c=labels, cmap="viridis"
    )
    ax1.set_title("PCA Projection")
    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")

    # Plot t-SNE
    scatter2 = ax2.scatter(
        tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels, cmap="viridis"
    )
    ax2.set_title("t-SNE Projection")
    ax2.set_xlabel("First Component")
    ax2.set_ylabel("Second Component")

    # Add color bar
    plt.colorbar(scatter1, ax=ax1)
    plt.colorbar(scatter2, ax=ax2)

    # Set main title
    fig.suptitle(title, fontsize=16)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def print_evidence_clusters(claim: Claim, cluster_labels: np.ndarray):
    """Display evidence clusters in a readable format"""
    table = Table(title=f"Evidence Clusters for Claim: {claim.claim_id}")
    table.add_column("Cluster", justify="center", style="cyan")
    table.add_column("Evidence Content", justify="left", style="green")

    for cluster_id in range(max(cluster_labels) + 1):
        # Get evidences for this cluster
        cluster_evidences = [
            ev
            for ev, label in zip(claim.evidences, cluster_labels)
            if label == cluster_id
        ]

        # Add row for each evidence in cluster
        for idx, evidence in enumerate(cluster_evidences):
            # Truncate content for readability
            content = (
                evidence.content[:100] + "..."
                if len(evidence.content) > 100
                else evidence.content
            )
            table.add_row(f"Cluster {cluster_id}" if idx == 0 else "", content)
        # Add separator between clusters
        table.add_row("", "")

    console.print(table)


def main():
    """Main function to run full text clustering pipeline"""
    # Setup output directory
    output_dir = "outputs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        # Load example data
        example_data = json.load(open("dataset/LIAR-RAW/test.json"))[403]
        example_claim = Claim(
            claim_id="403",
            content=example_data["claim"],
            label=example_data["label"],
            explanation=example_data["explain"],
        )

        # Load evidences
        for evidence in example_data["reports"]:
            example_claim.evidences.append(
                Evidence(evidence_id=evidence["report_id"], content=evidence["content"])
            )

        print(f"Claim: {example_claim.content}")
        print(f"Number of evidences: {len(example_claim.evidences)}")

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TextUMC().to(device)

        # Train model
        trained_evidences, metrics = umc_train(
            model=model,
            evidences=example_claim.evidences,
            num_clusters=min(len(example_claim.evidences), 5),
            batch_size=32,
            num_epochs=200,
            learning_rate=2e-5,
        )

        # Get evidence embeddings
        evidence_texts = [ev.content for ev in example_claim.evidences]
        with torch.no_grad():
            evidence_embeddings = model(evidence_texts)
            evidence_embeddings_np = evidence_embeddings.cpu().numpy()

        print(f"Evidence Embeddings: {evidence_embeddings_np.shape}")

        # Cluster evidences
        n_clusters = min(len(evidence_texts), 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(evidence_embeddings_np)

        # Print clusters
        print_evidence_clusters(example_claim, cluster_labels)

        # Group evidences
        example_claim.clustered_evidences = {
            i: [
                ev
                for ev, label in zip(example_claim.evidences, cluster_labels)
                if label == i
            ]
            for i in range(n_clusters)
        }

        # Visualize clusters
        plot_path = os.path.join(run_dir, "evidence_clusters.png")
        visualize_clusters(
            evidence_embeddings_np,
            cluster_labels,
            f"Evidence Clusters for Claim {example_claim.claim_id}",
            plot_path,
        )

        # Save results
        results = {
            "claim": {
                "claim_id": example_claim.claim_id,
                "content": example_claim.content,
                "label": example_claim.label,
                "explanation": example_claim.explanation,
                "clustered_evidences": {
                    str(key): [
                        {"evidence_id": ev.evidence_id, "content": ev.content}
                        for ev in value
                    ]
                    for key, value in example_claim.clustered_evidences.items()
                },
            },
            "training_metrics": metrics,
        }

        with open(os.path.join(run_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=4)

        logging.info(f"Results saved in {run_dir}")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
