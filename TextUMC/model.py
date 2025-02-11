""" TextUMC model for unsupervised text clustering """

# TODO: Refactor code, split into files, add type hints, add docstrings, add logging, add rich output

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.cluster import KMeans
import numpy as np
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm, trange
import logging
import torch.cuda
from torch.utils.tensorboard import SummaryWriter
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
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from rich.console import Console
from rich.traceback import install
from rich.logging import RichHandler
from rich.progress import track
from rich.table import Table
from rich import print as rprint
import sys
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from cluster_evaluator import ClusteringEvaluator

# Install rich traceback
install(show_locals=False)

# Initialize rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

import warnings

warnings.filterwarnings("ignore")


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


@dataclass
class ModelConfig:
    """Configuration for TextUMC model."""

    bert_model_name: str = "bert-large-uncased"
    embedding_dim: int = 1024
    hidden_dim: int = 512
    reduced_dim: int = 256
    dropout_rate: float = 0.2
    temperature: float = 0.07


class TextUMC(nn.Module):
    """Text Unsupervised Multi-view Clustering model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize BERT components
        self._init_bert()

        # Initialize projection layers
        self._init_layers()

        # Optimize memory usage
        self._optimize_memory()

        self.to(self.device)

        self.optimizer = None

    def _init_bert(self):
        """Initialize BERT encoder and tokenizer."""
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_model_name)
        self.bert_encoder = BertModel.from_pretrained(self.config.bert_model_name)

    def _init_layers(self):
        """Initialize projection and augmentation layers."""
        self.projection = nn.Sequential(
            nn.Linear(self.config.embedding_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.reduced_dim),
        )

        self.dropout1 = nn.Dropout(p=self.config.dropout_rate)
        self.dropout2 = nn.Dropout(p=self.config.dropout_rate)

        self.reduction = nn.Sequential(
            nn.Linear(self.config.reduced_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.reduced_dim),
        )

    def _optimize_memory(self):
        """Apply memory optimization settings."""
        self.bert_encoder.config.gradient_checkpointing = True
        torch.backends.cudnn.benchmark = True

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
        # x = self.reduction(x)
        x = F.normalize(x, dim=1)

        # Check for NaN values
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        return x

    def augment_views(self, z_t):
        # Add validation
        if torch.isnan(z_t).any():
            raise ValueError("NaN values in input to augment_views")

        # Simple dropout-based augmentation
        z1 = self.dropout1(z_t)
        z2 = self.dropout2(z_t)

        return z1, z2

    def save_model(self, path: str):
        """Save model to disk"""
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        """Load model from disk"""
        self.load_state_dict(torch.load(path))


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
    # Convert to tensor
    labels = torch.tensor(labels, device=device)
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
    """Enhanced density calculation with optimal k-near selection using reachability distance.

    Args:
        embeddings: Tensor of shape (n_samples, n_features)
        cluster_indices: Array of current cluster assignments
        num_candidates: Number of k-nearest neighbor candidates to try
        lower_bound: Minimum proportion of points to consider as neighbors
        interval: Step size for generating k-near candidates
        device: Device to perform calculations on

    Returns:
        tuple: (final_densities, optimal_knear)
    """
    num_embeddings = len(embeddings)
    embeddings_np = embeddings.detach().cpu().numpy()

    # Generate k-near candidates as proportions of total points
    k_candidate_proportions = np.arange(
        lower_bound, lower_bound + interval * num_candidates, interval
    )
    k_near_candidates = [
        max(1, int(len(embeddings_np) * k)) for k in k_candidate_proportions
    ]
    k_near_candidates = sorted(set(k_near_candidates))

    best_score = float("-inf")
    optimal_knear = k_near_candidates[0]
    final_densities = None

    # Find optimal k-near using reachability distance
    for k_near in k_near_candidates:
        # Use NearestNeighbors for efficient neighbor computation
        nbrs = NearestNeighbors(n_neighbors=k_near + 1, algorithm="auto").fit(
            embeddings_np
        )
        distances, indices = nbrs.kneighbors(embeddings_np)

        # Calculate reachability-based density
        reachable_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self
        densities = 1 / (reachable_distances + 1e-8)  # Avoid division by zero

        # Calculate cluster cohesion score
        cohesion_score = calculate_cluster_cohesion(
            embeddings, densities, k_near, cluster_indices, device
        )

        if cohesion_score > best_score:
            best_score = cohesion_score
            optimal_knear = k_near
            final_densities = densities

    return np.array(final_densities), optimal_knear


def select_diverse_samples(embeddings, densities, optimal_knear, num_samples):
    """Select diverse high-density samples"""
    selected = []
    remaining = list(range(len(embeddings)))
    distances = pairwise_distances(embeddings)

    # Select first sample with highest density
    first_idx = np.argmax(densities)
    selected.append(first_idx)
    remaining.remove(first_idx)

    # Iteratively select samples that maximize density and diversity
    while len(selected) < num_samples and remaining:
        max_score = float("-inf")
        best_idx = None

        # Calculate scores for remaining samples
        for idx in remaining:
            # Density component
            density_score = densities[idx]

            # Diversity component - minimum distance to already selected samples
            diversity_score = min(distances[idx][j] for j in selected)

            # Combined score with density and diversity
            score = density_score * diversity_score

            if score > max_score:
                max_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)
        else:
            break

    return np.array(selected)


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

    cluster_indices = np.zeros(len(embeddings_np))

    # Calculate densities and optimal k-near
    densities, optimal_knear = calculate_densities_and_optimal_knear(
        embeddings=embeddings,
        cluster_indices=cluster_indices,
        num_candidates=10,
        lower_bound=0.1,
        interval=0.02,
        device=device,
    )

    # Select diverse samples
    selected_indices = select_diverse_samples(
        embeddings=embeddings_np,
        densities=densities,
        optimal_knear=optimal_knear,
        num_samples=num_clusters,
    )

    # Update cluster indices
    for idx in selected_indices:
        cluster_indices[idx] = 1  # Mark as selected

    if optimal_knear <= len(embeddings_np):
        # Use selected samples as initial centroids
        initial_centroids = embeddings_np[selected_indices]
    else:
        # Fallback to existing centroids or k-means++
        initial_centroids = (
            current_centroids if current_centroids is not None else "k-means++"
        )

    # Perform KMeans clustering
    kmeans = KMeans(
        n_clusters=(
            len(initial_centroids)
            if optimal_knear > len(embeddings_np)
            else num_clusters
        ),
        init=initial_centroids,
        n_init=1,
        max_iter=200,
        random_state=42,  # Add for reproducibility
    )

    # Fit KMeans
    cluster_labels = kmeans.fit_predict(embeddings_np)

    return cluster_labels, cluster_indices, kmeans.cluster_centers_


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
        return 0.0

    # Convert embeddings to numpy if needed
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()

    # Calculate pairwise distances
    distances = pairwise_distances(embeddings)

    # Get k nearest neighbors for each point
    k = min(k_near, num_embeddings - 1)
    nearest_distances = np.partition(distances, k, axis=1)[:, :k]

    # Weight distances by density
    density_weights = densities.reshape(-1, 1)
    weighted_distances = nearest_distances * density_weights

    # Calculate cohesion score
    cohesion = -np.mean(weighted_distances)

    # Normalize by cluster size
    cohesion = cohesion / (num_embeddings + 1e-8)

    return float(cohesion)


def umc_train(
    model,
    evidences: List[Evidence],
    num_clusters: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: float = 5.0,
    temp_1: float = 0.2,
    temp_2: float = 0.2,
    log_dir: Optional[str] = None,
    save_after_n_epochs: int = 0,  # Add this parameter
) -> Tuple[List[Evidence], dict]:
    """Train model on evidence texts using combined UMC loss with batch processing

    Args:
        model (TextUMC): TextUMC model instance
        evidences (List[Evidence]): List of evidence objects
        num_clusters (int): Number of clusters to form
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        num_epochs (float): Number of training epochs
        temp_1 (float): Temperature for unsupervised loss
        temp_2 (float): Temperature for supervised loss

    Returns:
        Tuple[List[Evidence], dict]: Tuple containing updated evidences and training metrics
    """

    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir)

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

    if batch_size != -1:
        # Create data loader
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(range(len(evidence_texts)), dtype=torch.long)
        )
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
    else:
        data_loader = [torch.tensor(range(len(evidence_texts)))]

    metrics = {
        "epoch_losses": [],
        "unsup_losses": [],
        "sup_losses": [],
        "batch_losses": [],
    }
    current_centroids = None

    # Training loop
    try:
        # Handle epochs < 1 as fraction of data_loader length
        if num_epochs < 1:
            total_iterations = max(1, int(num_epochs * len(data_loader)))
        else:
            total_iterations = int(num_epochs * len(data_loader))

        iteration = 0
        current_epoch = 0
        last_saved_epoch = -1  # Track last epoch when model was saved

        progress_bar = trange(total_iterations, desc="Training", unit="iteration")

        while iteration < total_iterations:
            epoch_loss = 0
            epoch_unsup_loss = 0
            epoch_sup_loss = 0
            num_batches = 0

            # Re-initialize data loader for each epoch
            if iteration % len(data_loader) == 0:
                data_loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True
                )

            for batch_idx, batch in enumerate(data_loader):
                if iteration >= total_iterations:
                    break

                # Extract texts for this batch
                if batch_size != -1:
                    batch_texts = list(
                        map(evidence_texts.__getitem__, batch[0].tolist())
                    )
                else:
                    batch_texts = evidence_texts

                # Get embeddings for the batch
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
                sup_loss = 0
                sup_loss = supervised_contrastive_loss(
                    embeddings, pseudo_labels, temp=temp_2
                )

                # Combine losses
                loss = unsup_loss + sup_loss

                # Zero gradients, backpropagate, and optimize
                model.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                model.optimizer.step()

                # Tracking metrics
                batch_loss = loss.item()
                epoch_loss += batch_loss
                epoch_unsup_loss += unsup_loss.item()
                # epoch_sup_loss += sup_loss.item()
                num_batches += 1

                # Add NaN checks
                if torch.isnan(embeddings).any():
                    raise ValueError("NaN embeddings detected in batch")

                metrics["batch_losses"].append(batch_loss)

                iteration += 1
                progress_bar.update(1)

                # Update epoch counter
                current_epoch = iteration / len(data_loader)

                # Save model if needed
                if save_after_n_epochs > 0:
                    current_epoch_int = int(current_epoch)
                    if (
                        current_epoch_int > last_saved_epoch
                        and current_epoch_int % save_after_n_epochs == 0
                    ):
                        if log_dir:
                            model_path = os.path.join(
                                log_dir, f"model_epoch_{current_epoch_int}.pt"
                            )
                            model.save_model(model_path)
                            logging.info(f"Model saved at epoch {current_epoch_int}")
                            last_saved_epoch = current_epoch_int

                if writer:
                    global_step = iteration
                    writer.add_scalar("Loss/batch_total", batch_loss, global_step)
                    writer.add_scalar(
                        "Loss/batch_unsupervised", unsup_loss.item(), global_step
                    )
                    writer.add_scalar(
                        "Loss/batch_supervised", sup_loss.item(), global_step
                    )

            # Compute and log epoch metrics only when a full epoch is completed
            if int(current_epoch) > int(current_epoch - 1.0 / len(data_loader)):
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
                avg_unsup_loss = (
                    epoch_unsup_loss / num_batches if num_batches > 0 else 0
                )
                avg_sup_loss = epoch_sup_loss / num_batches if num_batches > 0 else 0

                metrics["epoch_losses"].append(avg_epoch_loss)
                metrics["unsup_losses"].append(avg_unsup_loss)
                metrics["sup_losses"].append(avg_sup_loss)

                if writer:
                    writer.add_scalar(
                        "Loss/epoch_total", avg_epoch_loss, int(current_epoch)
                    )
                    writer.add_scalar(
                        "Loss/epoch_unsupervised", avg_unsup_loss, int(current_epoch)
                    )
                    writer.add_scalar(
                        "Loss/epoch_supervised", avg_sup_loss, int(current_epoch)
                    )

        # Save final model
        if log_dir:
            final_model_path = os.path.join(log_dir, "model_final.pt")
            model.save_model(final_model_path)
            logging.info("Final model saved")

        progress_bar.close()

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


def visualize_clusters(
    embeddings: torch.Tensor,
    labels: np.ndarray,
    title: str,
    save_path: str,
    metric_name: str,
):
    """Plot clusters using both PCA and t-SNE side by side"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Convert cuda:0 to cpu if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    # PCA reduction
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)

    # t-SNE reduction
    tsne = TSNE(n_components=2, perplexity=5, max_iter=1000)
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Plot PCA
    scatter1 = ax1.scatter(
        pca_embeddings[:, 0], pca_embeddings[:, 1], c=labels, cmap="viridis"
    )
    ax1.set_title(f"PCA Projection - {metric_name}")
    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")

    # Plot t-SNE
    scatter2 = ax2.scatter(
        tsne_embeddings[:, 0], tsne_embeddings[:, 1], c=labels, cmap="viridis"
    )
    ax2.set_title(f"t-SNE Projection - {metric_name}")
    ax2.set_xlabel("First Component")
    ax2.set_ylabel("Second Component")

    # Add color bar
    plt.colorbar(scatter1, ax=ax1)
    plt.colorbar(scatter2, ax=ax2)

    # Set main title
    fig.suptitle(f"{title} - {metric_name}", fontsize=16)

    # Adjust layout and save
    plt.tight_layout()
    save_file = os.path.join(save_path, f"clusters_{title}_{metric_name}.png")
    plt.savefig(save_file)
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


# def main():
#     """Main function to run full text clustering pipeline"""
#     # Setup output directory
#     output_dir = "outputs"
#     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = os.path.join(output_dir, run_id)
#     os.makedirs(run_dir, exist_ok=True)

#     try:
#         # Load example data
#         # data = json.load(open("dataset/LIAR-RAW/test.json"))

#         # Get 403rd claim
#         example_data = json.load(open("dataset/LIAR-RAW/test.json"))[403]

#         example_claim = Claim(
#             claim_id="403",
#             content=example_data["claim"],
#             label=example_data["label"],
#             explanation=example_data["explain"],
#         )

#         # Load evidences
#         for evidence in example_data["reports"]:
#             example_claim.evidences.append(
#                 Evidence(evidence_id=evidence["report_id"], content=evidence["content"])
#             )
#         # print(f"Number of claims: {len(data)}")

#         # Initialize clustering evaluator
#         evaluator = ClusteringEvaluator()

#         with Progress(
#             SpinnerColumn(),
#             TextColumn("[progress.description]{task.description}"),
#             BarColumn(),
#             TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
#             TimeElapsedColumn(),
#         ) as progress:
#             # Create the main claims progress bar
#             claims_task = progress.add_task(
#                 "[cyan]Processing Claims...", total=len(data)
#             )

#             for idx, claim_data in enumerate(data):
#                 # Create a task for the training epochs
#                 training_task = progress.add_task(
#                     f"[green]Training Claim {idx}", total=100
#                 )

#                 if len(claim_data["reports"]) < 5:
#                     continue

#                 claim = Claim(
#                     claim_id=str(idx),
#                     content=claim_data["claim"],
#                     label=claim_data["label"],
#                     explanation=claim_data["explain"],
#                 )

#                 # Load evidences
#                 for evidence in claim_data["reports"]:
#                     claim.evidences.append(
#                         Evidence(
#                             evidence_id=evidence["report_id"],
#                             content=evidence["content"],
#                         )
#                     )

#         # Clear GPU cache and initialize model
#         torch.cuda.empty_cache()

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = TextUMC().to(device)

#         # Train model
#         _, metrics = umc_train(
#             model=model,
#             evidences=claim.evidences,
#             num_clusters=min(len(claim.evidences), 5),
#             batch_size=32,
#             num_epochs=100,
#             learning_rate=2e-5,
#             progress=progress,
#             train_task=training_task,
#         )

#         # Remove the training task when done
#         progress.remove_task(training_task)

#         # Get evidence embeddings
#         evidence_texts = [ev.content for ev in claim.evidences]
#         with torch.no_grad():
#             evidence_embeddings = model(evidence_texts)
#             evidence_embeddings_np = evidence_embeddings.cpu().numpy()

#         # Save embeddings into evidence objects
#         for ev, emb in zip(claim.evidences, evidence_embeddings_np):
#             ev.embedding = emb

#         # print(f"Number of Evidence(s): {evidence_embeddings_np.shape}")

#         # Cluster evidences
#         n_clusters = min(len(evidence_texts), 5)
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         cluster_labels = kmeans.fit_predict(evidence_embeddings_np)

#         metrics = evaluator.evaluate_claim(
#             claim_id=claim.claim_id,
#             embeddings=evidence_embeddings,
#             cluster_labels=cluster_labels,
#         )

#         # Update the main claims progress
#         progress.update(claims_task, advance=1)

#         # -------------------------------------------------------------------------
#         # Print clusters
#         print_evidence_clusters(claim, cluster_labels)

#         # -------------------------------------------------------------------------
#         # Calculate clustering metrics
#         # Group evidences
#         claim.clustered_evidences = {
#             i: [ev for ev, label in zip(claim.evidences, cluster_labels) if label == i]
#             for i in range(n_clusters)
#         }

#         # Visualize clusters
#         plot_path = os.path.join(run_dir, "evidence_clusters.png")
#         visualize_clusters(
#             evidence_embeddings_np,
#             cluster_labels,
#             f"Evidence Clusters for Claim {example_claim.claim_id}",
#             plot_path,
#         )

#         # Save results
#         results = {
#             "claim": {
#                 "claim_id": claim.claim_id,
#                 "content": claim.content,
#                 "label": claim.label,
#                 "explanation": claim.explanation,
#                 "clustered_evidences": {
#                     str(key): [
#                         {"evidence_id": ev.evidence_id, "content": ev.content}
#                         for ev in value
#                     ]
#                     for key, value in claim.clustered_evidences.items()
#                 },
#             },
#             "training_metrics": metrics,
#         }
#         # -------------------------------------------------------------------------

#         with open(os.path.join(run_dir, "results.json"), "w") as f:
#             json.dump(results, f, indent=4)

#         logging.info(f"Results saved in {run_dir}")

#         # Get aggregate metrics
#         aggregate_metrics = evaluator.get_aggregate_metrics()

#         # Get detailed report for all claims
#         detailed_report = evaluator.get_detailed_report()

#         # Save all results
#         evaluator.save_results(
#             output_path="./clustering_results",
#             experiment_name="umc_clustering_evaluation",
#         )

#     except Exception as e:
#         logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
#         raise


# if __name__ == "__main__":
#     main()
