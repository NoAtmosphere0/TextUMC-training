"""TextUMC model using DBSCAN for clustering"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.cluster import DBSCAN
import numpy as np
from torch.nn import functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import logging
import torch.cuda
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import os
from datetime import datetime
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
from rich.table import Table
import warnings
from .cluster_evaluator import ClusteringEvaluator
from tqdm import tqdm, trange
from typing import Any

# Install rich traceback
install(show_locals=False)

# Initialize rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)

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

    bert_model_name: str = "bert-base-uncased"
    embedding_dim: int = 768
    hidden_dim: int = 256
    reduced_dim: int = 128
    dropout_rate: float = 0.3
    temperature: float = 0.07
    eps: float = 0.5  # DBSCAN epsilon parameter
    min_samples: int = 6  # DBSCAN min_samples parameter


class TextUMC(nn.Module):
    """Text Unsupervised Multi-view Clustering model with DBSCAN."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = None

        # Initialize BERT components
        self._init_bert()

        # Initialize projection layers
        self._init_layers()

        # Optimize memory usage
        self._optimize_memory()

        self.to(self.device)

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

    def _optimize_memory(self):
        """Apply memory optimization settings."""
        self.bert_encoder.config.gradient_checkpointing = True
        torch.backends.cudnn.benchmark = True

    @torch.amp.autocast(device_type="cuda")
    def forward(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("Empty text input")

        # Process in smaller batches to save memory
        batch_size = 16
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Clear CUDA cache before BERT forward pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Run BERT with gradients and memory-efficient settings
            outputs = self.bert_encoder(
                **encoded, output_attentions=False, output_hidden_states=False
            )
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

            # Project and normalize embeddings
            x = self.projection(embeddings)
            x = F.normalize(x, dim=1)  # L2 normalization

            if torch.isnan(x).any():
                x = torch.nan_to_num(x, nan=0.0)

            all_embeddings.append(x)

            # Clear memory while maintaining gradients
            del encoded, outputs, embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all batches
        x = torch.cat(all_embeddings, dim=0)
        return x

    def augment_views(self, z_t):
        if torch.isnan(z_t).any():
            raise ValueError("NaN values in input to augment_views")

        z1 = self.dropout1(z_t)
        z2 = self.dropout2(z_t)

        return z1, z2

    def save_model(self, path: str):
        """Save model to disk"""
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        """Load model from disk"""
        self.load_state_dict(torch.load(path))


def estimate_dbscan_params(embeddings: np.ndarray) -> Tuple[float, int]:
    """Estimate good DBSCAN parameters using nearest neighbors distances"""
    # Calculate distances to k nearest neighbors
    n_neighbors = min(len(embeddings) - 1, 5)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    # Sort distances to kth nearest neighbor
    k_dist = np.sort(distances[:, -1])

    # Find elbow point for epsilon
    from kneed import KneeLocator

    kneedle = KneeLocator(
        range(len(k_dist)), k_dist, curve="convex", direction="increasing"
    )
    elbow_idx = kneedle.elbow if kneedle.elbow is not None else len(k_dist) // 2
    eps = k_dist[elbow_idx]

    # Estimate min_samples based on data density
    min_samples = max(int(np.log(len(embeddings))), 3)

    return eps, min_samples


def cluster_and_evaluate(
    embeddings: torch.Tensor,
    eps: Optional[float] = None,
    min_samples: Optional[int] = None,
    n_components: Optional[int] = None,  # Number of PCA components
) -> Tuple[np.ndarray, Dict]:
    """Perform DBSCAN clustering and evaluate results with PCA dimension reduction"""
    embeddings_np = embeddings.detach().cpu().numpy()

    # Calculate optimal number of components to preserve 50% variance
    if n_components is None:
        pca_temp = PCA(n_components=0.5)  # Keep 50% of variance
        pca_temp.fit(embeddings_np)
        n_components = pca_temp.n_components_
        del pca_temp  # Free memory

    # Apply PCA dimension reduction
    pca = PCA(
        n_components=min(n_components, embeddings_np.shape[1], embeddings_np.shape[0])
    )
    embeddings_reduced = pca.fit_transform(embeddings_np)
    explained_variance_ratio = pca.explained_variance_ratio_.sum()

    # Estimate parameters if not provided
    if eps is None or min_samples is None:
        eps, min_samples = estimate_dbscan_params(embeddings_reduced)

    # Perform DBSCAN clustering on reduced embeddings
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings_reduced)

    # Handle noise points (-1 labels)
    if -1 in labels:
        # Assign noise points to nearest cluster
        noise_mask = labels == -1
        if not all(noise_mask):  # If there are non-noise points
            non_noise_embeddings = embeddings_np[~noise_mask]
            non_noise_labels = labels[~noise_mask]

            for idx in np.where(noise_mask)[0]:
                distances = np.linalg.norm(
                    embeddings_np[idx] - non_noise_embeddings, axis=1
                )
                nearest_idx = np.argmin(distances)
                labels[idx] = non_noise_labels[nearest_idx]

    # Ensure consecutive cluster labels starting from 0
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])

    metrics = {
        "n_clusters": len(np.unique(labels)),
        "eps": eps,
        "min_samples": min_samples,
        "pca_components": n_components,
        "explained_variance_ratio": explained_variance_ratio,
    }

    return labels, metrics


def visualize_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: str,
    metric_name: str = "DBSCAN",
):
    """Plot clusters using both PCA and t-SNE side by side"""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # PCA reduction
    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)

    # Plot PCA
    scatter1 = ax1.scatter(
        pca_embeddings[:, 0], pca_embeddings[:, 1], c=labels, cmap="viridis"
    )
    ax1.set_title(f"PCA Projection - {metric_name}")
    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")

    # Add color bars
    plt.colorbar(scatter1, ax=ax1)

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
        cluster_evidences = [
            ev
            for ev, label in zip(claim.evidences, cluster_labels)
            if label == cluster_id
        ]

        for idx, evidence in enumerate(cluster_evidences):
            content = (
                evidence.content[:100] + "..."
                if len(evidence.content) > 100
                else evidence.content
            )
            table.add_row(f"Cluster {cluster_id}" if idx == 0 else "", content)
        table.add_row("", "")

    console.print(table)


def evaluate_claim(
    claim: Claim, model: TextUMC, run_dir: str, evaluator: ClusteringEvaluator
) -> Tuple[Claim, Dict[str, Any]]:
    """Evaluate clustering for a single claim using DBSCAN"""
    evidence_texts = [ev.content for ev in claim.evidences]

    # Process evidence in smaller batches to save memory
    batch_size = 16
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(evidence_texts), batch_size):
            batch_texts = evidence_texts[i : i + batch_size]
            batch_embeddings = model(batch_texts)
            all_embeddings.append(batch_embeddings)

            # Clear memory after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Concatenate all batches
        evidence_embeddings = torch.cat(all_embeddings, dim=0)

        # Apply PCA for memory efficiency during evaluation
        embeddings_np = evidence_embeddings.cpu().numpy()
        pca = PCA(n_components=0.5)  # Keep 50% of variance
        embeddings_reduced = torch.tensor(
            pca.fit_transform(embeddings_np), device=evidence_embeddings.device
        )
        del embeddings_np, evidence_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Perform DBSCAN clustering on reduced embeddings
    cluster_labels, clustering_metrics = cluster_and_evaluate(embeddings_reduced)

    # Store results for evaluation
    optimal_configs = {
        "silhouette": {
            "score": clustering_metrics.get("silhouette", 0.0),
            "n_clusters": clustering_metrics["n_clusters"],
            "labels": cluster_labels,
        },
        "calinski": {
            "score": clustering_metrics.get("calinski", 0.0),
            "n_clusters": clustering_metrics["n_clusters"],
            "labels": cluster_labels,
        },
        "davies": {
            "score": clustering_metrics.get("davies", float("inf")),
            "n_clusters": clustering_metrics["n_clusters"],
            "labels": cluster_labels,
        },
    }

    # Group evidences by cluster
    claim.clustered_evidences = {}
    for metric in ["silhouette", "calinski", "davies"]:
        claim.clustered_evidences[metric] = {}
        for ev, label in zip(claim.evidences, cluster_labels):
            if str(label) not in claim.clustered_evidences[metric]:
                claim.clustered_evidences[metric][str(label)] = []
            claim.clustered_evidences[metric][str(label)].append(ev.content)

    # Get evaluation metrics
    metrics = evaluator.evaluate_claim(
        claim_id=claim.claim_id,
        embeddings=evidence_embeddings,
        optimal_configs=optimal_configs,
    )

    return claim, metrics


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

    return F.cross_entropy(logits, labels)


def supervised_contrastive_loss(embeddings, labels, temp=0.07):
    """Supervised contrastive loss with label information"""
    batch_size = embeddings.size(0)
    device = embeddings.device

    # Normalize embeddings
    norm_emb = F.normalize(embeddings, dim=1)

    # Create label mask
    labels = torch.tensor(labels, device=device)
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float()

    # Compute similarities
    sim_matrix = torch.matmul(norm_emb, norm_emb.T)
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

    return -mean_log_prob_pos.mean()


def umc_train(
    model,
    evidences: List[Evidence],
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: float = 5.0,
    temp_1: float = 0.2,
    temp_2: float = 0.2,
    log_dir: Optional[str] = None,
    save_after_n_epochs: int = 0,
    pca_variance_ratio: float = 0.5,  # Target variance ratio for PCA
) -> Tuple[List[Evidence], dict]:
    """Train model on evidence texts using combined UMC loss with batch processing"""
    writer = None
    if log_dir:
        writer = SummaryWriter(log_dir)

    # Get evidence texts
    evidence_texts = [ev.content for ev in evidences]
    batch_size = min(batch_size, len(evidence_texts))

    # Initialize optimizer
    if model.optimizer is None:
        model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Create data loader
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(range(len(evidence_texts)), dtype=torch.long)
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    metrics = {
        "epoch_losses": [],
        "unsup_losses": [],
        "sup_losses": [],
        "batch_losses": [],
    }

    try:
        total_iterations = int(num_epochs * len(data_loader))
        iteration = 0
        current_epoch = 0
        last_saved_epoch = -1

        progress_bar = tqdm(total=total_iterations, desc="Training", unit="iteration")

        while iteration < total_iterations:
            epoch_loss = 0
            epoch_unsup_loss = 0
            epoch_sup_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(data_loader):
                if iteration >= total_iterations:
                    break

                # Clear CUDA cache before processing batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                batch_texts = [evidence_texts[i] for i in batch[0].tolist()]
                embeddings = model(batch_texts)

                # Generate augmented views first
                z1, z2 = model.augment_views(embeddings)
                aug_embeddings = torch.cat([z1, z2], dim=0)

                # Compute unsupervised loss on original embeddings
                unsup_loss = unsupervised_contrastive_loss(aug_embeddings, temp=temp_1)

                # Get cluster assignments using DBSCAN
                # We detach here since we don't need gradients for clustering
                with torch.no_grad():
                    embeddings_np = embeddings.detach().cpu().numpy()
                    pca = PCA(n_components=pca_variance_ratio)
                    embeddings_reduced = pca.fit_transform(embeddings_np)
                    cluster_labels, _ = cluster_and_evaluate(
                        torch.tensor(embeddings_reduced, device=embeddings.device)
                    )
                    del embeddings_np, embeddings_reduced

                # Compute supervised loss on original embeddings
                sup_loss = supervised_contrastive_loss(
                    embeddings, cluster_labels, temp=temp_2
                )

                # Clear memory
                del z1, z2, aug_embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                loss = unsup_loss + sup_loss

                model.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                model.optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss
                epoch_unsup_loss += unsup_loss.item()
                epoch_sup_loss += sup_loss.item()
                num_batches += 1

                metrics["batch_losses"].append(batch_loss)

                iteration += 1
                progress_bar.update(1)

                current_epoch = iteration / len(data_loader)

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
                    writer.add_scalar("Loss/batch_total", batch_loss, iteration)
                    writer.add_scalar(
                        "Loss/batch_unsupervised", unsup_loss.item(), iteration
                    )
                    writer.add_scalar(
                        "Loss/batch_supervised", sup_loss.item(), iteration
                    )

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

        if log_dir:
            final_model_path = os.path.join(log_dir, "model_final.pt")
            model.save_model(final_model_path)
            logging.info("Final model saved")

        progress_bar.close()

    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

    return evidences, metrics
