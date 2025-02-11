"""
TextUMC: Unsupervised Text Clustering with Multi-View Contrastive Learning

This script implements TextUMC, a model for unsupervised text clustering using a multi-view contrastive learning approach.
It leverages pre-trained BERT embeddings and enhances them through contrastive learning to create meaningful clusters
of text documents. The script is designed to be flexible and supports unified model training.

# Key Components:
    - Model Architecture:
        - TextUMC (nn.Module): The core model that uses BERT to generate initial text embeddings, projects these embeddings,
            augments them to create multiple views, and applies contrastive loss for training.
        - Projection layers: A multi-layer perceptron used to project the BERT embeddings.
        - Data augmentation with dropout.
    - Clustering:
        - KMeans Clustering: The script uses KMeans to group the text embeddings into clusters.
        - Enhanced density-based clustering with optimal k-near selection: a custom method to identify the optimal k for the clustering process.
        - Diverse sample selection to guide initial cluster centroids.
    - Contrastive Learning:
        - Unsupervised Contrastive Loss: Uses SimCLR-style contrastive loss to bring different augmentations of the same text closer.
    - Training Strategies:
        - Unified Model Training (`train_normal`): Trains a single model across all available data.
    - Evaluation:
        - The model is evaluated using silhouette score, calinski harabasz score, and davies bouldin score.
        - The best number of clusters is chosen by trying multiple cluster numbers and selecting the best score.

# Usage:
    This script can be used to cluster various kinds of text data. The main execution takes place with the `main()` function.
    It offers command-line arguments to control various aspects of training and evaluation.

## Command-Line Arguments:
    --batch_size:  Batch size for training. Default is 32.
    --num_epochs: Number of training epochs. Default is 10.
    --learning_rate: Learning rate for the Adam optimizer. Default is 1e-5.
    --n_train_claims: Number of training claims to load. Default is to use all.
    --n_eval_claims: Number of evaluation claims to load. Default is to use all.
    --save_after_n_epochs:  Save the model after every n epochs. Default is 0, meaning save only at the end.
    --output_dir:  Directory for saving model checkpoints and evaluation results. Default is "outputs".
    --eval_only: If True, only performs evaluation using a specified model checkpoint. Model checkpoint path is required.
    --model_checkpoint: Path to model checkpoint for evaluation (when --eval_only is True).
    --visualize_clusters: If True, generates PCA and t-SNE cluster visualizations for every claim.

# Data Loading:
    - The data for training and evaluation is assumed to be in `train.json` and `test.json` files respectively,
      in a `dataset` subfolder of where the script is.
    - Each data point should have the format {'claim': str, 'label': int, 'explain': str, 'reports': List[Dict[str, str]]}.
    - The `reports` is a list of dictionaries with keys `report_id` and `content`.
    - The `train.json` and `test.json` should be in a format as the `LIAR-RAW/test.json` but with an array.

# Output:
    - The script will save model checkpoints (if training), evaluation metrics, and cluster visualizations to the designated output directory.
    - Evaluation metrics include: silhouette score, calinski harabasz score, and davies bouldin score.

# Dependencies:
    - torch: PyTorch for neural network operations.
    - transformers: Hugging Face Transformers for pre-trained language models.
    - scikit-learn: For clustering, dimensionality reduction, and evaluation metrics.
    - pandas: For data manipulation and output.
    - matplotlib: For creating plots.
    - numpy: For numerical operations.

# Notes:
    - Make sure the JSON datasets are in the correct format for data to load correctly.
    - The script is designed to be run in a Google Colab environment but could work on other machines too.
    - The code uses GPU if available and falls back to CPU if not.

"""

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.cuda
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    pairwise_distances,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm, trange
from transformers import BertModel, BertTokenizer

warnings.filterwarnings("ignore")


def get_current_time_utc7_basic_formatted():
    """Gets current time, applies UTC+7 offset (NO DST), and formats as %d%m%Y_%H%M%S."""

    utc_timestamp = time.time()
    utc7_offset = 7 * 3600
    utc7_timestamp = utc_timestamp + utc7_offset
    utc7_time_tuple = time.localtime(utc7_timestamp)
    formatted_time = time.strftime("%d%m%Y_%H%M%S", utc7_time_tuple)  # Formatted string
    return formatted_time


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
    clustered_evidences: Optional[Dict[str, Dict[str, List[str]]]] = None


def load_claims(data: Dict, start_idx: int, count: int) -> List[Claim]:
    """Load a specified number of claims starting from given index"""
    claims = []
    for idx in range(start_idx, min(start_idx + count, len(data))):
        claim_data = data[idx]

        # Skip claims with too few evidences
        if len(claim_data["reports"]) <= 6:
            continue

        claim = Claim(
            claim_id=str(idx),
            content=claim_data["claim"],
            label=claim_data["label"],
            explanation=claim_data["explain"],
        )

        # Load evidences
        for evidence in claim_data["reports"]:
            claim.evidences.append(
                Evidence(evidence_id=evidence["report_id"], content=evidence["content"])
            )
        claims.append(claim)

    return claims


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
    tsne = TSNE(
        n_components=2, perplexity=5, n_iter=300
    )  # Increased n_iter for more stable TSNE
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Plot PCA
    scatter1 = ax1.scatter(
        pca_embeddings[:, 0], pca_embeddings[:, 1], c=labels, cmap="viridis", alpha=0.7
    )  # Added alpha for better visualization
    ax1.set_title(f"PCA Projection - {metric_name}")
    ax1.set_xlabel("First Principal Component")
    ax1.set_ylabel("Second Principal Component")

    # Plot t-SNE
    scatter2 = ax2.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.7,
    )  # Added alpha for better visualization
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
    plt.savefig(
        save_file, bbox_inches="tight"
    )  # Added bbox_inches to prevent labels cutting off
    plt.close()


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
    hidden_dim: int = 768
    reduced_dim: int = 256
    dropout_rate: float = 0.2
    temperature: float = 0.07
    learning_rate: float = 0.001
    batch_size: int = 256


class TextUMC(nn.Module):
    """Text Unsupervised Multi-view Clustering model."""

    def __init__(self, config: ModelConfig = ModelConfig()):
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
        print(f"Tensor Debug: {name}")
        print(f"Shape: {tensor.shape}")
        print(f"Device: {tensor.device}")
        print(f"Type: {tensor.dtype}")
        print(f"Mean: {tensor.mean().item():.4f}")
        print(f"Std: {tensor.std().item():.4f}")

    @torch.amp.autocast(device_type="cuda")
    def forward(self, texts: List[str]) -> torch.Tensor:
        # Input validation
        if not texts:
            raise ValueError("Input text list is empty.")

        # Move tokenization to device for efficiency
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)

        # Disable gradient calculation during forward pass for efficiency
        with torch.no_grad():
            outputs = self.bert_encoder(**encoded_input)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Project through MLP
        x = self.projection(embeddings)
        x = F.relu(x)
        x = F.normalize(x, dim=1)

        # Handle NaN values
        if torch.isnan(x).any():
            x = torch.nan_to_num(x, nan=0.0)

        return x

    def augment_views(self, z_t):
        # Input validation
        if torch.isnan(z_t).any():
            raise ValueError("Input tensor contains NaN values.")

        # Simple dropout-based augmentation
        z1 = self.dropout1(z_t)
        z2 = self.dropout2(z_t)

        return z1, z2

    def save_model(self, path: str):
        """Save model to disk"""
        torch.save(self.state_dict(), path)

    def load_model(self, path: str):
        """Load model from disk"""
        self.load_state_dict(
            torch.load(path, map_location=self.device)
        )  # Ensure loading on correct device


def unsupervised_contrastive_loss(embeddings, temp=0.07):
    """SimCLR-style unsupervised contrastive loss"""
    batch_size = embeddings.size(0) // 2
    device = embeddings.device

    # Normalize embeddings
    norm_emb = F.normalize(embeddings, dim=1)

    # Create instance mask
    mask = torch.eye(batch_size, device=device)
    mask = mask.repeat(2, 2).bool()
    mask = ~mask

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
    """clustering and diverse sample selection"""
    # Input validation
    if torch.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaN values before clustering.")

    # Convert to numpy safely
    embeddings_np = embeddings.detach().cpu().numpy()

    # Handle potential NaN values in numpy array
    if np.isnan(embeddings_np).any():
        print(
            "Warning: NaN values detected in embeddings for clustering. Replacing with zeros."
        )
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
        n_init=1,  # Reduced n_init for efficiency in training loop, consider increasing for final evaluation
        max_iter=200,
        random_state=42,  # For reproducibility
        tol=1e-4,  # Added tolerance for convergence
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


def train_loop(
    dataloader,
    model,
    optimizer,
    num_clusters,
    temp_1,
    epoch,
    total_iterations,
    epochs,
):
    """Performs a single training epoch using only unsupervised contrastive loss."""
    model.train()  # Set model to training mode
    total_loss, unsup_loss_total = 0, 0
    num_batches = len(dataloader)
    iteration = 0  # Initialize iteration counter for the epoch

    progress_bar = tqdm(
        dataloader, desc=f"Epoch [{epoch+1}/{int(epochs)}]", unit="batch", leave=False
    )  # Set up progress bar for iterations

    for batch_idx, batch_texts in enumerate(progress_bar):
        iteration += 1  # Increment iteration counter
        current_iteration = (
            epoch * num_batches
        ) + iteration  # Calculate global iteration number

        # Forward pass
        embeddings = model(batch_texts)
        z1, z2 = model.augment_views(embeddings)
        aug_embeddings = torch.cat([z1, z2], dim=0)

        # Calculate unsupervised contrastive loss
        unsup_loss = unsupervised_contrastive_loss(aug_embeddings, temp=temp_1)

        # Combine losses (only unsupervised loss now)
        loss = unsup_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate loss
        batch_loss = loss.item()
        total_loss += batch_loss
        unsup_loss_total += unsup_loss.item()

        # Update progress bar description to show iteration and loss
        progress_bar.set_postfix(
            {
                "loss": f"{batch_loss:.4f}",
                "Unsup Loss": f"{unsup_loss.item():.4f}",
                "Iteration": f"{current_iteration}/{total_iterations}",
            }
        )

    avg_loss = total_loss / num_batches
    avg_unsup_loss = unsup_loss_total / num_batches

    print(
        f"Epoch [{epoch+1}/{int(epochs)}] Training Loss: {avg_loss:>7f} (Unsupervised: {avg_unsup_loss:>7f})"
    )
    return avg_loss, avg_unsup_loss


def eval_loop(dataloader, model, num_clusters, temp_1, epoch, epochs):
    """Performs evaluation on the validation set using only unsupervised loss."""
    model.eval()  # Set model to evaluation mode
    total_loss, unsup_loss_total = 0, 0
    num_batches = len(dataloader)

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch [{epoch+1}/{int(epochs)}] Validation",
        unit="batch",
        leave=False,
    )  # Set up progress bar for validation

    with torch.no_grad():  # Disable gradients during validation
        for batch_idx, batch_texts in enumerate(progress_bar):
            # Forward pass
            embeddings = model(batch_texts)
            z1, z2 = model.augment_views(embeddings)
            aug_embeddings = torch.cat([z1, z2], dim=0)

            # Calculate unsupervised contrastive loss
            unsup_loss = unsupervised_contrastive_loss(aug_embeddings, temp=temp_1)

            # Combine losses (only unsupervised loss)
            loss = unsup_loss

            # Accumulate loss
            batch_loss = loss.item()
            total_loss += batch_loss
            unsup_loss_total += unsup_loss.item()

            # Update progress bar description to show loss
            progress_bar.set_postfix(
                {"loss": f"{batch_loss:.4f}", "Unsup Loss": f"{unsup_loss.item():.4f}"}
            )

    avg_loss = total_loss / num_batches
    avg_unsup_loss = unsup_loss_total / num_batches

    print(
        f"Epoch [{epoch+1}/{int(epochs)}] Validation Loss: {avg_loss:>7f} (Unsupervised: {avg_unsup_loss:>7f})"
    )
    return avg_loss, avg_unsup_loss


def umc_train(
    model,
    evidences: List[Evidence],
    num_clusters: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    num_epochs: float = 10.0,  # Increased epochs to 10.0 for full epochs in progress bar
    temp_1: float = 0.2,
    save_after_n_epochs: int = 0,
    output_dir: str = "outputs",
    current_epoch: int = 0,
) -> Tuple[List[Evidence], dict]:
    """Train model using only unsupervised contrastive loss.

    metrics = {
        "epoch_losses": [],
        "unsup_losses": [],
        "val_losses": [],
        "val_unsup_losses": [],
    }
    """

    evidence_texts = [ev.content for ev in evidences]

    # Adjust parameters based on evidence count
    batch_size = min(batch_size, len(evidence_texts))
    num_clusters = min(num_clusters, len(evidence_texts))

    if num_clusters < 2:
        num_clusters = 2
        print(f"Adjusted clusters to minimum: {num_clusters}")

    # Initialize optimizer if not already initialized
    if model.optimizer is None:
        model.optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )  # Using AdamW and adding weight decay

    full_dataset = TextDataset(
        evidence_texts
    )  # Use TextDataset for better data handling

    # Split dataset into training and validation (e.g., 80/20 split)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [
            train_size,
            val_size,
        ],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True if len(train_dataset) > batch_size else False,
    )  # DataLoader with TextDataset
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )  # No shuffle for validation, no drop_last

    metrics = {
        "epoch_losses": [],
        "unsup_losses": [],
        "val_losses": [],
        "val_unsup_losses": [],
    }

    total_iterations = int(
        num_epochs * len(train_dataloader)
    )  # Calculate total iterations

    print("Starting training (Unsupervised Only)...")
    progress_bar_epoch = trange(
        int(num_epochs), desc="Training Epochs", unit="epoch"
    )  # Epoch-level progress bar

    try:
        for epoch in progress_bar_epoch:
            epoch_loss, epoch_unsup_loss = train_loop(
                train_dataloader,
                model,
                model.optimizer,
                num_clusters,
                temp_1,
                epoch,
                total_iterations,
                num_epochs,
            )

            val_loss, val_unsup_loss = eval_loop(
                val_dataloader, model, num_clusters, temp_1, epoch, num_epochs
            )

            metrics["epoch_losses"].append(epoch_loss)
            metrics["unsup_losses"].append(epoch_unsup_loss)
            metrics["val_losses"].append(val_loss)
            metrics["val_unsup_losses"].append(val_unsup_loss)

            if save_after_n_epochs > 0 and (epoch + 1) % save_after_n_epochs == 0:
                model_path = os.path.join(output_dir, f"model_epoch_{epoch + 1}.pt")
                model.save_model(model_path)
                print(f"Model saved at epoch {epoch + 1}")

        # final_model_path = os.path.join(output_dir, "model_final.pt")
        # model.save_model(final_model_path)
        # print("Final model saved")

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

    return evidences, metrics


def train_normal(
    model: TextUMC,
    train_claims: List[Claim],
    args: argparse.Namespace,
    run_dir: str,
    current_epoch: int = 0,
) -> Tuple[TextUMC, Dict[str, Any]]:
    """Train single model on all claims

    metrics = {
        "epoch_losses": [],
        "unsup_losses": [],
        "val_losses": [],
        "val_unsup_losses": [],
    }
    """
    all_evidences = []
    for claim in train_claims:
        all_evidences.extend(claim.evidences)

    print("Starting training for unified model")
    _, metrics = umc_train(
        model=model,
        evidences=all_evidences,
        num_clusters=3,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_after_n_epochs=args.save_after_n_epochs,
        output_dir=run_dir,
        current_epoch=current_epoch,
    )

    return model, metrics


def evaluate_claim(
    claim: Claim, model: TextUMC, run_dir: str, evaluator: "ClusteringEvaluator"
) -> Tuple[Claim, Dict[str, Any]]:
    """Evaluate clustering for a single claim and return results"""
    evidence_texts = [ev.content for ev in claim.evidences]
    with torch.no_grad():
        evidence_embeddings = model(evidence_texts)
        evidence_embeddings_np = evidence_embeddings.cpu().numpy()

    labels_dummy = [0] * len(evidence_texts)

    # Track best scores for each metric
    best_configs = {
        "silhouette": {"score": float("-inf"), "n_clusters": 3, "labels": labels_dummy},
        "calinski": {"score": float("-inf"), "n_clusters": 3, "labels": labels_dummy},
        "davies": {
            "score": float("inf"),
            "n_clusters": 3,
            "labels": labels_dummy,
        },  # Note: Lower is better for Davies-Bouldin
    }

    cluster_metrics = {}

    # Try different numbers of clusters
    for n_clusters in range(
        2, min(10, len(evidence_texts))
    ):  # Start from 2 clusters as 1 is trivial
        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=10,  # Increased n_init for evaluation
                max_iter=500,
                tol=1e-5,
                random_state=42,
            )
            labels = kmeans.fit_predict(evidence_embeddings_np)

            # Calculate metrics
            scores = {
                "silhouette": silhouette_score(evidence_embeddings_np, labels),
                "calinski": calinski_harabasz_score(evidence_embeddings_np, labels),
                "davies": davies_bouldin_score(evidence_embeddings_np, labels),
            }

            cluster_metrics[n_clusters] = scores

            # Update best configurations
            if scores["silhouette"] > best_configs["silhouette"]["score"]:
                best_configs["silhouette"] = {
                    "score": scores["silhouette"],
                    "n_clusters": n_clusters,
                    "labels": labels.copy(),
                }

            if scores["calinski"] > best_configs["calinski"]["score"]:
                best_configs["calinski"] = {
                    "score": scores["calinski"],
                    "n_clusters": n_clusters,
                    "labels": labels.copy(),
                }

            if scores["davies"] < best_configs["davies"]["score"]:  # Lower is better
                best_configs["davies"] = {
                    "score": scores["davies"],
                    "n_clusters": n_clusters,
                    "labels": labels.copy(),
                }

        except ValueError as e:
            continue

    # Store results for each metric
    metrics = {
        "cluster_metrics": cluster_metrics,
        "optimal_configs": best_configs,
    }

    # Group evidences by cluster of each metric
    claim.clustered_evidences = {}

    for metric, config in best_configs.items():
        claim.clustered_evidences[metric] = {}
        for ev, label in zip(claim.evidences, config["labels"]):
            if str(label) not in claim.clustered_evidences[metric]:
                claim.clustered_evidences[metric][str(label)] = []
            claim.clustered_evidences[metric][str(label)].append(ev.content)

    # Add evaluation metrics
    metrics = evaluator.evaluate_claim(
        claim_id=claim.claim_id,
        embeddings=evidence_embeddings,
        optimal_configs=best_configs,
    )

    return claim, metrics


def main() -> None:
    start_time = time.time()  # Record start time
    # Setup output directory
    output_dir = "outputs"
    run_id = get_current_time_utc7_basic_formatted()
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Load data
    train_data = json.load(open("./dataset/train.json"))
    test_data = json.load(open("./dataset/test.json"))

    # Define Hyperparameters
    learning_rate = 1e-5
    batch_size = 32
    epochs = 10
    save_after_n_epochs = 0  # Set to > 0 to save model every n epochs
    num_clusters = 3  # Number of clusters for KMeans
    temp_1 = 0.2  # Temperature for unsupervised loss
    temp_2 = 0.2  # Temperature for supervised loss

    # Setup argument parser for hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", type=int, default=batch_size, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=float, default=epochs, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=learning_rate,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--n_train_claims",
        type=int,
        default=len(train_data),
        help="Number of training claims",
    )
    parser.add_argument(
        "--n_eval_claims",
        type=int,
        default=len(test_data),
        help="Number of evaluation claims",
    )
    parser.add_argument(
        "--save_after_n_epochs",
        type=int,
        default=save_after_n_epochs,
        help="Save model after every n epochs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="Directory for saving model checkpoints and evaluation results",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",  # Use action='store_true' for boolean flags
        help="Evaluate model without training (need to provide model checkpoint)",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint for evaluation",
    )
    parser.add_argument(
        "--visualize_clusters",
        action="store_true",  # Use action='store_true' for boolean flags
        help="Visualize clusters using PCA and TSNE",
    )
    args = parser.parse_args()

    try:
        # Override hyperparameters with arguments if provided
        learning_rate = args.learning_rate
        batch_size = args.batch_size
        epochs = args.num_epochs
        save_after_n_epochs = args.save_after_n_epochs
        output_dir = args.output_dir

        # Clear gpu cache
        torch.cuda.empty_cache()

        # Load training and evaluation claims
        train_claims = load_claims(train_data, 0, args.n_train_claims)
        eval_claims = load_claims(test_data, 0, args.n_eval_claims)

        # example_data = json.load(open("./test.json"))[403]

        # example_claim = Claim(
        #     claim_id="403",
        #     content=example_data["claim"],
        #     label=example_data["label"],
        #     explanation=example_data["explain"],
        # )

        # # Load evidences
        # for evidence in example_data["reports"]:
        #     example_claim.evidences.append(
        #         Evidence(evidence_id=evidence["report_id"], content=evidence["content"])
        #     )

        # Uncomment to test with a single claim
        # eval_claims = [example_claim]

        print(
            f"Loaded {len(train_claims)} training claims and {len(eval_claims)} evaluation claims"
        )

        # Initialize model and move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = ModelConfig(
            learning_rate=args.learning_rate, batch_size=args.batch_size
        )
        model = TextUMC(config=model_config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=1e-4
        )  # Using AdamW and adding weight decay
        print(f"Model initialized on {device}")

        if args.model_checkpoint:
            try:
                checkpoint = torch.load(args.model_checkpoint)
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                metrics = checkpoint["metrics"]
                current_epoch = checkpoint["epoch"]

                print(
                    f"Model loaded from checkpoint: {args.model_checkpoint}, Epoch={current_epoch}"
                )
                print(
                    f"Training model with hyperparameters: Epochs={epochs}, Batch Size={batch_size}, Learning Rate={learning_rate}, Save After N Epochs={save_after_n_epochs}"
                )
                print(f"Metrics from model checkpoint: {metrics}")
            except Exception as e:
                raise ValueError(f"Failed to load model checkpoint: {str(e)}")

        # Initialize evaluator
        evaluator = ClusteringEvaluator()
        print("Evaluator initialized")

        evaluated_claims = []

        if args.eval_only:
            if not args.model_checkpoint:
                raise ValueError(
                    "Model checkpoint path is required for evaluation only"
                )

            for claim in tqdm(eval_claims, desc="Evaluating claims"):
                claim, _ = evaluate_claim(claim, model, run_dir, evaluator)
                evaluated_claims.append(claim)
        else:
            if args.num_epochs < 1:
                eval_claims = eval_claims[: int(args.num_epochs * len(eval_claims))]

            # Training Phase - Unified Model
            print("Training unified model on all claims")
            model, metrics = train_normal(
                model, train_claims, args, run_dir, current_epoch
            )

            # Save model
            model_path = os.path.join(run_dir, "textumc_model.pt")
            torch.save(
                {
                    "epoch": args.num_epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),  # Saving optimizer state might not be always needed for inference
                    "metrics": metrics,
                },
                model_path,
            )
            print(f"Model saved to {model_path}")

            for claim in tqdm(eval_claims, desc="Evaluating claims"):
                claim, _ = evaluate_claim(claim, model, run_dir, evaluator)
                evaluated_claims.append(claim)

        # Save all claims with their clustering results
        results_path = os.path.join(run_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(
                [
                    {
                        "claim_id": claim.claim_id,
                        "content": claim.content,
                        "label": claim.label,
                        "explanation": claim.explanation,
                        "clustered_evidences": claim.clustered_evidences,
                    }
                    for claim in evaluated_claims
                ],
                f,
                indent=4,
            )
        # Get final metrics
        aggregate_metrics = evaluator.get_aggregate_metrics()
        detailed_report = evaluator.get_detailed_report()

        # Visualize clusters
        if args.visualize_clusters:
            for claim in evaluated_claims:
                with torch.no_grad():
                    evidence_embeddings = model([ev.content for ev in claim.evidences])
                    embeddings = evidence_embeddings.cpu().numpy()

                # Get best labels for each metric
                metric_labels = {
                    "silhouette": evaluator.metrics[-1].silhouette_labels,
                    "calinski": evaluator.metrics[-1].calinski_labels,
                    "davies": evaluator.metrics[-1].davies_labels,
                }

                # Create visualizations for each metric
                for metric_name, labels in metric_labels.items():
                    visualize_clusters(
                        embeddings=embeddings,
                        labels=labels,
                        title=f"Claim_{claim.claim_id}",
                        save_path=run_dir,
                        metric_name=metric_name,
                    )

        # Save evaluation results
        evaluator.save_results(
            output_path=run_dir, experiment_name="train100_eval100_results"
        )

        print("[green]Pipeline completed successfully!")

    except Exception as e:
        print(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

    end_time = time.time()
    running_time = end_time - start_time
    hours = int(running_time // 3600)
    minutes = int((running_time % 3600) // 60)
    seconds = int(running_time % 60)
    print(f"Total running time: {hours}h:{minutes}m:{seconds}s")


@dataclass
class ClusteringMetrics:
    """Store clustering metrics for a single claim"""

    claim_id: str
    n_samples: int

    # Silhouette score metrics
    silhouette_score: float
    silhouette_n_clusters: int
    silhouette_labels: np.ndarray

    # Calinski-Harabasz metrics
    calinski_score: float
    calinski_n_clusters: int
    calinski_labels: np.ndarray

    # Davies-Bouldin metrics
    davies_score: float
    davies_n_clusters: int
    davies_labels: np.ndarray


class ClusteringEvaluator:
    def __init__(self):
        self.metrics: List[ClusteringMetrics] = []

    def evaluate_claim(
        self, claim_id: str, embeddings: torch.Tensor, optimal_configs: Dict[str, Dict]
    ) -> ClusteringMetrics:
        """Calculate clustering metrics for a single claim's embeddings"""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        metrics = ClusteringMetrics(
            claim_id=claim_id,
            n_samples=len(embeddings),
            silhouette_score=optimal_configs["silhouette"]["score"],
            silhouette_n_clusters=optimal_configs["silhouette"]["n_clusters"],
            silhouette_labels=optimal_configs["silhouette"]["labels"],
            calinski_score=optimal_configs["calinski"]["score"],
            calinski_n_clusters=optimal_configs["calinski"]["n_clusters"],
            calinski_labels=optimal_configs["calinski"]["labels"],
            davies_score=optimal_configs["davies"]["score"],
            davies_n_clusters=optimal_configs["davies"]["n_clusters"],
            davies_labels=optimal_configs["davies"]["labels"],
        )

        self.metrics.append(metrics)
        return metrics

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate statistics across all processed claims"""
        if not self.metrics:
            return {}

        df = pd.DataFrame([vars(m) for m in self.metrics])

        metrics = {
            "silhouette_score_avg": df["silhouette_score"].mean(),
            "silhouette_n_clusters_avg": df["silhouette_n_clusters"].mean(),
            "calinski_score_avg": df["calinski_score"].mean(),
            "calinski_n_clusters_avg": df["calinski_n_clusters"].mean(),
            "davies_score_avg": df["davies_score"].mean(),
            "davies_n_clusters_avg": df["davies_n_clusters"].mean(),
        }

        return metrics

    def get_detailed_report(self) -> pd.DataFrame:
        """
        Return detailed DataFrame with all metrics for all claims
        """
        return pd.DataFrame([vars(m) for m in self.metrics])

    def save_results(self, output_path: str, experiment_name: Optional[str] = None):
        """
        Save both aggregate metrics and detailed results to files
        """
        # Create unique experiment name if none provided
        if experiment_name is None:
            experiment_name = (
                f"clustering_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
            )

        # Save aggregate metrics
        if not self.metrics:
            print("No metrics to save.")
        else:
            aggregate_metrics = self.get_aggregate_metrics()
            pd.DataFrame([aggregate_metrics]).to_csv(
                f"{output_path}/{experiment_name}_aggregate_metrics.csv", index=False
            )

        # Save detailed results
        if not self.metrics:
            print("No detailed results to save.")
        else:
            self.get_detailed_report().to_csv(
                f"{output_path}/{experiment_name}_detailed_metrics.csv", index=False
            )


if __name__ == "__main__":
    main()
