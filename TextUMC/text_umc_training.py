import argparse
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from model import (
    TextUMC,
    Claim,
    Evidence,
    umc_train,
    print_evidence_clusters,
    visualize_clusters,
    ClusteringEvaluator,
)
import json
import torch
import logging
from rich.console import Console
from sklearn.cluster import KMeans
import os
from datetime import datetime
from typing import List, Dict, Tuple, Any
from tqdm import tqdm, trange
from model import *

# Initialize console and logging
console = Console()


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


# OG training
# def train_normal(
#     model: TextUMC, train_claims: List[Claim], args: argparse.Namespace, run_dir: str
# ) -> TextUMC:
#     """Train single model on all claims"""

#     # Collect all evidences
#     all_evidences = []
#     for claim in train_claims:
#         all_evidences.extend(claim.evidences)

#     # Create tensorboard log directory
#     log_dir = os.path.join(run_dir, "tensorboard", "unified")
#     os.makedirs(log_dir, exist_ok=True)
#     writer = SummaryWriter(log_dir)

#     # Initialize optimizer if not already initialized
#     if model.optimizer is None:
#         model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

#     # Training loop
#     for epoch in range(int(args.num_epochs)):
#         total_loss = 0

#         # Process in batches
#         for i in tqdm(range(0, len(all_evidences), args.batch_size)):
#             batch = all_evidences[i : i + args.batch_size]
#             batch_texts = [ev.content for ev in batch]

#             # Get embeddings
#             embeddings = model(batch_texts)

#             # Simple MSE loss for demonstration
#             loss = torch.mean((embeddings - embeddings.mean(0)) ** 2)

#             model.optimizer.zero_grad()
#             loss.backward()
#             model.optimizer.step()

#             total_loss += loss.item()

#         # Log epoch loss
#         writer.add_scalar("Loss/train", total_loss, epoch)

#         if args.save_after_n_epochs > 0 and (epoch + 1) % args.save_after_n_epochs == 0:
#             model_path = os.path.join(run_dir, f"model_epoch_{epoch+1}.pt")
#             torch.save(model.state_dict(), model_path)

#     writer.close()
#     return model


# Curricular training
def train_normal(
    model: TextUMC, train_claims: List[Claim], args: argparse.Namespace, run_dir: str
) -> TextUMC:
    """Train single model on all claims using curriculum learning"""

    # Collect all evidences
    all_evidences = []
    for claim in train_claims:
        all_evidences.extend(claim.evidences)

    # Create tensorboard log directory
    log_dir = os.path.join(run_dir, "tensorboard", "unified")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Initialize optimizer if not already initialized
    if model.optimizer is None:
        model.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Sort evidences by length (shorter texts first as they're typically easier)
    sorted_evidences = sorted(all_evidences, key=lambda x: len(x.content.split()))

    # Define curriculum stages
    num_stages = 4
    samples_per_stage = len(sorted_evidences) // num_stages

    # Training loop with curriculum
    for stage in range(num_stages):
        logging.info(f"Starting curriculum stage {stage+1}/{num_stages}")

        # Get progressively more data for each stage
        current_evidences = sorted_evidences[: (stage + 1) * samples_per_stage]

        # Train on current curriculum stage
        for epoch in range(int(args.num_epochs // num_stages)):
            total_loss = 0

            # Process in batches
            for i in tqdm(range(0, len(current_evidences), args.batch_size)):
                batch = current_evidences[i : i + args.batch_size]
                batch_texts = [ev.content for ev in batch]

                # Get embeddings
                embeddings = model(batch_texts)

                # Get augmented views for contrastive learning
                z1, z2 = model.augment_views(embeddings)
                aug_embeddings = torch.cat([z1, z2], dim=0)

                # Contrastive loss instead of simple MSE
                loss = unsupervised_contrastive_loss(aug_embeddings, temp=0.2)

                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()

                total_loss += loss.item()

            # Log epoch loss for current stage
            avg_loss = total_loss / (len(current_evidences) / args.batch_size)
            writer.add_scalar(f"Loss/train/stage_{stage+1}", avg_loss, epoch)

            # Save model checkpoint
            if (
                args.save_after_n_epochs > 0
                and (epoch + 1) % args.save_after_n_epochs == 0
            ):
                model_path = os.path.join(
                    run_dir, f"model_stage{stage+1}_epoch_{epoch+1}.pt"
                )
                torch.save(model.state_dict(), model_path)

    writer.close()
    return model
