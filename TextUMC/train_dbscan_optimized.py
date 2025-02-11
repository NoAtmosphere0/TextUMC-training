"""Optimized training script for TextUMC model with DBSCAN clustering"""

from sklearn.decomposition import PCA
from model_dbscan import TextUMC, Claim, Evidence, umc_train, ClusteringEvaluator
import json
import torch
import logging
import os
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
import argparse
from typing import List, Dict
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler
import numpy as np
from sklearn.preprocessing import StandardScaler
from model2 import evaluate_claim


# Initialize console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console)],
)


def load_claims(data: Dict, start_idx: int, count: int) -> List[Claim]:
    """Load a specified number of claims starting from given index"""
    claims = []
    for idx in range(start_idx, min(start_idx + count, len(data))):
        claim_data = data[idx]

        # Skip claims with too few evidences
        if len(claim_data["reports"]) < 5:
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


def optimize_dbscan_params(
    model: TextUMC,
    evidences: List[Evidence],
    device: torch.device,
    n_components: float = 0.5,
):
    """Find optimal DBSCAN parameters using grid search"""
    # Get embeddings
    with torch.no_grad():
        texts = [ev.content for ev in evidences]
        embeddings = model(texts).cpu().numpy()

    # Apply PCA dimension reduction
    pca = PCA(n_components=min(n_components, embeddings.shape[1], embeddings.shape[0]))
    embeddings_reduced = pca.fit_transform(embeddings)

    # Normalize reduced embeddings
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings_reduced)

    # Grid search parameters
    eps_range = np.linspace(0.1, 2.0, 20)
    min_samples_range = range(2, min(20, len(evidences)))

    best_score = -float("inf")
    best_params = {"eps": 0.5, "min_samples": 5}

    for eps in eps_range:
        for min_samples in min_samples_range:
            # Update model config
            model.config.eps = eps
            model.config.min_samples = min_samples

            # Perform clustering
            try:
                cluster_labels, metrics = model.cluster_and_evaluate(
                    torch.tensor(embeddings_normalized).to(device),
                    eps=eps,
                    min_samples=min_samples,
                )

                # Skip if only one cluster or all points are noise
                if len(np.unique(cluster_labels)) < 2:
                    continue

                # Calculate combined score
                silhouette = metrics.get("silhouette", 0)

                if silhouette > best_silhouette:
                    best_score = combined_score
                    best_params = {"eps": eps, "min_samples": min_samples}

            except Exception as e:
                continue

    return best_params


def main():
    parser = argparse.ArgumentParser(
        description="Train TextUMC model with optimized DBSCAN clustering"
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=50,
        help="Number of PCA components for dimension reduction",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs", type=float, default=10.0, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Initial learning rate"
    )
    parser.add_argument(
        "--save_after_n_epochs", type=int, default=1, help="Save model every N epochs"
    )
    parser.add_argument(
        "--warmup_epochs", type=float, default=2.0, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for optimizer"
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = "outputs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        # Load data
        data = json.load(open("dataset/LIAR-RAW/test.json"))
        train_claims = load_claims(data, 0, 100)  # First 100 claims
        eval_claims = load_claims(data, 100, 100)  # Next 100 claims

        logging.info(
            f"Loaded {len(train_claims)} training claims and {len(eval_claims)} evaluation claims"
        )

        # Initialize model and move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TextUMC().to(device)

        # Initialize evaluator
        evaluator = ClusteringEvaluator()

        # Collect all evidences for training
        all_evidences = []
        for claim in train_claims:
            all_evidences.extend(claim.evidences)

        # Find optimal DBSCAN parameters with PCA
        logging.info(
            "Finding optimal DBSCAN parameters with PCA dimension reduction..."
        )
        best_params = optimize_dbscan_params(
            model, all_evidences, device, args.pca_components
        )
        logging.info(
            f"Optimal parameters: eps={best_params['eps']:.3f}, min_samples={best_params['min_samples']}"
        )

        # Update model config with optimal parameters
        model.config.eps = best_params["eps"]
        model.config.min_samples = best_params["min_samples"]

        # Create tensorboard log directory
        log_dir = os.path.join(run_dir, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)

        # Initialize gradient scaler for mixed precision training
        scaler = GradScaler()

        # Train model with optimizations
        trained_evidences, metrics = umc_train(
            model=model,
            evidences=all_evidences,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            log_dir=log_dir,
            save_after_n_epochs=args.save_after_n_epochs,
            warmup_epochs=args.warmup_epochs,
            weight_decay=args.weight_decay,
            scaler=scaler,  # Enable mixed precision training
        )

        # Save final model
        model_path = os.path.join(run_dir, "textumc_dbscan_model.pt")
        model.save_model(model_path)
        logging.info(f"Model saved to {model_path}")

        # Evaluate model on eval claims
        logging.info("Evaluating model on test claims...")
        for claim in eval_claims:
            claim, _ = evaluate_claim(claim, model, run_dir, evaluator)

        # Save evaluation results
        evaluator.save_results(output_path=run_dir, experiment_name="dbscan_evaluation")

        # Print final metrics
        metrics = evaluator.get_aggregate_metrics()
        console.print("\n[bold]Final Metrics:[/bold]")
        for metric, value in metrics.items():
            console.print(f"{metric}: {value:.4f}")

        console.print("\n[green]Training and evaluation completed successfully!")
        console.print(f"\nResults saved in: {run_dir}")
        console.print("\nTo train with different parameters, use arguments like:")
        console.print(
            "python train_dbscan_optimized.py --batch_size 64 --num_epochs 15 --learning_rate 1e-5 --warmup_epochs 3"
        )

    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
