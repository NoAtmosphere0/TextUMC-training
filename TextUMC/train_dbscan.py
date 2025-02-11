"""Training script for TextUMC model with DBSCAN clustering"""

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


def main():
    parser = argparse.ArgumentParser(
        description="Train TextUMC model with DBSCAN clustering"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--num_epochs", type=float, default=5.0, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--save_after_n_epochs", type=int, default=0, help="Save model every N epochs"
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
        train_claims = load_claims(data, 0, 300)  # First 100 claims
        eval_claims = load_claims(data, 100, 100)  # Next 100 claims

        logging.info(
            f"Loaded {len(train_claims)} training claims and {len(eval_claims)} evaluation claims"
        )

        # Initialize model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TextUMC().to(device)

        # Initialize evaluator
        evaluator = ClusteringEvaluator()

        # Collect all evidences for training
        all_evidences = []
        for claim in train_claims:
            all_evidences.extend(claim.evidences)

        # Create tensorboard log directory
        log_dir = os.path.join(run_dir, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)

        # Train model
        trained_evidences, metrics = umc_train(
            model=model,
            evidences=all_evidences,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            log_dir=log_dir,
            save_after_n_epochs=args.save_after_n_epochs,
        )

        # Save final model
        model_path = os.path.join(run_dir, "textumc_dbscan_model.pt")
        model.save_model(model_path)
        logging.info(f"Model saved to {model_path}")

        # Evaluate model on eval claims
        for claim in eval_claims:
            claim, _ = evaluate_claim(claim, model, run_dir, evaluator)

        # Save evaluation results
        evaluator.save_results(output_path=run_dir, experiment_name="dbscan_evaluation")

        console.print("[green]Training and evaluation completed successfully!")
        console.print(f"\nResults saved in: {run_dir}")
        console.print("\nTo train with different parameters, use arguments like:")
        console.print(
            f"python train_dbscan.py --batch_size {args.batch_size} --num_epochs {args.num_epochs} --learning_rate {args.learning_rate}"
        )

    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
