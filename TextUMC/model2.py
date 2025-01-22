"""Train TextUMC model on first 100 claims and evaluate on next 100 claims"""

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
from typing import List, Dict

# Initialize console and logging
console = Console()


def load_claims(data: Dict, start_idx: int, count: int) -> List[Claim]:
    """Load a specified number of claims starting from given index"""
    claims = []
    for idx in range(start_idx, min(start_idx + count, len(data))):
        claim_data = data[idx]

        # Skip claims with too few evidences
        if len(claim_data["reports"]) <= 5:
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


import argparse


def train_demonic(model, train_claims, args):
    """Train separate model for each claim"""
    models = {}

    for claim in train_claims:

        claim_model = TextUMC().to(model.device)
        _, metrics = umc_train(
            model=claim_model,
            evidences=claim.evidences,
            num_clusters=min(len(claim.evidences), 5),
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )

        models[claim.claim_id] = claim_model

    return models


def train_normal(model, train_claims, args):
    """Train single model on all claims"""
    all_evidences = []
    for claim in train_claims:
        all_evidences.extend(claim.evidences)

    _, metrics = umc_train(
        model=model,
        evidences=all_evidences,
        num_clusters=3,
        batch_size=-1,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )

    return model


def evaluate_claim(claim, model, run_dir, evaluator):
    """Evaluate clustering for a single claim and return results"""
    evidence_texts = [ev.content for ev in claim.evidences]
    with torch.no_grad():
        evidence_embeddings = model(evidence_texts)
        evidence_embeddings_np = evidence_embeddings.cpu().numpy()

    n_clusters = min(len(evidence_texts), 5)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(evidence_embeddings_np)

    metrics = evaluator.evaluate_claim(
        claim_id=claim.claim_id,
        embeddings=evidence_embeddings,
        cluster_labels=cluster_labels,
    )

    # Group evidences by cluster
    clustered_evidences = {}
    for ev, label in zip(claim.evidences, cluster_labels):
        if label not in clustered_evidences:
            clustered_evidences[label] = []
        clustered_evidences[label].append(
            {"evidence_id": ev.evidence_id, "content": ev.content}
        )

    return {
        "claim_id": claim.claim_id,
        "content": claim.content,
        "label": claim.label,
        "explanation": claim.explanation,
        "clustered_evidences": clustered_evidences,
        "metrics": metrics,
    }


def main():
    # Setup output directory
    output_dir = "outputs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # Load data
    train_data = json.load(open("dataset/train.json"))
    test_data = json.load(open("dataset/test.json"))
    # Setup argument parser for hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_type",
        type=str,
        choices=["normal", "demonic"],
        default="normal",
        help="Training mode: normal (unified) or demonic (per-claim)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate for training"
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
    args = parser.parse_args()

    try:
        # Clear gpu cache
        torch.cuda.empty_cache()

        # Load training and evaluation claims
        train_claims = load_claims(train_data, 0, args.n_train_claims)
        eval_claims = load_claims(test_data, 0, args.n_eval_claims)

        logging.info(
            f"Loaded {len(train_claims)} training claims and {len(eval_claims)} evaluation claims"
        )

        # Initialize model and move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TextUMC().to(device)

        # Initialize evaluator
        evaluator = ClusteringEvaluator()

        # Training Phase
        if args.train_type == "demonic":
            models = train_demonic(model, train_claims, args)
        else:
            model = train_normal(model, train_claims, args)

            # Save model
            model_path = os.path.join(run_dir, "textumc_model.pt")
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model saved to {model_path}")

        # Load model
        # model.load_state_dict(torch.load(model_path))

        # Evaluation Phase

        # Store all evaluation results
        all_results = []

        # Evaluate on each claim in evaluation set
        for claim in eval_claims:
            if args.train_type == "demonic":
                claim_model = models.get(claim.claim_id, model)
                result = evaluate_claim(claim, claim_model, run_dir, evaluator)
            else:
                result = evaluate_claim(claim, model, run_dir, evaluator)
            all_results.append(result)

        # Save all results to single JSON file
        results_path = os.path.join(run_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(
                {
                    "train_type": args.train_type,
                    "results": all_results,
                    "aggregate_metrics": evaluator.get_aggregate_metrics(),
                },
                f,
                indent=2,
            )

        # Get final metrics
        aggregate_metrics = evaluator.get_aggregate_metrics()
        detailed_report = evaluator.get_detailed_report()

        # Save evaluation results
        evaluator.save_results(
            output_path=run_dir, experiment_name="train100_eval100_results"
        )

        console.print("[green]Pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
