"""Evaluation script for TextUMC model"""

from model import (
    TextUMC,
    Claim,
    Evidence,
    visualize_clusters,
    print_evidence_clusters,
    ClusteringEvaluator,
)
import json
import torch
import logging
import os
from datetime import datetime
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.console import Console
from rich.logging import RichHandler
from sklearn.cluster import KMeans
from typing import List, Dict

# Initialize console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=console)]
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
    # Setup paths
    model_path = "TextUMC/outputs/20250116_113631/textumc_model.pt"
    output_dir = "eval_outputs"
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    try:
        torch.cuda.empty_cache()
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TextUMC().to(device)
        model.load_state_dict(torch.load(model_path))

        # Load evaluation data
        data = json.load(open("dataset/LIAR-RAW/test.json"))
        eval_claims = load_claims(data, 100, 100)  # Next 100 claims
        logging.info(f"Loaded {len(eval_claims)} claims for evaluation")

        # Initialize evaluator
        evaluator = ClusteringEvaluator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:

            eval_progress = progress.add_task(
                "[cyan]Evaluating claims...", total=len(eval_claims)
            )

            # Evaluate each claim
            for claim in eval_claims:
                # Get embeddings
                evidence_texts = [ev.content for ev in claim.evidences]
                with torch.no_grad():
                    evidence_embeddings = model(evidence_texts)
                    evidence_embeddings_np = evidence_embeddings.cpu().numpy()

                # Save embeddings
                # for ev, emb in zip(claim.evidences, evidence_embeddings_np):
                #     ev.embedding = emb

                # Cluster evidences
                n_clusters = min(len(evidence_texts), 5)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(evidence_embeddings_np)

                # Evaluate clustering
                metrics = evaluator.evaluate_claim(
                    claim_id=claim.claim_id,
                    embeddings=evidence_embeddings,
                    cluster_labels=cluster_labels,
                )

                print(metrics)

                # # Visualize clusters
                # plot_path = os.path.join(
                #     run_dir, f"claim_{claim.claim_id}_clusters.png"
                # )
                # visualize_clusters(
                #     evidence_embeddings_np,
                #     cluster_labels,
                #     f"Evidence Clusters for Claim {claim.claim_id}",
                #     plot_path,
                # )

                # Print cluster assignments
                # print_evidence_clusters(claim, cluster_labels)

                progress.update(eval_progress, advance=1)

        # Save evaluation results
        metrics_file = os.path.join(run_dir, "evaluation_metrics.json")
        aggregate_metrics = evaluator.get_aggregate_metrics()
        detailed_report = evaluator.get_detailed_report()

        evaluator.save_results(
            output_path=run_dir, experiment_name="evaluation_results"
        )

        console.print("[green]Evaluation completed successfully!")

    except Exception as e:
        logging.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
