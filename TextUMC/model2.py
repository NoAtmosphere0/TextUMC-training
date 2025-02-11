"""Train TextUMC model on first 100 claims and evaluate on next 100 claims"""

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


import argparse


def find_optimal_clusters(
    embeddings: torch.Tensor, min_clusters: int = 2, max_clusters: int = 10
) -> int:
    """Find optimal number of clusters using silhouette score"""
    from sklearn.metrics import silhouette_score

    embeddings_np = embeddings.detach().cpu().numpy()

    best_score = -1
    best_n_clusters = min_clusters

    for n in range(min_clusters, min(max_clusters + 1, len(embeddings_np))):
        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_np)

        if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
            score = silhouette_score(embeddings_np, labels)
            if score > best_score:
                best_score = score
                best_n_clusters = n

    return best_n_clusters


def train_demonic(
    model: TextUMC,
    train_claims: List[Claim],
    args: argparse.Namespace,
    run_dir: str,
    evaluator: ClusteringEvaluator,
    eval_claims: List[Claim],
    evaluated_claims: List[Claim],
) -> Dict[str, TextUMC]:
    """Train separate model for each claim"""
    # Create base tensorboard directory
    log_base_dir = os.path.join(run_dir, "tensorboard", "demonic")
    os.makedirs(log_base_dir, exist_ok=True)

    # eval_claims = eval_claims[: int(args.num_epochs * len(eval_claims))]

    for claim in tqdm(eval_claims, desc="Training claims"):
        # Create separate log directory for each claim
        claim_log_dir = os.path.join(log_base_dir, f"claim_{claim.claim_id}")

        model = TextUMC().to(model.device)

        evidence_text = [ev.content for ev in claim.evidences]

        batch_size = min(args.batch_size, len(evidence_text))
        num_clusters = min(5, len(evidence_text))

        if model.optimizer is None:
            model.optimizer = torch.optim.Adam(
                model.parameters(), lr=args.learning_rate
            )

        metrics = {
            "epoch_losses": [],
            "unsup_losses": [],
            "sup_losses": [],
            "batch_losses": [],
        }

        current_centroids = None
        # training loop
        try:
            pbar = trange(int(args.num_epochs), desc="Training")
            for epoch in pbar:
                pbar.set_description(
                    f"Training claim {claim.claim_id} at epoch {epoch}"
                )
                epoch_loss = 0
                unsup_loss = 0
                sup_loss = 0

                for i in range(0, len(evidence_text), batch_size):
                    batch_texts = evidence_text[i : i + batch_size]
                    embeddings = model(batch_texts)

                    z1, z2 = model.augment_views(embeddings)
                    aug_embeddings = torch.cat([z1, z2], dim=0)

                    unsup_loss = unsupervised_contrastive_loss(aug_embeddings, temp=0.5)

                    embeddings_np = embeddings.detach().cpu().numpy()

                    kmeans = KMeans(
                        n_clusters=5,
                        init=(
                            current_centroids
                            if current_centroids is not None
                            else "k-means++"
                        ),
                        n_init=5,
                        tol=1e-5,
                        max_iter=500,
                    )

                    kmeans.fit(embeddings_np)

                    _ = kmeans.labels_
                    pseudo_labels = torch.tensor(kmeans.labels_).to(model.device)
                    current_centroids = kmeans.cluster_centers_

                    sup_loss = supervised_contrastive_loss(
                        embeddings, pseudo_labels, temp=0.5
                    )

                    loss = unsup_loss + sup_loss

                    model.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    model.optimizer.step()

                    epoch_loss += loss.item()

                    metrics["batch_losses"].append(loss.item())
                    metrics["unsup_losses"].append(unsup_loss.item())
                    metrics["sup_losses"].append(sup_loss.item())

                metrics["epoch_losses"].append(epoch_loss)
                # logging.info(f"Epoch {epoch} loss: {epoch_loss}")

        except Exception as e:
            logging.error(f"Training failed for claim {claim.claim_id}: {str(e)}")

        # Evaluate claim, if error then run again
        # try:
        claim, claim_metrics = evaluate_claim(claim, model, run_dir, evaluator)
        evaluated_claims.append(claim)
        # except Exception as e:
        #     logging.error(f"Evaluation failed for claim {claim.claim_id}: {str(e)}")
        #     continue

    return evaluated_claims


def train_normal(
    model: TextUMC, train_claims: List[Claim], args: argparse.Namespace, run_dir: str
) -> TextUMC:
    """Train single model on all claims"""
    all_evidences = []
    for claim in train_claims:
        all_evidences.extend(claim.evidences)

    # Create tensorboard log directory for unified model
    log_dir = os.path.join(run_dir, "tensorboard", "unified")
    os.makedirs(log_dir, exist_ok=True)

    _, metrics = umc_train(
        model=model,
        evidences=all_evidences,
        num_clusters=3,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        log_dir=log_dir,
        save_after_n_epochs=args.save_after_n_epochs,
    )

    return model


# def evaluate_claim(
#     claim: Claim, model: TextUMC, run_dir: str, evaluator: ClusteringEvaluator
# ) -> Tuple[Claim, Dict[str, Any]]:
#     """Evaluate clustering for a single claim and return results"""
#     evidence_texts = [ev.content for ev in claim.evidences]
#     with torch.no_grad():
#         evidence_embeddings = model(evidence_texts)
#         evidence_embeddings_np = evidence_embeddings.cpu().numpy()

#     # Grid search for optimal number of clusters
#     best_score = float("-inf")
#     best_n_clusters = 3
#     best_labels = None
#     cluster_metrics = {}

#     # Try different numbers of clusters
#     for n_clusters in range(3, min(7, len(evidence_texts))):
#         try:
#             # Run KMeans
#             kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)
#             labels = kmeans.fit_predict(evidence_embeddings_np)

#             # Calculate cluster quality metrics
#             silhouette = silhouette_score(evidence_embeddings_np, labels)
#             calinski = calinski_harabasz_score(evidence_embeddings_np, labels)
#             davies = davies_bouldin_score(evidence_embeddings_np, labels)

#             # Normalize and combine scores
#             combined_score = (
#                 silhouette  # Already in [-1,1]
#                 + calinski / (calinski + 100)  # Normalize large values
#                 + 1 / (1 + davies)  # Invert and normalize
#             ) / 3  # Average

#             cluster_metrics[n_clusters] = {
#                 "silhouette": silhouette,
#                 "calinski": calinski,
#                 "davies": davies,
#                 "combined": combined_score,
#             }

#             if combined_score > best_score:
#                 best_score = combined_score
#                 best_n_clusters = n_clusters
#                 best_labels = labels

#         except ValueError as e:
#             continue

#     # Log metrics
#     # logging.info(f"Claim {claim.claim_id} cluster metrics: {cluster_metrics}")
#     # logging.info(f"Selected optimal number of clusters: {best_n_clusters}")

#     # Get evaluation metrics using the best clustering
#     metrics = evaluator.evaluate_claim(
#         claim_id=claim.claim_id,
#         embeddings=evidence_embeddings,
#         cluster_labels=best_labels,
#     )

#     # Add clustering metrics to the results
#     # metrics["cluster_metrics"] = cluster_metrics
#     # metrics["optimal_n_clusters"] = best_n_clusters

#     # Group evidences by cluster
#     claim.clustered_evidences = {}

#     for ev, label in zip(claim.evidences, best_labels):
#         if str(label) not in claim.clustered_evidences:
#             claim.clustered_evidences[str(label)] = []
#         claim.clustered_evidences[str(label)].append(ev.content)

#     return claim, metrics


def evaluate_claim(
    claim: Claim, model: TextUMC, run_dir: str, evaluator: ClusteringEvaluator
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
    for n_clusters in range(3, min(10, len(evidence_texts))):
        try:
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=10,
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
        "--num_epochs", type=float, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-5, help="Learning rate for training"
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
        default=0,
        help="Save model after every n epochs",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="tensorboard",
        help="Directory for tensorboard logs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory for saving model checkpoints and evaluation results",
    )
    parser.add_argument(
        "--eval_only",
        type=bool,
        default=False,
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
        type=bool,
        default=False,
        help="Visualize clusters using PCA and TSNE",
    )
    args = parser.parse_args()

    try:
        # Clear gpu cache
        torch.cuda.empty_cache()

        # Load training and evaluation claims
        train_claims = load_claims(train_data, 0, args.n_train_claims)
        eval_claims = load_claims(test_data, 0, args.n_eval_claims)

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

        # Uncomment to test with a single claim
        # eval_claims = [example_claim]

        logging.info(
            f"Loaded {len(train_claims)} training claims and {len(eval_claims)} evaluation claims"
        )

        # Initialize model and move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_config = ModelConfig()
        model = TextUMC(config=model_config).to(device)

        # Initialize evaluator
        evaluator = ClusteringEvaluator()

        evaluated_claims = []

        if args.eval_only and args.train_type == "normal":
            if not args.model_checkpoint:
                raise ValueError(
                    "Model checkpoint path is required for evaluation only"
                )
            model.load_state_dict(torch.load(args.model_checkpoint))

            for claim in tqdm(eval_claims, desc="Evaluating claims"):
                claim, _ = evaluate_claim(claim, model, run_dir, evaluator)
                evaluated_claims.append(claim)
        else:
            if args.num_epochs < 1:
                eval_claims = eval_claims[: int(args.num_epochs * len(eval_claims))]
            # Training Phase
            if args.train_type == "demonic":
                evaluated_claims = train_demonic(
                    model,
                    train_claims,
                    args,
                    run_dir,
                    evaluator=evaluator,
                    eval_claims=eval_claims,
                    evaluated_claims=evaluated_claims,
                )
            elif args.train_type == "normal":
                model = train_normal(model, train_claims, args, run_dir)

                # Save model
                model_path = os.path.join(run_dir, "textumc_model.pt")
                torch.save(model.state_dict(), model_path)
                logging.info(f"Model saved to {model_path}")

                for claim in tqdm(eval_claims, desc="Evaluating claims"):
                    claim, _ = evaluate_claim(claim, model, run_dir, evaluator)
                    evaluated_claims.append(claim)
            else:
                raise ValueError(f"Invalid training type: {args.train_type}")

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
                )
            # Get final metrics
            aggregate_metrics = evaluator.get_aggregate_metrics()
            detailed_report = evaluator.get_detailed_report()

            # Visualize clusters
            if args.visualize_clusters:
                for claim in evaluated_claims:
                    with torch.no_grad():
                        evidence_embeddings = model(
                            [ev.content for ev in claim.evidences]
                        )
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

        console.print("[green]Pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
