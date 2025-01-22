from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
import torch


@dataclass
class ClusteringMetrics:
    """Store clustering metrics for a single claim"""

    claim_id: str
    n_samples: int  # number of evidences
    n_clusters: int
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float


class ClusteringEvaluator:
    def __init__(self):
        self.metrics: List[ClusteringMetrics] = []

    def evaluate_claim(
        self, claim_id: str, embeddings: torch.Tensor, cluster_labels: torch.Tensor
    ) -> ClusteringMetrics:
        """
        Calculate clustering metrics for a single claim's embeddings

        Args:
            claim_id: Unique identifier for the claim
            embeddings: Tensor of shape (n_samples, n_features)
            cluster_labels: Tensor of shape (n_samples,) containing cluster assignments
        """
        # Convert tensors to numpy arrays if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        if isinstance(cluster_labels, torch.Tensor):
            cluster_labels = cluster_labels.cpu().numpy()

        # Skip evaluation if there are too few samples
        if len(embeddings) < 2 or len(np.unique(cluster_labels)) < 2:
            print(f"Skipping claim {claim_id} due to insufficient samples or clusters.")
            return ClusteringMetrics(
                claim_id=claim_id,
                n_samples=len(embeddings),
                n_clusters=len(np.unique(cluster_labels)),
                silhouette=np.nan,
                calinski_harabasz=np.nan,
                davies_bouldin=np.nan,
            )

        # Calculate metrics
        metrics = ClusteringMetrics(
            claim_id=claim_id,
            n_samples=len(embeddings),
            n_clusters=len(np.unique(cluster_labels)),
            silhouette=silhouette_score(embeddings, cluster_labels),
            calinski_harabasz=calinski_harabasz_score(embeddings, cluster_labels),
            davies_bouldin=davies_bouldin_score(embeddings, cluster_labels),
        )

        self.metrics.append(metrics)
        return metrics

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        Calculate aggregate statistics across all processed claims

        Returns dictionary with various aggregate metrics and their values
        """
        if not self.metrics:
            return {}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame([vars(m) for m in self.metrics])

        # Calculate weighted averages based on number of samples
        metrics = {
            "silhouette_avg": np.average(
                df["silhouette"],
                # weights=df["n_samples"],
                # where=~np.isnan(df["silhouette"]),
            ),
            "calinski_harabasz_avg": np.average(
                df["calinski_harabasz"],
                # weights=df["n_samples"],
                # where=~np.isnan(df["calinski_harabasz"]),
            ),
            "davies_bouldin_avg": np.average(
                df["davies_bouldin"],
                # weights=df["n_samples"],
                # where=~np.isnan(df["davies_bouldin"]),
            ),
        }

        # Calculate additional statistics
        basic_stats = {
            "total_claims": len(df),
            "total_samples": df["n_samples"].sum(),
            "avg_samples_per_claim": df["n_samples"].mean(),
            "avg_clusters_per_claim": df["n_clusters"].mean(),
            "silhouette_std": df["silhouette"].std(),
            "calinski_harabasz_std": df["calinski_harabasz"].std(),
            "davies_bouldin_std": df["davies_bouldin"].std(),
        }

        return {**metrics, **basic_stats}

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
