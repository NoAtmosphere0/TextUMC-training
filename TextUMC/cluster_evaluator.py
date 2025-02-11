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
