"""Results aggregation utilities."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import statistics

from .manager import ExperimentResult, ResultsManager
from ..utils.language_utils import LANGUAGE_CODES, get_language_name
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ResultsAggregator:
    """Aggregator for combining and analyzing experiment results."""

    def __init__(self, results: List[ExperimentResult]):
        """
        Initialize aggregator.

        Args:
            results: List of experiment results to aggregate
        """
        self.results = results

    def aggregate_by_language(
        self,
        metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate results by target language.

        Args:
            metrics: Metrics to aggregate

        Returns:
            Dict mapping languages to aggregated metrics
        """
        by_lang = defaultdict(list)

        for result in self.results:
            lang = result.target_lang
            if lang:
                by_lang[lang].append(result)

        aggregated = {}
        for lang, lang_results in by_lang.items():
            aggregated[lang] = self._aggregate_metrics(lang_results, metrics)
            aggregated[lang]["language_name"] = get_language_name(lang)
            aggregated[lang]["num_experiments"] = len(lang_results)

        return aggregated

    def aggregate_by_model(
        self,
        metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate results by model.

        Args:
            metrics: Metrics to aggregate

        Returns:
            Dict mapping models to aggregated metrics
        """
        by_model = defaultdict(list)

        for result in self.results:
            model = result.model_name or "unknown"
            by_model[model].append(result)

        aggregated = {}
        for model, model_results in by_model.items():
            aggregated[model] = self._aggregate_metrics(model_results, metrics)
            aggregated[model]["num_experiments"] = len(model_results)

        return aggregated

    def aggregate_by_domain(
        self,
        metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate results by scientific domain.

        Args:
            metrics: Metrics to aggregate

        Returns:
            Dict mapping domains to aggregated metrics
        """
        by_domain = defaultdict(list)

        for result in self.results:
            domain = result.metadata.get("domain", "unknown")
            by_domain[domain].append(result)

        aggregated = {}
        for domain, domain_results in by_domain.items():
            aggregated[domain] = self._aggregate_metrics(domain_results, metrics)
            aggregated[domain]["num_experiments"] = len(domain_results)

        return aggregated

    def _aggregate_metrics(
        self,
        results: List[ExperimentResult],
        metrics: List[str],
    ) -> Dict[str, Any]:
        """Aggregate metrics from multiple results."""
        aggregated = {}

        for metric in metrics:
            values = []
            for result in results:
                value = result.metrics.get(metric)
                if value is not None:
                    values.append(value)

            if values:
                aggregated[metric] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return aggregated

    def get_best_results(
        self,
        metric: str = "ssa_comet",
        n: int = 5,
    ) -> List[Tuple[ExperimentResult, float]]:
        """
        Get top N results by a metric.

        Args:
            metric: Metric to rank by
            n: Number of results to return

        Returns:
            List of (result, score) tuples
        """
        scored = []
        for result in self.results:
            score = result.metrics.get(metric)
            if score is not None:
                scored.append((result, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def compare_models(
        self,
        metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
        group_by: str = "lang_pair",
    ) -> Dict[str, Any]:
        """
        Compare models across different groupings.

        Args:
            metrics: Metrics to compare
            group_by: How to group comparisons ("lang_pair", "language", "domain")

        Returns:
            Comparison results
        """
        # Group results
        groups = defaultdict(lambda: defaultdict(list))

        for result in self.results:
            if group_by == "lang_pair":
                key = result.lang_pair
            elif group_by == "language":
                key = result.target_lang
            elif group_by == "domain":
                key = result.metadata.get("domain", "unknown")
            else:
                key = "all"

            model = result.model_name or "unknown"
            groups[key][model].append(result)

        # Compare within each group
        comparisons = {}
        for group_key, model_results in groups.items():
            comparisons[group_key] = {}

            for model, results in model_results.items():
                agg = self._aggregate_metrics(results, metrics)
                comparisons[group_key][model] = {
                    metric: agg.get(metric, {}).get("mean")
                    for metric in metrics
                }

        return comparisons

    def statistical_significance(
        self,
        result1: ExperimentResult,
        result2: ExperimentResult,
        metric: str = "bleu",
        num_samples: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Compute statistical significance using paired bootstrap.

        Args:
            result1: First result (with per-sample scores)
            result2: Second result (with per-sample scores)
            metric: Metric to test
            num_samples: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            Significance test results
        """
        import numpy as np

        # Get per-sample scores
        scores1 = result1.metrics.get(f"{metric}_scores", [])
        scores2 = result2.metrics.get(f"{metric}_scores", [])

        if not scores1 or not scores2 or len(scores1) != len(scores2):
            return {"significant": None, "error": "Cannot compare without per-sample scores"}

        scores1 = np.array(scores1)
        scores2 = np.array(scores2)
        n = len(scores1)

        # Bootstrap
        diffs = []
        for _ in range(num_samples):
            indices = np.random.randint(0, n, n)
            diff = np.mean(scores1[indices]) - np.mean(scores2[indices])
            diffs.append(diff)

        diffs = np.array(diffs)

        # Compute confidence interval
        alpha = 1 - confidence
        lower = np.percentile(diffs, 100 * alpha / 2)
        upper = np.percentile(diffs, 100 * (1 - alpha / 2))

        # Significant if CI doesn't contain 0
        significant = lower > 0 or upper < 0

        return {
            "significant": significant,
            "mean_diff": np.mean(diffs),
            "ci_lower": lower,
            "ci_upper": upper,
            "confidence": confidence,
            "p_value_approx": np.mean(diffs <= 0) if np.mean(diffs) > 0 else np.mean(diffs >= 0),
        }


def aggregate_by_language(
    results_dir: str,
    experiment_type: Optional[str] = None,
    metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to aggregate by language.

    Args:
        results_dir: Results directory
        experiment_type: Filter by experiment type
        metrics: Metrics to aggregate

    Returns:
        Aggregated results by language
    """
    manager = ResultsManager(results_dir)
    experiments = manager.list_experiments(experiment_type=experiment_type)

    results = []
    for exp in experiments:
        try:
            result = manager.load(exp["name"], exp["timestamp"])
            results.append(result)
        except Exception as e:
            logger.warning(f"Failed to load {exp['name']}: {e}")

    aggregator = ResultsAggregator(results)
    return aggregator.aggregate_by_language(metrics)


def aggregate_by_model(
    results_dir: str,
    experiment_type: Optional[str] = None,
    metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
) -> Dict[str, Dict[str, Any]]:
    """Convenience function to aggregate by model."""
    manager = ResultsManager(results_dir)
    experiments = manager.list_experiments(experiment_type=experiment_type)

    results = []
    for exp in experiments:
        try:
            result = manager.load(exp["name"], exp["timestamp"])
            results.append(result)
        except Exception:
            pass

    aggregator = ResultsAggregator(results)
    return aggregator.aggregate_by_model(metrics)


def aggregate_by_domain(
    results_dir: str,
    experiment_type: Optional[str] = None,
    metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
) -> Dict[str, Dict[str, Any]]:
    """Convenience function to aggregate by domain."""
    manager = ResultsManager(results_dir)
    experiments = manager.list_experiments(experiment_type=experiment_type)

    results = []
    for exp in experiments:
        try:
            result = manager.load(exp["name"], exp["timestamp"])
            results.append(result)
        except Exception:
            pass

    aggregator = ResultsAggregator(results)
    return aggregator.aggregate_by_domain(metrics)
