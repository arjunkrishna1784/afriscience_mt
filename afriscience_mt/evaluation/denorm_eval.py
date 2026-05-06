"""Denormalization evaluation for diacritics analysis."""

from typing import Any, Dict, List, Optional

from ..data.denormalization import (
    Denormalizer,
    strip_diacritics,
    compute_denormalized_metrics,
)
from ..utils.language_utils import has_diacritics
from .metrics import MetricsCalculator, compute_all_metrics
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DenormalizationEvaluator:
    """Evaluator for analyzing impact of diacritics on translation quality."""

    def __init__(
        self,
        target_lang: str,
        metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
    ):
        """
        Initialize denormalization evaluator.

        Args:
            target_lang: Target language code
            metrics: Metrics to compute
        """
        self.target_lang = target_lang
        self.metrics = metrics
        self.denormalizer = Denormalizer(target_lang)
        self.metrics_calculator = MetricsCalculator(metrics=metrics)

    def evaluate(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate translations with and without diacritics.

        Args:
            sources: Source texts
            hypotheses: Model translations
            references: Reference translations

        Returns:
            Evaluation results with diacritic analysis
        """
        results = {
            "target_lang": self.target_lang,
            "has_diacritics": self.denormalizer.has_diacritics,
            "num_samples": len(sources),
        }

        # Original metrics (with diacritics)
        logger.info("Computing metrics with diacritics...")
        results["original"] = self.metrics_calculator.compute(
            sources, hypotheses, references
        )

        # Denormalized metrics (without diacritics)
        if self.denormalizer.has_diacritics:
            logger.info("Computing metrics without diacritics...")

            hyp_denorm = [self.denormalizer.denormalize(h) for h in hypotheses]
            ref_denorm = [self.denormalizer.denormalize(r) for r in references]

            results["denormalized"] = self.metrics_calculator.compute(
                sources, hyp_denorm, ref_denorm
            )

            # Compute impact
            results["diacritic_impact"] = {}
            for metric in self.metrics:
                orig_val = results["original"].get(metric, 0)
                denorm_val = results["denormalized"].get(metric, 0)
                results["diacritic_impact"][metric] = {
                    "absolute_diff": denorm_val - orig_val,
                    "relative_diff_pct": 100 * (denorm_val - orig_val) / orig_val if orig_val != 0 else 0,
                }

            # Per-sample diacritic accuracy
            results["diacritic_accuracy"] = self._compute_diacritic_accuracy(
                hypotheses, references
            )
        else:
            results["denormalized"] = results["original"]
            results["diacritic_impact"] = {m: {"absolute_diff": 0, "relative_diff_pct": 0} for m in self.metrics}

        return results

    def _compute_diacritic_accuracy(
        self,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """Compute diacritic-specific accuracy metrics."""
        all_metrics = []

        for hyp, ref in zip(hypotheses, references):
            metrics = self.denormalizer.compute_diacritic_accuracy(hyp, ref)
            all_metrics.append(metrics)

        # Aggregate
        if not all_metrics:
            return {}

        aggregate = {}
        for key in ["diacritic_precision", "diacritic_recall", "diacritic_f1"]:
            values = [m.get(key, 0) for m in all_metrics if key in m]
            if values:
                aggregate[f"mean_{key}"] = sum(values) / len(values)

        aggregate["total_diacritic_chars_ref"] = sum(
            m.get("total_diacritic_chars_ref", 0) for m in all_metrics
        )
        aggregate["total_diacritic_chars_hyp"] = sum(
            m.get("total_diacritic_chars_hyp", 0) for m in all_metrics
        )

        return aggregate

    def evaluate_per_sample(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Evaluate each sample with diacritic analysis.

        Args:
            sources: Source texts
            hypotheses: Model translations
            references: Reference translations

        Returns:
            List of per-sample evaluation results
        """
        per_sample = []

        for src, hyp, ref in zip(sources, hypotheses, references):
            sample_result = {}

            # Original metrics
            orig_metrics = compute_all_metrics(
                [src], [hyp], [ref], metrics=self.metrics
            )
            sample_result["original"] = orig_metrics

            # Denormalized metrics
            if self.denormalizer.has_diacritics:
                hyp_denorm = self.denormalizer.denormalize(hyp)
                ref_denorm = self.denormalizer.denormalize(ref)

                denorm_metrics = compute_all_metrics(
                    [src], [hyp_denorm], [ref_denorm], metrics=self.metrics
                )
                sample_result["denormalized"] = denorm_metrics

                # Diacritic accuracy
                sample_result["diacritic_metrics"] = self.denormalizer.compute_diacritic_accuracy(
                    hyp, ref
                )
            else:
                sample_result["denormalized"] = orig_metrics

            per_sample.append(sample_result)

        return per_sample


def evaluate_with_denormalization(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    target_lang: str,
    metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
) -> Dict[str, Any]:
    """
    Convenience function to evaluate with denormalization analysis.

    Args:
        sources: Source texts
        hypotheses: Model translations
        references: Reference translations
        target_lang: Target language code
        metrics: Metrics to compute

    Returns:
        Evaluation results with diacritic analysis
    """
    evaluator = DenormalizationEvaluator(
        target_lang=target_lang,
        metrics=metrics,
    )

    return evaluator.evaluate(sources, hypotheses, references)


def get_languages_with_diacritics() -> List[str]:
    """Get list of languages that have diacritics."""
    from ..utils.language_utils import LANGUAGE_CODES

    return [lang for lang in LANGUAGE_CODES if has_diacritics(lang)]
