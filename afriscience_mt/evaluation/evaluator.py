"""Main translation evaluator combining all evaluation methods."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .metrics import MetricsCalculator, compute_all_metrics, SSA_COMET_MODEL
from .llm_eval import LLMEvaluator
from .denorm_eval import DenormalizationEvaluator
from ..utils.language_utils import has_diacritics, get_language_name
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TranslationEvaluator:
    """Comprehensive evaluator for translation quality."""

    def __init__(
        self,
        metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
        enable_llm_eval: bool = False,
        enable_denorm_eval: bool = True,
        llm_eval_aspects: List[str] = ["fluency", "accuracy"],
        ssa_comet_model: str = SSA_COMET_MODEL,
        comet_model: str = "Unbabel/wmt22-comet-da",
    ):
        """
        Initialize evaluator.

        Args:
            metrics: Automatic metrics to compute
            enable_llm_eval: Whether to use LLM-as-judge
            enable_denorm_eval: Whether to evaluate denormalization
            llm_eval_aspects: Aspects for LLM evaluation
            ssa_comet_model: SSA-COMET model (McGill-NLP/ssa-comet-stl)
            comet_model: Standard COMET model name
        """
        self.metrics = metrics
        self.enable_llm_eval = enable_llm_eval
        self.enable_denorm_eval = enable_denorm_eval
        self.llm_eval_aspects = llm_eval_aspects

        self.metrics_calculator = MetricsCalculator(
            metrics=metrics,
            ssa_comet_model=ssa_comet_model,
            comet_model=comet_model,
        )

        self.llm_evaluator = None
        if enable_llm_eval:
            self.llm_evaluator = LLMEvaluator()

    def evaluate(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        source_lang: str,
        target_lang: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation.

        Args:
            sources: Source texts
            hypotheses: Model translations
            references: Reference translations
            source_lang: Source language code
            target_lang: Target language code
            metadata: Optional metadata to include

        Returns:
            Complete evaluation results
        """
        results = {
            "source_lang": source_lang,
            "target_lang": target_lang,
            "source_lang_name": get_language_name(source_lang),
            "target_lang_name": get_language_name(target_lang),
            "num_samples": len(sources),
            "metrics": self.metrics,
        }

        if metadata:
            results["metadata"] = metadata

        # Automatic metrics
        logger.info("Computing automatic metrics...")
        results["automatic_metrics"] = self.metrics_calculator.compute(
            sources, hypotheses, references
        )

        # Denormalization evaluation
        if self.enable_denorm_eval and has_diacritics(target_lang):
            logger.info(f"Computing denormalization metrics for {target_lang}...")
            denorm_evaluator = DenormalizationEvaluator(
                target_lang=target_lang,
                metrics=self.metrics,
            )
            results["denormalization"] = denorm_evaluator.evaluate(
                sources, hypotheses, references
            )

        # LLM evaluation
        if self.enable_llm_eval and self.llm_evaluator is not None:
            results["llm_evaluation"] = {}

            for aspect in self.llm_eval_aspects:
                logger.info(f"Running LLM evaluation for {aspect}...")

                eval_results = self.llm_evaluator.evaluate_batch(
                    sources=sources,
                    hypotheses=hypotheses,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    references=references if aspect == "accuracy" else None,
                    aspect=aspect,
                )

                aggregate = self.llm_evaluator.compute_aggregate_scores(eval_results)

                results["llm_evaluation"][aspect] = {
                    "aggregate": aggregate,
                    "per_sample": eval_results,
                }

        # Summary
        results["summary"] = self._create_summary(results)

        return results

    def _create_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of evaluation results."""
        summary = {}

        # Primary metrics
        auto_metrics = results.get("automatic_metrics", {})
        for metric in self.metrics:
            if metric in auto_metrics:
                summary[metric] = auto_metrics[metric]

        # Denormalization impact
        if "denormalization" in results:
            denorm = results["denormalization"]
            if "diacritic_impact" in denorm:
                summary["diacritic_impact_bleu"] = denorm["diacritic_impact"].get("bleu", {}).get("absolute_diff", 0)

        # LLM evaluation scores
        if "llm_evaluation" in results:
            for aspect, data in results["llm_evaluation"].items():
                if "aggregate" in data and "mean" in data["aggregate"]:
                    summary[f"llm_{aspect}"] = data["aggregate"]["mean"]

        return summary

    def evaluate_by_domain(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        domains: List[str],
        source_lang: str,
        target_lang: str,
    ) -> Dict[str, Any]:
        """
        Evaluate grouped by domain.

        Args:
            sources: Source texts
            hypotheses: Model translations
            references: Reference translations
            domains: Domain for each sample
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Results grouped by domain
        """
        # Group by domain
        domain_groups = {}
        for src, hyp, ref, dom in zip(sources, hypotheses, references, domains):
            if dom not in domain_groups:
                domain_groups[dom] = {"sources": [], "hypotheses": [], "references": []}
            domain_groups[dom]["sources"].append(src)
            domain_groups[dom]["hypotheses"].append(hyp)
            domain_groups[dom]["references"].append(ref)

        # Evaluate each domain
        results = {
            "overall": self.evaluate(
                sources, hypotheses, references, source_lang, target_lang
            ),
            "by_domain": {},
        }

        for domain, data in domain_groups.items():
            logger.info(f"Evaluating domain: {domain} ({len(data['sources'])} samples)")
            results["by_domain"][domain] = self.evaluate(
                data["sources"],
                data["hypotheses"],
                data["references"],
                source_lang,
                target_lang,
                metadata={"domain": domain},
            )

        # Domain ranking
        results["domain_ranking"] = self._rank_domains(results["by_domain"])

        return results

    def _rank_domains(self, domain_results: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Rank domains by each metric."""
        rankings = {}

        for metric in self.metrics:
            domain_scores = []
            for domain, results in domain_results.items():
                score = results.get("automatic_metrics", {}).get(metric)
                if score is not None:
                    domain_scores.append((domain, score))

            # Sort by score (higher is better for these metrics)
            domain_scores.sort(key=lambda x: x[1], reverse=True)
            rankings[metric] = [d[0] for d in domain_scores]

        return rankings

    def save_results(
        self,
        results: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> None:
        """
        Save evaluation results to file.

        Args:
            results: Evaluation results
            output_path: Output path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove non-serializable items
        results_clean = self._clean_for_json(results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_clean, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {output_path}")

    def _clean_for_json(self, obj: Any) -> Any:
        """Clean object for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)


def evaluate_translations(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    source_lang: str,
    target_lang: str,
    metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
    enable_llm_eval: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to evaluate translations.

    Args:
        sources: Source texts
        hypotheses: Model translations
        references: Reference translations
        source_lang: Source language code
        target_lang: Target language code
        metrics: Metrics to compute
        enable_llm_eval: Whether to use LLM evaluation
        output_path: Optional path to save results
        **kwargs: Additional arguments

    Returns:
        Evaluation results
    """
    evaluator = TranslationEvaluator(
        metrics=metrics,
        enable_llm_eval=enable_llm_eval,
        **kwargs,
    )

    results = evaluator.evaluate(
        sources=sources,
        hypotheses=hypotheses,
        references=references,
        source_lang=source_lang,
        target_lang=target_lang,
    )

    if output_path:
        evaluator.save_results(results, output_path)

    return results
