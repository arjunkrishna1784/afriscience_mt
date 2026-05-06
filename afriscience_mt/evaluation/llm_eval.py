"""LLM-as-judge evaluation for translation quality."""

import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from ..api.base import APIClient
from ..api.factory import create_api_client
from ..utils.config import load_evaluation_prompts
from ..utils.language_utils import get_language_name
from ..utils.logging import get_logger

logger = get_logger(__name__)


class LLMEvaluator:
    """Evaluator using LLM-as-judge for translation quality."""

    def __init__(
        self,
        api_client: Optional[APIClient] = None,
        provider: str = "openai",
        model_name: str = "gpt-4o",
        config_dir: Optional[Path] = None,
    ):
        """
        Initialize LLM evaluator.

        Args:
            api_client: Pre-configured API client
            provider: API provider if no client provided
            model_name: Model name if no client provided
            config_dir: Directory containing evaluation prompts
        """
        if api_client is not None:
            self.api_client = api_client
        else:
            self.api_client = create_api_client(
                provider=provider,
                model_name=model_name,
                temperature=0.0,
            )

        # Load evaluation prompts
        try:
            self.prompts = load_evaluation_prompts(config_dir)
        except FileNotFoundError:
            self.prompts = None
            logger.warning("Evaluation prompts not found, using defaults")

    def _get_prompt_template(self, aspect: str) -> str:
        """Get prompt template for an evaluation aspect."""
        if self.prompts and hasattr(self.prompts, "evaluation_prompts"):
            aspect_config = self.prompts.evaluation_prompts.get(aspect)
            if aspect_config:
                return aspect_config.prompt

        # Default prompts
        defaults = {
            "fluency": """Evaluate the FLUENCY of this translation on a scale of 1-5:
- 5: Native-level fluency, completely natural
- 4: Minor issues but reads naturally
- 3: Noticeable errors but understandable
- 2: Significant fluency problems
- 1: Very unnatural, difficult to read

Source ({source_language}):
{source_text}

Translation ({target_language}):
{hypothesis}

Provide your evaluation:
Score: [1-5]
Justification: [Brief explanation]""",

            "accuracy": """Evaluate the ACCURACY of this translation on a scale of 1-5:
- 5: Perfect meaning preservation
- 4: Minor semantic differences
- 3: Some meaning lost but main ideas present
- 2: Significant meaning errors
- 1: Translation fails to convey original meaning

Source ({source_language}):
{source_text}

Reference:
{reference}

Translation:
{hypothesis}

Provide your evaluation:
Score: [1-5]
Justification: [Brief explanation]""",

            "cohesion": """Evaluate the COHESION of this document translation on a scale of 1-5:
- 5: Excellent cohesion, consistent terminology, natural flow
- 4: Good cohesion with minor inconsistencies
- 3: Moderate cohesion, some terminology issues
- 2: Poor cohesion, notable inconsistencies
- 1: No cohesion, reads as disconnected sentences

Source document ({source_language}):
{source_document}

Translation ({target_language}):
{hypothesis_document}

Provide your evaluation:
Score: [1-5]
Justification: [Brief explanation]""",
        }

        return defaults.get(aspect, defaults["fluency"])

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse evaluation response to extract score and justification."""
        result = {"score": None, "justification": "", "raw_response": response}

        # Try to extract score
        score_patterns = [
            r"Score:\s*(\d)",
            r"(\d)/5",
            r"Rating:\s*(\d)",
            r"^(\d)\s*$",
        ]

        for pattern in score_patterns:
            match = re.search(pattern, response, re.MULTILINE | re.IGNORECASE)
            if match:
                result["score"] = int(match.group(1))
                break

        # Extract justification
        just_patterns = [
            r"Justification:\s*(.+?)(?:\n\n|$)",
            r"Explanation:\s*(.+?)(?:\n\n|$)",
            r"Reason:\s*(.+?)(?:\n\n|$)",
        ]

        for pattern in just_patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                result["justification"] = match.group(1).strip()
                break

        return result

    def evaluate_single(
        self,
        source_text: str,
        hypothesis: str,
        source_lang: str,
        target_lang: str,
        reference: Optional[str] = None,
        aspect: str = "fluency",
    ) -> Dict[str, Any]:
        """
        Evaluate a single translation.

        Args:
            source_text: Source text
            hypothesis: Model translation
            source_lang: Source language code
            target_lang: Target language code
            reference: Reference translation (optional)
            aspect: Evaluation aspect

        Returns:
            Evaluation result with score and justification
        """
        source_name = get_language_name(source_lang)
        target_name = get_language_name(target_lang)

        template = self._get_prompt_template(aspect)

        prompt = template.format(
            source_language=source_name,
            target_language=target_name,
            source_text=source_text,
            hypothesis=hypothesis,
            reference=reference or "Not provided",
            source_document=source_text,
            hypothesis_document=hypothesis,
        )

        response = self.api_client.generate(prompt)
        result = self._parse_response(response)
        result["aspect"] = aspect

        return result

    def evaluate_batch(
        self,
        sources: List[str],
        hypotheses: List[str],
        source_lang: str,
        target_lang: str,
        references: Optional[List[str]] = None,
        aspect: str = "fluency",
        show_progress: bool = True,
        delay: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of translations.

        Args:
            sources: Source texts
            hypotheses: Model translations
            source_lang: Source language code
            target_lang: Target language code
            references: Reference translations
            aspect: Evaluation aspect
            show_progress: Show progress bar
            delay: Delay between API calls

        Returns:
            List of evaluation results
        """
        results = []
        refs = references or [None] * len(sources)

        iterator = zip(sources, hypotheses, refs)
        if show_progress:
            iterator = tqdm(
                list(iterator),
                desc=f"Evaluating {aspect}"
            )

        for src, hyp, ref in iterator:
            result = self.evaluate_single(
                source_text=src,
                hypothesis=hyp,
                source_lang=source_lang,
                target_lang=target_lang,
                reference=ref,
                aspect=aspect,
            )
            results.append(result)

            if delay > 0:
                time.sleep(delay)

        return results

    def compute_aggregate_scores(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute aggregate scores from evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dict with mean, std, and distribution of scores
        """
        scores = [r["score"] for r in results if r["score"] is not None]

        if not scores:
            return {"mean": None, "std": None, "count": 0}

        import statistics

        aggregate = {
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "count": len(scores),
            "min": min(scores),
            "max": max(scores),
        }

        # Score distribution
        for i in range(1, 6):
            aggregate[f"score_{i}_count"] = scores.count(i)
            aggregate[f"score_{i}_pct"] = 100 * scores.count(i) / len(scores)

        return aggregate


def evaluate_fluency(
    sources: List[str],
    hypotheses: List[str],
    source_lang: str,
    target_lang: str,
    api_client: Optional[APIClient] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to evaluate fluency."""
    evaluator = LLMEvaluator(api_client=api_client)
    results = evaluator.evaluate_batch(
        sources=sources,
        hypotheses=hypotheses,
        source_lang=source_lang,
        target_lang=target_lang,
        aspect="fluency",
        **kwargs,
    )
    aggregate = evaluator.compute_aggregate_scores(results)
    return {"results": results, "aggregate": aggregate}


def evaluate_accuracy(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    source_lang: str,
    target_lang: str,
    api_client: Optional[APIClient] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to evaluate accuracy."""
    evaluator = LLMEvaluator(api_client=api_client)
    results = evaluator.evaluate_batch(
        sources=sources,
        hypotheses=hypotheses,
        source_lang=source_lang,
        target_lang=target_lang,
        references=references,
        aspect="accuracy",
        **kwargs,
    )
    aggregate = evaluator.compute_aggregate_scores(results)
    return {"results": results, "aggregate": aggregate}


def evaluate_cohesion(
    source_documents: List[str],
    hypothesis_documents: List[str],
    source_lang: str,
    target_lang: str,
    api_client: Optional[APIClient] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to evaluate document cohesion."""
    evaluator = LLMEvaluator(api_client=api_client)
    results = evaluator.evaluate_batch(
        sources=source_documents,
        hypotheses=hypothesis_documents,
        source_lang=source_lang,
        target_lang=target_lang,
        aspect="cohesion",
        **kwargs,
    )
    aggregate = evaluator.compute_aggregate_scores(results)
    return {"results": results, "aggregate": aggregate}
