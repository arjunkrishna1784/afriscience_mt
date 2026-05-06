"""Denormalization utilities for diacritics analysis."""

import unicodedata
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils.language_utils import (
    DIACRITIC_LANGUAGES,
    has_diacritics,
    get_special_chars,
)


# Character mapping for stripping diacritics
DIACRITIC_MAPPINGS = {
    # Yoruba
    "ẹ": "e", "Ẹ": "E",
    "ọ": "o", "Ọ": "O",
    "ṣ": "s", "Ṣ": "S",
    "à": "a", "á": "a", "ā": "a", "À": "A", "Á": "A", "Ā": "A",
    "è": "e", "é": "e", "ē": "e", "È": "E", "É": "E", "Ē": "E",
    "ì": "i", "í": "i", "ī": "i", "Ì": "I", "Í": "I", "Ī": "I",
    "ò": "o", "ó": "o", "ō": "o", "Ò": "O", "Ó": "O", "Ō": "O",
    "ù": "u", "ú": "u", "ū": "u", "Ù": "U", "Ú": "U", "Ū": "U",

    # Hausa (hooked letters)
    "ɓ": "b", "Ɓ": "B",
    "ɗ": "d", "Ɗ": "D",
    "ƙ": "k", "Ƙ": "K",
    "ƴ": "y", "Ƴ": "Y",

    # Northern Sotho
    "š": "s", "Š": "S",
}


def strip_diacritics(
    text: str,
    language: Optional[str] = None,
    method: str = "mapping",
) -> str:
    """
    Remove diacritics from text.

    Args:
        text: Input text
        language: Language code for language-specific handling
        method: Method to use ('mapping', 'nfd', 'both')

    Returns:
        Text with diacritics removed
    """
    if method in ("mapping", "both"):
        # Apply character mappings
        for char, replacement in DIACRITIC_MAPPINGS.items():
            text = text.replace(char, replacement)

    if method in ("nfd", "both"):
        # NFD normalization approach
        # Decompose characters and remove combining marks
        nfd = unicodedata.normalize("NFD", text)
        text = "".join(
            char for char in nfd
            if unicodedata.category(char) != "Mn"  # Mn = Mark, Nonspacing
        )

    return text


def normalize_for_comparison(
    text: str,
    language: str,
    strip_diacritics_flag: bool = True,
    lowercase: bool = True,
    strip_whitespace: bool = True,
) -> str:
    """
    Normalize text for fair comparison.

    Args:
        text: Input text
        language: Language code
        strip_diacritics_flag: Whether to strip diacritics
        lowercase: Whether to lowercase
        strip_whitespace: Whether to normalize whitespace

    Returns:
        Normalized text
    """
    if strip_diacritics_flag:
        text = strip_diacritics(text, language)

    if lowercase:
        text = text.lower()

    if strip_whitespace:
        text = " ".join(text.split())

    return text


class Denormalizer:
    """
    Denormalization handler for evaluation.

    Provides methods to evaluate translations with and without
    diacritics for fair comparison.
    """

    def __init__(self, language: str):
        """
        Initialize denormalizer for a language.

        Args:
            language: Target language code
        """
        self.language = language
        self.has_diacritics = has_diacritics(language)
        self.special_chars = get_special_chars(language)

    def denormalize(self, text: str) -> str:
        """
        Strip diacritics from text.

        Args:
            text: Input text

        Returns:
            Text without diacritics
        """
        return strip_diacritics(text, self.language)

    def compute_diacritic_accuracy(
        self,
        hypothesis: str,
        reference: str,
    ) -> Dict[str, float]:
        """
        Compute diacritic-specific accuracy metrics.

        Args:
            hypothesis: Model output
            reference: Reference translation

        Returns:
            Dictionary with diacritic metrics
        """
        if not self.has_diacritics:
            return {"diacritic_accuracy": 1.0, "has_diacritics": False}

        metrics = {
            "has_diacritics": True,
            "total_diacritic_chars_ref": 0,
            "total_diacritic_chars_hyp": 0,
            "correct_diacritic_positions": 0,
        }

        # Count diacritic characters
        ref_lower = reference.lower()
        hyp_lower = hypothesis.lower()

        for char in DIACRITIC_MAPPINGS:
            metrics["total_diacritic_chars_ref"] += ref_lower.count(char)
            metrics["total_diacritic_chars_hyp"] += hyp_lower.count(char)

        # Simple position-based matching (approximate)
        # This is a rough measure; could be improved with alignment
        ref_norm = self.denormalize(reference)
        hyp_norm = self.denormalize(hypothesis)

        if ref_norm == hyp_norm:
            # Same normalized form, count matching diacritics
            min_len = min(len(reference), len(hypothesis))
            for i in range(min_len):
                if reference[i] == hypothesis[i] and reference[i] in DIACRITIC_MAPPINGS:
                    metrics["correct_diacritic_positions"] += 1

        # Compute accuracy
        if metrics["total_diacritic_chars_ref"] > 0:
            metrics["diacritic_recall"] = (
                metrics["correct_diacritic_positions"] /
                metrics["total_diacritic_chars_ref"]
            )
        else:
            metrics["diacritic_recall"] = 1.0

        if metrics["total_diacritic_chars_hyp"] > 0:
            metrics["diacritic_precision"] = (
                metrics["correct_diacritic_positions"] /
                metrics["total_diacritic_chars_hyp"]
            )
        else:
            metrics["diacritic_precision"] = 1.0

        # F1
        prec = metrics["diacritic_precision"]
        rec = metrics["diacritic_recall"]
        if prec + rec > 0:
            metrics["diacritic_f1"] = 2 * prec * rec / (prec + rec)
        else:
            metrics["diacritic_f1"] = 0.0

        return metrics


def compute_denormalized_metrics(
    hypotheses: List[str],
    references: List[str],
    language: str,
    metric_fn: Callable,
    metric_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute metrics both with and without diacritics.

    Args:
        hypotheses: Model outputs
        references: Reference translations
        language: Target language code
        metric_fn: Metric function to apply
        metric_kwargs: Additional arguments for metric function

    Returns:
        Dictionary with original and denormalized metrics
    """
    metric_kwargs = metric_kwargs or {}

    denormalizer = Denormalizer(language)

    results = {}

    # Original metrics (with diacritics)
    results["original"] = metric_fn(hypotheses, references, **metric_kwargs)

    # Denormalized metrics (without diacritics)
    if denormalizer.has_diacritics:
        hyp_denorm = [denormalizer.denormalize(h) for h in hypotheses]
        ref_denorm = [denormalizer.denormalize(r) for r in references]

        results["denormalized"] = metric_fn(hyp_denorm, ref_denorm, **metric_kwargs)

        # Compute difference
        if isinstance(results["original"], dict):
            results["diacritic_impact"] = {
                k: results["denormalized"].get(k, 0) - results["original"].get(k, 0)
                for k in results["original"]
            }
        else:
            results["diacritic_impact"] = results["denormalized"] - results["original"]
    else:
        results["denormalized"] = results["original"]
        results["diacritic_impact"] = 0

    # Per-sample diacritic accuracy
    diacritic_metrics = []
    for hyp, ref in zip(hypotheses, references):
        diacritic_metrics.append(
            denormalizer.compute_diacritic_accuracy(hyp, ref)
        )

    # Aggregate diacritic metrics
    if diacritic_metrics:
        results["diacritic_metrics"] = {
            "mean_f1": sum(m.get("diacritic_f1", 0) for m in diacritic_metrics) / len(diacritic_metrics),
            "mean_precision": sum(m.get("diacritic_precision", 0) for m in diacritic_metrics) / len(diacritic_metrics),
            "mean_recall": sum(m.get("diacritic_recall", 0) for m in diacritic_metrics) / len(diacritic_metrics),
        }

    return results
