"""Translation quality metrics with SSA-COMET support for African languages."""

from typing import Any, Dict, List, Optional, Union
import torch

from ..utils.logging import get_logger

logger = get_logger(__name__)

# SSA-COMET model from McGill-NLP (specifically designed for African languages)
SSA_COMET_MODEL = "McGill-NLP/ssa-comet-mtl"

# Supported African languages in SSA-COMET
SSA_COMET_LANGUAGES = {
    "amh": "Amharic",
    "hau": "Hausa",
    "ibo": "Igbo",
    "kin": "Kinyarwanda",
    "lug": "Luganda",
    "luo": "Luo",
    "nso": "Northern Sotho",
    "nya": "Chichewa",
    "orm": "Oromo",
    "sna": "Shona",
    "som": "Somali",
    "swa": "Swahili",
    "tsn": "Tswana",
    "twi": "Twi",
    "wol": "Wolof",
    "xho": "Xhosa",
    "yor": "Yoruba",
    "zul": "isiZulu",
    "eng": "English",
}


def compute_bleu(
    hypotheses: List[str],
    references: List[str],
    tokenize: str = "13a",
    lowercase: bool = False,
) -> Dict[str, float]:
    """
    Compute BLEU score using sacrebleu.

    Args:
        hypotheses: Model translations
        references: Reference translations
        tokenize: Tokenization method
        lowercase: Whether to lowercase

    Returns:
        Dict with BLEU score and n-gram precisions
    """
    try:
        import sacrebleu
    except ImportError:
        raise ImportError("sacrebleu required: pip install sacrebleu")

    # sacrebleu raises TypeError on empty strings; replace with a single space
    # so failed/empty batch items don't crash the whole evaluation.
    hypotheses = [h if h and h.strip() else " " for h in hypotheses]

    bleu = sacrebleu.corpus_bleu(
        hypotheses,
        [references],
        tokenize=tokenize,
        lowercase=lowercase,
    )

    return {
        "bleu": bleu.score,
        "bleu_bp": bleu.bp,
        "bleu_precisions": bleu.precisions,
    }


def compute_chrf(
    hypotheses: List[str],
    references: List[str],
    word_order: int = 2,
) -> Dict[str, float]:
    """
    Compute chrF score using sacrebleu.

    Args:
        hypotheses: Model translations
        references: Reference translations
        word_order: Word n-gram order (0 = chrF, 2 = chrF++)

    Returns:
        Dict with chrF score
    """
    try:
        import sacrebleu
    except ImportError:
        raise ImportError("sacrebleu required: pip install sacrebleu")

    chrf = sacrebleu.corpus_chrf(
        hypotheses,
        [references],
        word_order=word_order,
    )

    return {
        "chrf": chrf.score,
    }


def compute_comet(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int = 1,
) -> Dict[str, Any]:
    """
    Compute COMET score.

    Args:
        sources: Source texts
        hypotheses: Model translations
        references: Reference translations
        model_name: COMET model name
        batch_size: Batch size for scoring
        gpus: Number of GPUs

    Returns:
        Dict with COMET score and per-sample scores
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        raise ImportError("unbabel-comet required: pip install unbabel-comet")

    # Download and load model
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    # Prepare data
    data = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(sources, hypotheses, references)
    ]

    # Score
    output = model.predict(data, batch_size=batch_size, gpus=gpus)

    return {
        "comet": output.system_score,
        "comet_scores": output.scores,
    }


def compute_ssa_comet(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    model_name: str = SSA_COMET_MODEL,
    batch_size: int = 8,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute SSA-COMET score using McGill-NLP's model for African languages.

    SSA-COMET (Sub-Saharan African COMET) is specifically trained for
    evaluating translations involving African languages.

    Reference: https://huggingface.co/McGill-NLP/ssa-comet-mtl

    Args:
        sources: Source texts
        hypotheses: Model translations
        references: Reference translations
        model_name: SSA-COMET model name (default: McGill-NLP/ssa-comet-mtl)
        batch_size: Batch size for scoring
        device: Device to use (cuda/cpu, auto-detected if None)

    Returns:
        Dict with SSA-COMET score and per-sample scores
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        raise ImportError("unbabel-comet required: pip install unbabel-comet")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpus = 1 if device == "cuda" else 0
    else:
        gpus = 1 if "cuda" in device else 0

    logger.info(f"Loading SSA-COMET model: {model_name}")

    # Download and load SSA-COMET model
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    # Move to device
    if device == "cuda":
        model = model.cuda()

    # Prepare data
    data = [
        {"src": src, "mt": hyp, "ref": ref}
        for src, hyp, ref in zip(sources, hypotheses, references)
    ]

    logger.info(f"Computing SSA-COMET scores for {len(data)} samples...")

    # Score
    output = model.predict(data, batch_size=batch_size, gpus=gpus)

    return {
        "ssa_comet": output.system_score,
        "ssa_comet_scores": output.scores,
    }


def compute_all_metrics(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
    ssa_comet_model: str = SSA_COMET_MODEL,
    comet_model: str = "Unbabel/wmt22-comet-da",
) -> Dict[str, Any]:
    """
    Compute all specified metrics.

    Args:
        sources: Source texts
        hypotheses: Model translations
        references: Reference translations
        metrics: List of metrics to compute
        ssa_comet_model: SSA-COMET model name
        comet_model: Standard COMET model name (if needed)

    Returns:
        Dict with all metric scores
    """
    results = {}

    if "bleu" in metrics:
        results.update(compute_bleu(hypotheses, references))

    if "chrf" in metrics:
        results.update(compute_chrf(hypotheses, references))

    # SSA-COMET (preferred for African languages)
    if "ssa_comet" in metrics:
        try:
            ssa_results = compute_ssa_comet(
                sources, hypotheses, references, model_name=ssa_comet_model
            )
            results.update(ssa_results)
        except Exception as e:
            logger.warning(f"SSA-COMET computation failed: {e}")

    # Standard COMET (fallback or comparison)
    if "comet" in metrics:
        try:
            comet_results = compute_comet(
                sources, hypotheses, references, model_name=comet_model
            )
            results.update(comet_results)
        except Exception as e:
            logger.warning(f"COMET computation failed: {e}")

    return results


class SSACOMETScorer:
    """
    SSA-COMET scorer with model caching for efficient batch evaluation.

    Uses McGill-NLP/ssa-comet-mtl, which is specifically trained for
    Sub-Saharan African languages.
    """

    def __init__(
        self,
        model_name: str = SSA_COMET_MODEL,
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        """
        Initialize SSA-COMET scorer.

        Args:
            model_name: SSA-COMET model name
            batch_size: Default batch size for scoring
            device: Device to use (auto-detected if None)
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.gpus = 1 if self.device == "cuda" else 0

    def _load_model(self):
        """Load the SSA-COMET model (lazy loading)."""
        if self._model is None:
            try:
                from comet import download_model, load_from_checkpoint
            except ImportError:
                raise ImportError("unbabel-comet required: pip install unbabel-comet")

            logger.info(f"Loading SSA-COMET model: {self.model_name}")
            model_path = download_model(self.model_name)
            self._model = load_from_checkpoint(model_path)

            if self.device == "cuda":
                self._model = self._model.cuda()

            logger.info(f"SSA-COMET model loaded on {self.device}")

        return self._model

    def score(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute SSA-COMET scores.

        Args:
            sources: Source texts
            hypotheses: Model translations
            references: Reference translations
            batch_size: Batch size (uses default if None)

        Returns:
            Dict with system score and per-sample scores
        """
        model = self._load_model()
        batch_size = batch_size or self.batch_size

        data = [
            {"src": src, "mt": hyp, "ref": ref}
            for src, hyp, ref in zip(sources, hypotheses, references)
        ]

        output = model.predict(data, batch_size=batch_size, gpus=self.gpus)

        return {
            "ssa_comet": output.system_score,
            "ssa_comet_scores": output.scores,
        }

    def score_single(
        self,
        source: str,
        hypothesis: str,
        reference: str,
    ) -> float:
        """
        Compute SSA-COMET score for a single sample.

        Args:
            source: Source text
            hypothesis: Model translation
            reference: Reference translation

        Returns:
            SSA-COMET score
        """
        result = self.score([source], [hypothesis], [reference])
        return result["ssa_comet_scores"][0]


class MetricsCalculator:
    """Calculator for translation metrics with caching and batching."""

    def __init__(
        self,
        metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
        ssa_comet_model: str = SSA_COMET_MODEL,
        comet_model: str = "Unbabel/wmt22-comet-da",
        cache_models: bool = True,
        device: Optional[str] = None,
    ):
        """
        Initialize metrics calculator.

        Args:
            metrics: Metrics to compute (bleu, chrf, ssa_comet, comet)
            ssa_comet_model: SSA-COMET model name
            comet_model: Standard COMET model name
            cache_models: Whether to cache loaded models
            device: Device for neural metrics (auto-detected if None)
        """
        self.metrics = metrics
        self.ssa_comet_model_name = ssa_comet_model
        self.comet_model_name = comet_model
        self.cache_models = cache_models
        self.device = device

        # Lazy-loaded scorers
        self._ssa_comet_scorer = None
        self._comet_model = None

    def _get_ssa_comet_scorer(self) -> SSACOMETScorer:
        """Get or create SSA-COMET scorer."""
        if self._ssa_comet_scorer is None or not self.cache_models:
            self._ssa_comet_scorer = SSACOMETScorer(
                model_name=self.ssa_comet_model_name,
                device=self.device,
            )
        return self._ssa_comet_scorer

    def _get_comet_model(self):
        """Get or load standard COMET model."""
        if self._comet_model is None or not self.cache_models:
            try:
                from comet import download_model, load_from_checkpoint
                model_path = download_model(self.comet_model_name)
                self._comet_model = load_from_checkpoint(model_path)
            except ImportError:
                raise ImportError("unbabel-comet required: pip install unbabel-comet")
        return self._comet_model

    def compute(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> Dict[str, Any]:
        """
        Compute all configured metrics.

        Args:
            sources: Source texts
            hypotheses: Model translations
            references: Reference translations

        Returns:
            Dict with all metric scores
        """
        results = {}

        if "bleu" in self.metrics:
            results.update(compute_bleu(hypotheses, references))

        if "chrf" in self.metrics:
            results.update(compute_chrf(hypotheses, references))

        # SSA-COMET (primary metric for African languages)
        if "ssa_comet" in self.metrics:
            try:
                scorer = self._get_ssa_comet_scorer()
                ssa_results = scorer.score(sources, hypotheses, references)
                results.update(ssa_results)
            except Exception as e:
                logger.warning(f"SSA-COMET computation failed: {e}")

        # Standard COMET (for comparison or non-African languages)
        if "comet" in self.metrics:
            try:
                model = self._get_comet_model()
                data = [
                    {"src": src, "mt": hyp, "ref": ref}
                    for src, hyp, ref in zip(sources, hypotheses, references)
                ]
                gpus = 1 if torch.cuda.is_available() else 0
                output = model.predict(data, batch_size=8, gpus=gpus)
                results["comet"] = output.system_score
                results["comet_scores"] = output.scores
            except Exception as e:
                logger.warning(f"COMET computation failed: {e}")

        return results

    def compute_per_sample(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
    ) -> List[Dict[str, float]]:
        """
        Compute metrics for each sample.

        Args:
            sources: Source texts
            hypotheses: Model translations
            references: Reference translations

        Returns:
            List of metric dicts for each sample
        """
        # Compute corpus-level first for neural metric scores
        corpus_metrics = self.compute(sources, hypotheses, references)

        per_sample = []
        ssa_comet_scores = corpus_metrics.get("ssa_comet_scores", [None] * len(hypotheses))
        comet_scores = corpus_metrics.get("comet_scores", [None] * len(hypotheses))

        for i, (src, hyp, ref) in enumerate(zip(sources, hypotheses, references)):
            sample_metrics = {}

            # BLEU (sentence-level)
            if "bleu" in self.metrics:
                try:
                    import sacrebleu
                    bleu = sacrebleu.sentence_bleu(hyp, [ref])
                    sample_metrics["bleu"] = bleu.score
                except Exception:
                    pass

            # chrF (sentence-level)
            if "chrf" in self.metrics:
                try:
                    import sacrebleu
                    chrf = sacrebleu.sentence_chrf(hyp, [ref])
                    sample_metrics["chrf"] = chrf.score
                except Exception:
                    pass

            # SSA-COMET (from pre-computed)
            if ssa_comet_scores[i] is not None:
                sample_metrics["ssa_comet"] = ssa_comet_scores[i]

            # COMET (from pre-computed)
            if comet_scores[i] is not None:
                sample_metrics["comet"] = comet_scores[i]

            per_sample.append(sample_metrics)

        return per_sample

    def get_primary_metric(self) -> str:
        """Get the primary metric name (SSA-COMET if available)."""
        if "ssa_comet" in self.metrics:
            return "ssa_comet"
        elif "comet" in self.metrics:
            return "comet"
        elif "bleu" in self.metrics:
            return "bleu"
        else:
            return self.metrics[0] if self.metrics else "unknown"


# Convenience function for quick evaluation
def evaluate_translations(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    use_ssa_comet: bool = True,
) -> Dict[str, Any]:
    """
    Quick evaluation of translations with standard metrics.

    Args:
        sources: Source texts
        hypotheses: Model translations
        references: Reference translations
        use_ssa_comet: Whether to use SSA-COMET (True) or standard COMET (False)

    Returns:
        Dict with BLEU, chrF, and COMET/SSA-COMET scores
    """
    metrics = ["bleu", "chrf", "ssa_comet" if use_ssa_comet else "comet"]
    calculator = MetricsCalculator(metrics=metrics)
    return calculator.compute(sources, hypotheses, references)
