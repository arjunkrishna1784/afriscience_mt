"""Predictions storage and management."""

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranslationPrediction:
    """Single translation prediction with metadata."""

    source: str
    hypothesis: str
    reference: str
    source_lang: str
    target_lang: str

    # Optional metadata
    domain: Optional[str] = None
    doc_id: Optional[str] = None
    sample_id: Optional[int] = None

    # Per-sample metrics (if computed)
    metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source,
            "hypothesis": self.hypothesis,
            "reference": self.reference,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "domain": self.domain,
            "doc_id": self.doc_id,
            "sample_id": self.sample_id,
            "metrics": self.metrics,
        }


@dataclass
class PredictionSet:
    """Collection of predictions from an experiment."""

    experiment_name: str
    model_name: str
    model_type: str  # seq2seq, lora, icl
    source_lang: str
    target_lang: str

    predictions: List[TranslationPrediction] = field(default_factory=list)

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)

    # Aggregate metrics
    aggregate_metrics: Optional[Dict[str, float]] = None

    def add_prediction(
        self,
        source: str,
        hypothesis: str,
        reference: str,
        domain: Optional[str] = None,
        doc_id: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Add a prediction to the set."""
        pred = TranslationPrediction(
            source=source,
            hypothesis=hypothesis,
            reference=reference,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            domain=domain,
            doc_id=doc_id,
            sample_id=len(self.predictions),
            metrics=metrics,
        )
        self.predictions.append(pred)

    def add_batch(
        self,
        sources: List[str],
        hypotheses: List[str],
        references: List[str],
        domains: Optional[List[str]] = None,
        doc_ids: Optional[List[str]] = None,
        per_sample_metrics: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """Add a batch of predictions."""
        n = len(sources)
        domains = domains or [None] * n
        doc_ids = doc_ids or [None] * n
        per_sample_metrics = per_sample_metrics or [None] * n

        for src, hyp, ref, dom, doc, met in zip(
            sources, hypotheses, references, domains, doc_ids, per_sample_metrics
        ):
            self.add_prediction(src, hyp, ref, dom, doc, met)

    @property
    def sources(self) -> List[str]:
        return [p.source for p in self.predictions]

    @property
    def hypotheses(self) -> List[str]:
        return [p.hypothesis for p in self.predictions]

    @property
    def references(self) -> List[str]:
        return [p.reference for p in self.predictions]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "timestamp": self.timestamp,
            "config": self.config,
            "aggregate_metrics": self.aggregate_metrics,
            "num_predictions": len(self.predictions),
            "predictions": [p.to_dict() for p in self.predictions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictionSet":
        """Create from dictionary."""
        predictions_data = data.pop("predictions", [])
        data.pop("num_predictions", None)

        pred_set = cls(**data)
        for p_data in predictions_data:
            pred = TranslationPrediction(**p_data)
            pred_set.predictions.append(pred)

        return pred_set


class PredictionsManager:
    """Manager for saving and loading predictions."""

    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize predictions manager.

        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        prediction_set: PredictionSet,
        formats: List[str] = ["json", "tsv"],
    ) -> Dict[str, Path]:
        """
        Save predictions in multiple formats.

        Args:
            prediction_set: PredictionSet to save
            formats: Output formats (json, tsv, txt)

        Returns:
            Dict mapping format to saved path
        """
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.output_dir / prediction_set.experiment_name / timestamp
        exp_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save JSON (complete with all metadata)
        if "json" in formats:
            json_path = exp_dir / "predictions.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(prediction_set.to_dict(), f, ensure_ascii=False, indent=2)
            saved_paths["json"] = json_path
            logger.info(f"Saved predictions JSON: {json_path}")

        # Save TSV (tabular format for easy viewing)
        if "tsv" in formats:
            tsv_path = exp_dir / "predictions.tsv"
            self._save_tsv(prediction_set, tsv_path)
            saved_paths["tsv"] = tsv_path
            logger.info(f"Saved predictions TSV: {tsv_path}")

        # Save plain text (source ||| hypothesis ||| reference)
        if "txt" in formats:
            txt_path = exp_dir / "predictions.txt"
            self._save_txt(prediction_set, txt_path)
            saved_paths["txt"] = txt_path
            logger.info(f"Saved predictions TXT: {txt_path}")

        # Save hypotheses only (for external evaluation tools)
        hyp_path = exp_dir / "hypotheses.txt"
        with open(hyp_path, "w", encoding="utf-8") as f:
            for p in prediction_set.predictions:
                f.write(p.hypothesis + "\n")
        saved_paths["hypotheses"] = hyp_path

        # Save references (for external evaluation tools)
        ref_path = exp_dir / "references.txt"
        with open(ref_path, "w", encoding="utf-8") as f:
            for p in prediction_set.predictions:
                f.write(p.reference + "\n")
        saved_paths["references"] = ref_path

        # Save sources
        src_path = exp_dir / "sources.txt"
        with open(src_path, "w", encoding="utf-8") as f:
            for p in prediction_set.predictions:
                f.write(p.source + "\n")
        saved_paths["sources"] = src_path

        # Save metadata
        meta_path = exp_dir / "metadata.json"
        metadata = {
            "experiment_name": prediction_set.experiment_name,
            "model_name": prediction_set.model_name,
            "model_type": prediction_set.model_type,
            "source_lang": prediction_set.source_lang,
            "target_lang": prediction_set.target_lang,
            "timestamp": prediction_set.timestamp,
            "config": prediction_set.config,
            "aggregate_metrics": prediction_set.aggregate_metrics,
            "num_predictions": len(prediction_set.predictions),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        saved_paths["metadata"] = meta_path

        return saved_paths

    def _save_tsv(self, prediction_set: PredictionSet, path: Path) -> None:
        """Save as TSV file."""
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")

            # Header
            header = ["id", "source", "hypothesis", "reference", "domain"]
            if prediction_set.predictions and prediction_set.predictions[0].metrics:
                metric_names = list(prediction_set.predictions[0].metrics.keys())
                header.extend(metric_names)
            writer.writerow(header)

            # Data
            for i, p in enumerate(prediction_set.predictions):
                row = [i, p.source, p.hypothesis, p.reference, p.domain or ""]
                if p.metrics:
                    row.extend([p.metrics.get(m, "") for m in metric_names])
                writer.writerow(row)

    def _save_txt(self, prediction_set: PredictionSet, path: Path) -> None:
        """Save as plain text with ||| separator."""
        with open(path, "w", encoding="utf-8") as f:
            for p in prediction_set.predictions:
                f.write(f"{p.source} ||| {p.hypothesis} ||| {p.reference}\n")

    def load(
        self,
        experiment_name: str,
        timestamp: Optional[str] = None,
    ) -> PredictionSet:
        """
        Load predictions.

        Args:
            experiment_name: Experiment name
            timestamp: Specific timestamp (latest if None)

        Returns:
            PredictionSet
        """
        exp_base = self.output_dir / experiment_name

        if timestamp:
            exp_dir = exp_base / timestamp
        else:
            # Get latest
            timestamps = sorted([d.name for d in exp_base.iterdir() if d.is_dir()])
            if not timestamps:
                raise FileNotFoundError(f"No predictions found for {experiment_name}")
            exp_dir = exp_base / timestamps[-1]

        json_path = exp_dir / "predictions.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return PredictionSet.from_dict(data)

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all saved prediction experiments."""
        experiments = []

        for exp_dir in self.output_dir.iterdir():
            if exp_dir.is_dir():
                for ts_dir in exp_dir.iterdir():
                    if ts_dir.is_dir():
                        meta_path = ts_dir / "metadata.json"
                        if meta_path.exists():
                            with open(meta_path) as f:
                                meta = json.load(f)
                            meta["path"] = str(ts_dir)
                            experiments.append(meta)

        return sorted(experiments, key=lambda x: x.get("timestamp", ""), reverse=True)


def save_predictions(
    sources: List[str],
    hypotheses: List[str],
    references: List[str],
    output_dir: Union[str, Path],
    experiment_name: str,
    model_name: str,
    model_type: str,
    source_lang: str,
    target_lang: str,
    domains: Optional[List[str]] = None,
    metrics: Optional[Dict[str, float]] = None,
    per_sample_metrics: Optional[List[Dict[str, float]]] = None,
    config: Optional[Dict[str, Any]] = None,
    formats: List[str] = ["json", "tsv"],
) -> Dict[str, Path]:
    """
    Convenience function to save predictions.

    Args:
        sources: Source texts
        hypotheses: Model translations
        references: Reference translations
        output_dir: Output directory
        experiment_name: Experiment name
        model_name: Model name
        model_type: Model type (seq2seq, lora, icl)
        source_lang: Source language
        target_lang: Target language
        domains: Domain for each sample
        metrics: Aggregate metrics
        per_sample_metrics: Per-sample metrics
        config: Experiment config
        formats: Output formats

    Returns:
        Dict mapping format to saved path
    """
    pred_set = PredictionSet(
        experiment_name=experiment_name,
        model_name=model_name,
        model_type=model_type,
        source_lang=source_lang,
        target_lang=target_lang,
        config=config or {},
        aggregate_metrics=metrics,
    )

    pred_set.add_batch(
        sources=sources,
        hypotheses=hypotheses,
        references=references,
        domains=domains,
        per_sample_metrics=per_sample_metrics,
    )

    manager = PredictionsManager(output_dir)
    return manager.save(pred_set, formats=formats)
