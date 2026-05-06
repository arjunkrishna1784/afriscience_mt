"""Results management and persistence."""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils.reproducibility import get_reproducibility_info, get_git_info
from ..utils.checksum import compute_hash
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    experiment_name: str
    experiment_type: str  # "seq2seq_finetune", "lora_finetune", "icl", "evaluation"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Languages
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    lang_pair: Optional[str] = None

    # Model info
    model_name: Optional[str] = None
    model_type: Optional[str] = None

    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Predictions (optional, can be large)
    predictions: Optional[List[str]] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Reproducibility
    environment: Optional[Dict[str, Any]] = None
    git_info: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Set derived fields."""
        if self.source_lang and self.target_lang and not self.lang_pair:
            self.lang_pair = f"{self.source_lang}-{self.target_lang}"

        if self.environment is None:
            self.environment = get_reproducibility_info()

        if self.git_info is None:
            self.git_info = get_git_info()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentResult":
        """Create from dictionary."""
        # Filter out unknown fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)

    def get_checksum(self) -> str:
        """Get checksum of results for integrity verification."""
        # Hash key fields
        data = {
            "experiment_name": self.experiment_name,
            "config": self.config,
            "metrics": self.metrics,
        }
        return compute_hash(data)


class ResultsManager:
    """Manager for saving, loading, and organizing experiment results.

    Simplified structure - saves results directly in the experiment directory:
    experiments/seq2seq/nllb_600m/amh-eng/
    ├── final/              # Model files
    ├── results.json        # Main results
    ├── predictions.json    # Test predictions (optional)
    └── training.log        # Training logs
    """

    def __init__(
        self,
        results_dir: Union[str, Path],
        experiment_name: Optional[str] = None,
    ):
        """
        Initialize results manager.

        Args:
            results_dir: Directory for storing results (the experiment directory itself)
            experiment_name: Name of current experiment (optional, for index)
        """
        self.results_dir = Path(results_dir)
        self.experiment_name = experiment_name

        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        result: ExperimentResult,
        include_predictions: bool = True,
    ) -> Path:
        """
        Save experiment result directly in the experiment directory.

        Args:
            result: ExperimentResult to save
            include_predictions: Whether to save predictions separately

        Returns:
            Path to saved file
        """
        # Prepare data
        data = result.to_dict()

        # Extract predictions to save separately if large
        predictions = None
        if data.get("predictions") and len(data.get("predictions", [])) > 0:
            predictions = data.pop("predictions")

        # Save main results directly in the directory
        results_path = self.results_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save predictions separately if present
        if predictions and include_predictions:
            predictions_path = self.results_dir / "predictions.json"
            with open(predictions_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False)

        logger.info(f"Results saved to {results_path}")
        return results_path

    def load(self, include_predictions: bool = False) -> ExperimentResult:
        """
        Load experiment result from the directory.

        Args:
            include_predictions: Whether to load predictions

        Returns:
            ExperimentResult
        """
        # Load main results
        results_path = self.results_dir / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"No results found at {results_path}")

        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load predictions if requested
        if include_predictions:
            predictions_path = self.results_dir / "predictions.json"
            if predictions_path.exists():
                with open(predictions_path, "r", encoding="utf-8") as f:
                    data["predictions"] = json.load(f)

        return ExperimentResult.from_dict(data)

    def exists(self) -> bool:
        """Check if results exist."""
        return (self.results_dir / "results.json").exists()

    def update_metrics(self, new_metrics: Dict[str, Any]) -> None:
        """Update metrics in existing results file."""
        if not self.exists():
            logger.warning(f"No results file to update at {self.results_dir}")
            return

        result = self.load()
        result.metrics.update(new_metrics)
        result.timestamp = datetime.now().isoformat()
        self.save(result)


def save_results(
    results: Union[Dict[str, Any], ExperimentResult],
    output_dir: Union[str, Path],
    experiment_name: str,
    **kwargs,
) -> Path:
    """
    Convenience function to save results.

    Args:
        results: Results dict or ExperimentResult
        output_dir: Output directory
        experiment_name: Experiment name
        **kwargs: Additional ExperimentResult fields

    Returns:
        Path to saved file
    """
    manager = ResultsManager(output_dir, experiment_name)

    if isinstance(results, dict):
        result = ExperimentResult(
            experiment_name=experiment_name,
            metrics=results.get("metrics", results),
            **kwargs,
        )
    else:
        result = results

    return manager.save(result)


def load_results(
    results_dir: Union[str, Path],
    include_predictions: bool = False,
) -> ExperimentResult:
    """
    Convenience function to load results.

    Args:
        results_dir: Results directory
        include_predictions: Whether to load predictions

    Returns:
        ExperimentResult
    """
    manager = ResultsManager(results_dir)
    return manager.load(include_predictions=include_predictions)
