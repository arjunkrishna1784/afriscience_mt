"""Results management and reporting module."""

from .manager import (
    ResultsManager,
    ExperimentResult,
    save_results,
    load_results,
)
from .predictions import (
    TranslationPrediction,
    PredictionSet,
    PredictionsManager,
    save_predictions,
)
from .aggregator import (
    ResultsAggregator,
    aggregate_by_language,
    aggregate_by_model,
    aggregate_by_domain,
)
from .tables import (
    TableGenerator,
    generate_results_table,
    generate_latex_table,
    generate_comparison_table,
)

__all__ = [
    # Manager
    "ResultsManager",
    "ExperimentResult",
    "save_results",
    "load_results",
    # Predictions
    "TranslationPrediction",
    "PredictionSet",
    "PredictionsManager",
    "save_predictions",
    # Aggregator
    "ResultsAggregator",
    "aggregate_by_language",
    "aggregate_by_model",
    "aggregate_by_domain",
    # Tables
    "TableGenerator",
    "generate_results_table",
    "generate_latex_table",
    "generate_comparison_table",
]
