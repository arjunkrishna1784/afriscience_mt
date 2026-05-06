"""Logging utilities for structured experiment logging."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

# Global logger cache
_loggers = {}


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for terminal output."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        log_message = super().format(record)
        color = self.COLORS.get(record.levelname, self.RESET)
        return f"{color}{log_message}{self.RESET}"


def setup_logging(
    name: str = "afriscience_mt",
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    file_level: Union[int, str] = logging.DEBUG,
) -> logging.Logger:
    """
    Set up logging with console and optional file output.

    Args:
        name: Logger name
        level: Console logging level
        log_file: Optional path to log file
        console: Whether to output to console
        file_level: File logging level

    Returns:
        Configured logger
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    if isinstance(file_level, str):
        file_level = getattr(logging, file_level.upper())

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Capture all, handlers filter

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = ColorFormatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "afriscience_mt") -> logging.Logger:
    """
    Get or create a logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name not in _loggers:
        return setup_logging(name)
    return _loggers[name]


class ExperimentLogger:
    """Structured logger for experiment tracking."""

    def __init__(
        self,
        experiment_name: str,
        log_dir: Optional[Path] = None,
        level: Union[int, str] = logging.INFO
    ):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for log files
            level: Logging level
        """
        self.experiment_name = experiment_name
        self.start_time = datetime.now()

        # Set up log file
        if log_dir:
            log_dir = Path(log_dir)
            timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"{experiment_name}_{timestamp}.log"
        else:
            log_file = None

        self.logger = setup_logging(
            name=f"afriscience_mt.{experiment_name}",
            level=level,
            log_file=log_file
        )

        self.metrics = {}
        self.step = 0

    def log_config(self, config: dict) -> None:
        """Log experiment configuration."""
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Configuration: {config}")

    def log_step(self, step: int, metrics: dict) -> None:
        """Log training/evaluation step."""
        self.step = step
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Step {step}: {metrics_str}")

        # Store metrics
        for k, v in metrics.items():
            if k not in self.metrics:
                self.metrics[k] = []
            self.metrics[k].append((step, v))

    def log_epoch(self, epoch: int, metrics: dict) -> None:
        """Log epoch summary."""
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self.logger.info(f"Epoch {epoch}: {metrics_str}")

    def log_evaluation(self, split: str, metrics: dict) -> None:
        """Log evaluation results."""
        self.logger.info(f"Evaluation on {split}:")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")

    def log_model_info(self, model_name: str, num_params: int, trainable_params: Optional[int] = None) -> None:
        """Log model information."""
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Total parameters: {num_params:,}")
        if trainable_params is not None:
            pct = 100 * trainable_params / num_params
            self.logger.info(f"Trainable parameters: {trainable_params:,} ({pct:.2f}%)")

    def log_data_info(self, split: str, num_samples: int, lang_pair: Optional[str] = None) -> None:
        """Log dataset information."""
        msg = f"Dataset {split}: {num_samples:,} samples"
        if lang_pair:
            msg += f" ({lang_pair})"
        self.logger.info(msg)

    def finish(self) -> None:
        """Log experiment completion."""
        duration = datetime.now() - self.start_time
        self.logger.info(f"Experiment completed in {duration}")


def log_translation_sample(
    logger: logging.Logger,
    source: str,
    reference: str,
    hypothesis: str,
    metrics: Optional[dict] = None
) -> None:
    """
    Log a translation sample for debugging.

    Args:
        logger: Logger instance
        source: Source text
        reference: Reference translation
        hypothesis: Model output
        metrics: Optional metrics for this sample
    """
    logger.debug("=" * 50)
    logger.debug(f"Source: {source}")
    logger.debug(f"Reference: {reference}")
    logger.debug(f"Hypothesis: {hypothesis}")
    if metrics:
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.debug(f"Metrics: {metrics_str}")
    logger.debug("=" * 50)
