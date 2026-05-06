"""Training callbacks for monitoring and model saving."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from ..utils.logging import get_logger

logger = get_logger(__name__)


class EvaluationCallback(TrainerCallback):
    """Callback for custom evaluation during training."""

    def __init__(
        self,
        eval_fn: Callable,
        eval_dataset: Any,
        eval_steps: int = 500,
        metric_name: str = "bleu",
    ):
        """
        Initialize evaluation callback.

        Args:
            eval_fn: Evaluation function that takes (model, dataset) and returns metrics dict
            eval_dataset: Dataset to evaluate on
            eval_steps: How often to evaluate (in steps)
            metric_name: Primary metric name for logging
        """
        self.eval_fn = eval_fn
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.metric_name = metric_name
        self.eval_history = []

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Run evaluation at specified intervals."""
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            model = kwargs.get("model")
            if model is None:
                return

            logger.info(f"Running evaluation at step {state.global_step}")

            # Run evaluation
            metrics = self.eval_fn(model, self.eval_dataset)

            # Log metrics
            self.eval_history.append({
                "step": state.global_step,
                "metrics": metrics,
            })

            primary_metric = metrics.get(self.metric_name)
            logger.info(f"Step {state.global_step}: {self.metric_name} = {primary_metric:.4f}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        return self.eval_history


class SaveBestModelCallback(TrainerCallback):
    """Callback to save the best model during training."""

    def __init__(
        self,
        metric_name: str = "eval_loss",
        mode: str = "min",
        save_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize best model callback.

        Args:
            metric_name: Metric to monitor
            mode: "min" or "max" (minimize or maximize metric)
            save_path: Path to save best model (uses trainer output_dir if None)
        """
        self.metric_name = metric_name
        self.mode = mode
        self.save_path = Path(save_path) if save_path else None

        self.best_metric = float("inf") if mode == "min" else float("-inf")
        self.best_step = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Dict[str, float],
        **kwargs,
    ):
        """Check if this is the best model and save if so."""
        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            return

        is_best = False
        if self.mode == "min" and current_metric < self.best_metric:
            is_best = True
        elif self.mode == "max" and current_metric > self.best_metric:
            is_best = True

        if is_best:
            self.best_metric = current_metric
            self.best_step = state.global_step

            # Save model
            save_path = self.save_path or Path(args.output_dir) / "best_model"
            save_path.mkdir(parents=True, exist_ok=True)

            model = kwargs.get("model")
            tokenizer = kwargs.get("tokenizer")

            if model is not None:
                model.save_pretrained(save_path)
            if tokenizer is not None:
                tokenizer.save_pretrained(save_path)

            logger.info(
                f"New best model! {self.metric_name}={current_metric:.4f} at step {state.global_step}. "
                f"Saved to {save_path}"
            )


class LoggingCallback(TrainerCallback):
    """Enhanced logging callback."""

    def __init__(
        self,
        log_interval: int = 100,
        metrics_to_log: Optional[List[str]] = None,
    ):
        """
        Initialize logging callback.

        Args:
            log_interval: How often to log (in steps)
            metrics_to_log: Specific metrics to log (all if None)
        """
        self.log_interval = log_interval
        self.metrics_to_log = metrics_to_log
        self.training_start_time = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log training start."""
        import time
        self.training_start_time = time.time()

        logger.info("=" * 50)
        logger.info("Training started")
        logger.info(f"  Model: {args.output_dir}")
        logger.info(f"  Epochs: {args.num_train_epochs}")
        logger.info(f"  Batch size: {args.per_device_train_batch_size}")
        logger.info(f"  Learning rate: {args.learning_rate}")
        logger.info("=" * 50)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float],
        **kwargs,
    ):
        """Log training metrics."""
        if state.global_step % self.log_interval != 0:
            return

        # Filter metrics if specified
        if self.metrics_to_log:
            logs = {k: v for k, v in logs.items() if k in self.metrics_to_log}

        # Format log message
        metrics_str = " | ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, (int, float)))

        logger.info(f"Step {state.global_step}: {metrics_str}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log training completion."""
        import time
        training_time = time.time() - self.training_start_time

        logger.info("=" * 50)
        logger.info("Training completed")
        logger.info(f"  Total steps: {state.global_step}")
        logger.info(f"  Training time: {training_time/3600:.2f} hours")
        logger.info("=" * 50)


class MemoryMonitorCallback(TrainerCallback):
    """Callback to monitor GPU memory usage."""

    def __init__(self, log_interval: int = 100):
        """
        Initialize memory monitor.

        Args:
            log_interval: How often to log memory (in steps)
        """
        self.log_interval = log_interval

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log memory usage."""
        if state.global_step % self.log_interval != 0:
            return

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3

            logger.debug(
                f"Step {state.global_step} - GPU Memory: "
                f"Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, "
                f"Max={max_allocated:.2f}GB"
            )
