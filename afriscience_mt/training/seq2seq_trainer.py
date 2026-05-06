"""Trainer for seq2seq models (M2M100, NLLB)."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    GenerationConfig,
    Seq2SeqTrainer as HFSeq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset as HFDataset

from ..data.dataset import AfriScienceDataset
from ..utils.language_utils import get_nllb_code, get_m2m100_code
from ..utils.logging import get_logger

logger = get_logger(__name__)


class DataCollatorForNLLB(DataCollatorForSeq2Seq):
    """
    Custom data collator for NLLB/M2M100 models that manually creates decoder_input_ids.

    NLLB models don't automatically generate decoder_input_ids during data collation,
    which causes errors when using label_smoothing_factor or certain training configs.
    This collator explicitly creates decoder_input_ids by shifting labels right.

    Reference: https://discuss.huggingface.co/t/label-smoothing-in-nllb-gives-valueerror-you-have-to-specify-either-decoder-input-ids-or-decoder-inputs-embeds/25320
    """

    def __call__(self, features, return_tensors=None):
        # Call parent collator to handle padding
        batch = super().__call__(features, return_tensors=return_tensors)

        # Manually create decoder_input_ids by shifting labels right
        if "labels" in batch:
            labels = batch["labels"]

            # Get decoder_start_token_id from model config
            if self.model is not None and hasattr(self.model, "config"):
                decoder_start_token_id = self.model.config.decoder_start_token_id
            else:
                # Fallback: use pad_token_id (this should be set properly)
                decoder_start_token_id = self.tokenizer.pad_token_id

            # Clone labels to create decoder_input_ids
            decoder_input_ids = labels.clone()

            # Shift right: move everything one position to the right
            # and put decoder_start_token_id at position 0
            decoder_input_ids[:, 1:] = labels[:, :-1].clone()
            decoder_input_ids[:, 0] = decoder_start_token_id

            # Replace -100 (ignore index) with pad_token_id in decoder_input_ids
            # because -100 is only valid for labels (loss masking), not for input
            decoder_input_ids = decoder_input_ids.masked_fill(
                decoder_input_ids == self.label_pad_token_id,
                self.tokenizer.pad_token_id
            )

            batch["decoder_input_ids"] = decoder_input_ids

        return batch


class Seq2SeqTrainer:
    """Trainer wrapper for seq2seq translation models."""

    def __init__(
        self,
        model_name: str,
        source_lang: str,
        target_lang: str,
        output_dir: Union[str, Path],
        model_type: str = "nllb",  # "nllb" or "m2m100"
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize trainer.

        Args:
            model_name: HuggingFace model name
            source_lang: Source language code
            target_lang: Target language code
            output_dir: Directory for outputs
            model_type: Type of model ("nllb" or "m2m100")
            cache_dir: Cache directory for model downloads
        """
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.output_dir = Path(output_dir)
        self.model_type = model_type
        self.cache_dir = cache_dir

        self.model = None
        self.tokenizer = None
        self.trainer = None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self, precision: str = "bf16") -> None:
        """
        Load model and tokenizer.

        Args:
            precision: Training precision (fp32, fp16, bf16)
                      Note: For fp16 mixed precision training, model is loaded in fp32
                      and the Trainer handles mixed precision. For bf16, we can load
                      directly in bf16.
        """
        logger.info(f"Loading model: {self.model_name}")

        # For mixed precision training:
        # - fp16: Load model in fp32, trainer handles mixed precision with gradient scaling
        # - bf16: Can load directly in bf16 (no gradient scaling needed)
        # - fp32: Load in fp32 for full precision training
        if precision == "bf16":
            model_dtype = torch.bfloat16
        else:
            # For fp16 mixed precision or fp32, load in fp32
            # The trainer will handle fp16 autocasting
            model_dtype = torch.float32

        # Load model (use 'dtype' instead of deprecated 'torch_dtype')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=model_dtype,
            cache_dir=self.cache_dir,
        )

        # Move generation config from model config to generation_config
        # This prevents the warning about max_length being in model config
        if hasattr(self.model.config, 'max_length'):
            max_length = self.model.config.max_length
            self.model.generation_config = GenerationConfig(
                max_length=max_length,
                decoder_start_token_id=self.model.config.decoder_start_token_id,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
            )
            # Remove from model config to avoid warning
            del self.model.config.max_length

        self.precision = precision

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        # Set language codes
        if self.model_type == "nllb":
            src_code = get_nllb_code(self.source_lang)
            tgt_code = get_nllb_code(self.target_lang)
        else:  # m2m100
            src_code = get_m2m100_code(self.source_lang)
            tgt_code = get_m2m100_code(self.target_lang)

        self.tokenizer.src_lang = src_code
        self.tokenizer.tgt_lang = tgt_code  # Required for NLLB target tokenization
        self.tgt_lang_code = tgt_code

        logger.info(f"Model loaded: {self.model.num_parameters():,} parameters")
        logger.info(f"Language codes: {src_code} -> {tgt_code}")

    def prepare_dataset(
        self,
        dataset: AfriScienceDataset,
        max_source_length: int = 256,
        max_target_length: int = 256,
    ) -> HFDataset:
        """
        Prepare dataset for training.

        Args:
            dataset: AfriScience dataset
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length

        Returns:
            HuggingFace Dataset
        """
        def tokenize_fn(examples):
            sources = examples["source"]
            targets = examples["target"]

            # Tokenize sources
            model_inputs = self.tokenizer(
                sources,
                max_length=max_source_length,
                truncation=True,
                padding=False,
            )

            # Tokenize targets separately to have proper control
            # For NLLB/M2M models, we need to set the target language for target tokenization
            labels = self.tokenizer(
                text_target=targets,
                max_length=max_target_length,
                truncation=True,
                padding=False,
            )

            # The labels are the input_ids from target tokenization
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        # Convert to HF Dataset
        data_dict = {
            "source": dataset.get_sources(),
            "target": dataset.get_targets(),
        }
        hf_dataset = HFDataset.from_dict(data_dict)

        # Tokenize
        tokenized = hf_dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=hf_dataset.column_names,
        )

        return tokenized

    def train(
        self,
        train_dataset: AfriScienceDataset,
        eval_dataset: Optional[AfriScienceDataset] = None,
        training_args: Optional[Dict[str, Any]] = None,
        early_stopping_patience: int = 3,
        resume_from_checkpoint: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments
            early_stopping_patience: Patience for early stopping
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results
        """
        if self.model is None:
            self.setup()

        # Determine mixed precision settings based on model precision
        # Default to bf16 as it's more stable and doesn't require gradient scaling
        precision = getattr(self, 'precision', 'bf16')
        use_fp16 = precision == "fp16"
        use_bf16 = precision == "bf16"

        # Default training arguments
        # Note: 'eval_strategy' is the new name (was 'evaluation_strategy' in older transformers)
        default_args = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": 10,
            "per_device_train_batch_size": 16,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 2,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "fp16": use_fp16,
            "bf16": use_bf16,
            "eval_strategy": "steps" if eval_dataset else "no",
            "eval_steps": 500 if eval_dataset else None,
            "save_strategy": "steps",
            "save_steps": 500,
            "save_total_limit": 3,  # Keep 3 checkpoints during training
            "load_best_model_at_end": True if eval_dataset else False,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "logging_steps": 100,
            "report_to": "none",
            "label_smoothing_factor": 0.1,
            "predict_with_generate": True,
        }

        if training_args:
            default_args.update(training_args)

        args = Seq2SeqTrainingArguments(**default_args)

        # Prepare datasets
        train_tokenized = self.prepare_dataset(train_dataset)
        eval_tokenized = self.prepare_dataset(eval_dataset) if eval_dataset else None

        # Use custom data collator for NLLB/M2M100 models
        # This collator explicitly creates decoder_input_ids by shifting labels right,
        # which is required for NLLB models (they don't auto-generate decoder_input_ids)
        data_collator = DataCollatorForNLLB(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        )

        # Callbacks
        callbacks = []
        if eval_dataset and early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            )

        # Create trainer
        # Use processing_class instead of deprecated tokenizer parameter
        self.trainer = HFSeq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=train_tokenized,
            eval_dataset=eval_tokenized,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train()

        # Save final model
        self.trainer.save_model(str(self.output_dir / "final"))
        self.tokenizer.save_pretrained(str(self.output_dir / "final"))

        logger.info(f"Training completed. Model saved to {self.output_dir / 'final'}")

        # Clean up checkpoint directories to save space
        # Only keep the final model directory
        self._cleanup_checkpoints()

        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        }

    def _cleanup_checkpoints(self) -> None:
        """Remove checkpoint directories after training to save disk space."""
        import shutil

        checkpoint_dirs = list(self.output_dir.glob("checkpoint-*"))
        if checkpoint_dirs:
            logger.info(f"Cleaning up {len(checkpoint_dirs)} checkpoint directories...")
            for checkpoint_dir in checkpoint_dirs:
                try:
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"Removed checkpoint: {checkpoint_dir.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint_dir}: {e}")

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save model and tokenizer."""
        path = Path(path) if path else self.output_dir / "checkpoint"
        path.mkdir(parents=True, exist_ok=True)

        if self.model:
            self.model.save_pretrained(path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)


def train_seq2seq(
    model_name: str,
    train_data_path: Union[str, Path],
    source_lang: str,
    target_lang: str,
    output_dir: Union[str, Path],
    eval_data_path: Optional[Union[str, Path]] = None,
    model_type: str = "nllb",
    training_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to train a seq2seq model.

    Args:
        model_name: HuggingFace model name
        train_data_path: Path to training data
        source_lang: Source language code
        target_lang: Target language code
        output_dir: Output directory
        eval_data_path: Path to evaluation data
        model_type: Model type ("nllb" or "m2m100")
        training_args: Training arguments
        **kwargs: Additional arguments

    Returns:
        Training results
    """
    # Load datasets
    train_dataset = AfriScienceDataset(
        data_path=train_data_path,
        source_lang=source_lang,
        target_lang=target_lang,
        split="train",
    )

    eval_dataset = None
    if eval_data_path:
        eval_dataset = AfriScienceDataset(
            data_path=eval_data_path,
            source_lang=source_lang,
            target_lang=target_lang,
            split="val",
        )

    # Create trainer
    trainer = Seq2SeqTrainer(
        model_name=model_name,
        source_lang=source_lang,
        target_lang=target_lang,
        output_dir=output_dir,
        model_type=model_type,
        **kwargs,
    )

    # Train
    return trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
    )
