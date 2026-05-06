"""LoRA/QLoRA trainer for LLMs."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, SFTConfig
from datasets import Dataset as HFDataset

from ..data.dataset import AfriScienceDataset
from ..utils.language_utils import get_language_name
from ..utils.logging import get_logger

logger = get_logger(__name__)


def get_lora_config(
    rank: int = 16,
    alpha: Optional[int] = None,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    bias: str = "none",
    use_dora: bool = False,
) -> LoraConfig:
    """
    Create LoRA/DoRA configuration.

    Args:
        rank: LoRA rank
        alpha: LoRA alpha (defaults to rank)
        dropout: LoRA dropout
        target_modules: Modules to apply LoRA to
        bias: Bias training strategy
        use_dora: Use DoRA (Weight-Decomposed Low-Rank Adaptation)

    Returns:
        LoraConfig instance
    """
    if alpha is None:
        alpha = rank

    if target_modules is None:
        # Default target modules for Llama/Gemma
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=TaskType.CAUSAL_LM,
        use_dora=use_dora,
    )


class LoRATrainer:
    """Trainer for LoRA/QLoRA fine-tuning of LLMs."""

    def __init__(
        self,
        model_name: str,
        source_lang: str,
        target_lang: str,
        output_dir: Union[str, Path],
        lora_rank: int = 16,
        load_in_4bit: bool = True,
        cache_dir: Optional[str] = None,
        use_dora: bool = False,
    ):
        """
        Initialize LoRA/DoRA trainer.

        Args:
            model_name: HuggingFace model name
            source_lang: Source language code
            target_lang: Target language code
            output_dir: Directory for outputs
            lora_rank: LoRA rank
            load_in_4bit: Use 4-bit quantization (QLoRA)
            cache_dir: Cache directory
            use_dora: Use DoRA (Weight-Decomposed Low-Rank Adaptation)
        """
        self.model_name = model_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.output_dir = Path(output_dir)
        self.lora_rank = lora_rank
        self.load_in_4bit = load_in_4bit
        self.cache_dir = cache_dir
        self.use_dora = use_dora

        self.model = None
        self.tokenizer = None
        self.trainer = None

        self.source_name = get_language_name(source_lang)
        self.target_name = get_language_name(target_lang)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self, lora_config: Optional[LoraConfig] = None) -> None:
        """
        Load model, tokenizer, and apply LoRA.

        Args:
            lora_config: Custom LoRA configuration
        """
        logger.info(f"Loading model: {self.model_name}")

        # Quantization config for QLoRA
        if self.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"

        # Prepare for k-bit training
        if self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA/DoRA
        if lora_config is None:
            lora_config = get_lora_config(rank=self.lora_rank, use_dora=self.use_dora)

        self.model = get_peft_model(self.model, lora_config)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        method = "DoRA" if self.use_dora else "LoRA"
        logger.info(f"{method} applied (rank={self.lora_rank}, use_dora={self.use_dora})")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    def _format_example(self, source: str, target: str) -> str:
        """Format a single training example."""
        instruction = f"Translate the following {self.source_name} scientific text to {self.target_name}."

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "user", "content": f"{instruction}\n\n{source}"},
                {"role": "assistant", "content": target},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{source}\n\n### Response:\n{target}"

    def prepare_dataset(
        self,
        dataset: AfriScienceDataset,
        max_length: int = 512,
    ) -> HFDataset:
        """
        Prepare dataset for training.

        Args:
            dataset: AfriScience dataset
            max_length: Maximum sequence length

        Returns:
            HuggingFace Dataset
        """
        # Format all examples
        texts = []
        for i in range(len(dataset)):
            sample = dataset[i]
            text = self._format_example(sample["source"], sample["target"])
            texts.append(text)

        return HFDataset.from_dict({"text": texts})

    def train(
        self,
        train_dataset: AfriScienceDataset,
        eval_dataset: Optional[AfriScienceDataset] = None,
        training_args: Optional[Dict[str, Any]] = None,
        max_length: int = 512,
        resume_from_checkpoint: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model with LoRA.

        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            training_args: Training arguments
            max_length: Maximum sequence length
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training results
        """
        if self.model is None:
            self.setup()

        # Default training arguments
        default_args = {
            "output_dir": str(self.output_dir),
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_ratio": 0.03,
            "weight_decay": 0.001,
            "max_grad_norm": 0.3,
            "optim": "paged_adamw_32bit",
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_strategy": "steps",
            "save_steps": 100,
            "save_total_limit": 2,  # Keep 2 checkpoints during training
            "eval_strategy": "steps" if eval_dataset else "no",
            "eval_steps": 100 if eval_dataset else None,
            "load_best_model_at_end": True if eval_dataset else False,
            "bf16": True,
            "report_to": "none",
        }

        if training_args:
            default_args.update(training_args)

        # SFT-specific args go into SFTConfig (not TrainingArguments)
        default_args["dataset_text_field"] = "text"
        default_args["max_length"] = max_length
        default_args["packing"] = False

        args = SFTConfig(**default_args)

        # Prepare datasets
        train_hf = self.prepare_dataset(train_dataset, max_length)
        eval_hf = self.prepare_dataset(eval_dataset, max_length) if eval_dataset else None

        # Create SFT trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=train_hf,
            eval_dataset=eval_hf,
            processing_class=self.tokenizer,
        )

        # Train
        logger.info("Starting LoRA training...")
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save adapter
        adapter_path = self.output_dir / "adapter"
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        logger.info(f"Training completed. Adapter saved to {adapter_path}")

        # Clean up checkpoint directories to save space
        self._cleanup_checkpoints()

        return {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime"),
            "lora_rank": self.lora_rank,
            "use_dora": self.use_dora,
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

    def save_adapter(self, path: Optional[Union[str, Path]] = None) -> None:
        """Save LoRA adapter."""
        path = Path(path) if path else self.output_dir / "adapter"
        path.mkdir(parents=True, exist_ok=True)

        if self.model:
            self.model.save_pretrained(path)
        if self.tokenizer:
            self.tokenizer.save_pretrained(path)

    def merge_and_save(self, path: Union[str, Path]) -> None:
        """Merge LoRA weights with base model and save."""
        if self.model is None:
            raise ValueError("Model not loaded")

        merged = self.model.merge_and_unload()
        merged.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def train_lora(
    model_name: str,
    train_data_path: Union[str, Path],
    source_lang: str,
    target_lang: str,
    output_dir: Union[str, Path],
    lora_rank: int = 16,
    eval_data_path: Optional[Union[str, Path]] = None,
    load_in_4bit: bool = True,
    training_args: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to train a model with LoRA.

    Args:
        model_name: HuggingFace model name
        train_data_path: Path to training data
        source_lang: Source language code
        target_lang: Target language code
        output_dir: Output directory
        lora_rank: LoRA rank
        eval_data_path: Path to evaluation data
        load_in_4bit: Use 4-bit quantization
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
    trainer = LoRATrainer(
        model_name=model_name,
        source_lang=source_lang,
        target_lang=target_lang,
        output_dir=output_dir,
        lora_rank=lora_rank,
        load_in_4bit=load_in_4bit,
        **kwargs,
    )

    # Train
    return trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_args=training_args,
    )
