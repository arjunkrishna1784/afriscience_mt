"""Seq2Seq model wrappers (M2M100, NLLB)."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    NllbTokenizer,
)

from .base import BaseTranslationModel, ModelConfig
from ..utils.language_utils import get_nllb_code, get_m2m100_code


class Seq2SeqModel(BaseTranslationModel):
    """Base wrapper for seq2seq translation models."""

    def __init__(self, config: ModelConfig):
        """
        Initialize seq2seq model.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self.generation_config = {
            "max_length": config.max_length,
            "num_beams": 5,
            "early_stopping": True,
        }

    def load(self) -> None:
        """Load model and tokenizer."""
        model_kwargs = {}

        # Set precision
        if self.config.precision == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.config.precision == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16

        if self.config.cache_dir:
            model_kwargs["cache_dir"] = self.config.cache_dir

        # Load model
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.name,
            **model_kwargs
        )
        self.model.to(self.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            cache_dir=self.config.cache_dir,
        )

    def _set_language_codes(self, source_lang: str, target_lang: str) -> None:
        """Set source and target language codes on tokenizer."""
        raise NotImplementedError("Subclasses must implement _set_language_codes")

    def translate(
        self,
        texts: Union[str, List[str]],
        source_lang: str,
        target_lang: str,
        **kwargs,
    ) -> Union[str, List[str]]:
        """
        Translate text(s).

        Args:
            texts: Input text or list of texts
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional generation arguments

        Returns:
            Translated text(s)
        """
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        translations = self.translate_batch(
            texts, source_lang, target_lang, batch_size=len(texts), **kwargs
        )

        return translations[0] if single_input else translations

    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        batch_size: int = 8,
        **kwargs,
    ) -> List[str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of input texts
            source_lang: Source language code
            target_lang: Target language code
            batch_size: Batch size for processing
            **kwargs: Additional generation arguments

        Returns:
            List of translated texts
        """
        self._set_language_codes(source_lang, target_lang)

        translations = []
        generation_config = {**self.generation_config, **kwargs}

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)

            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    **generation_config,
                )

            # Decode
            batch_translations = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            translations.extend(batch_translations)

        return translations


class M2M100Model(Seq2SeqModel):
    """Wrapper for M2M100 models."""

    def load(self) -> None:
        """Load M2M100 model and tokenizer."""
        model_kwargs = {}

        if self.config.precision == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.config.precision == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16

        if self.config.cache_dir:
            model_kwargs["cache_dir"] = self.config.cache_dir

        self.model = M2M100ForConditionalGeneration.from_pretrained(
            self.config.name,
            **model_kwargs
        )
        self.model.to(self.device)

        self.tokenizer = M2M100Tokenizer.from_pretrained(
            self.config.name,
            cache_dir=self.config.cache_dir,
        )

    def _set_language_codes(self, source_lang: str, target_lang: str) -> None:
        """Set M2M100 language codes."""
        src_code = get_m2m100_code(source_lang)
        tgt_code = get_m2m100_code(target_lang)

        self.tokenizer.src_lang = src_code
        self.generation_config["forced_bos_token_id"] = self.tokenizer.get_lang_id(tgt_code)


class NLLBModel(Seq2SeqModel):
    """Wrapper for NLLB models."""

    def load(self) -> None:
        """Load NLLB model and tokenizer."""
        model_kwargs = {}

        if self.config.precision == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
        elif self.config.precision == "bf16":
            model_kwargs["torch_dtype"] = torch.bfloat16

        if self.config.cache_dir:
            model_kwargs["cache_dir"] = self.config.cache_dir

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.name,
            **model_kwargs
        )
        self.model.to(self.device)

        self.tokenizer = NllbTokenizer.from_pretrained(
            self.config.name,
            cache_dir=self.config.cache_dir,
        )

    def _set_language_codes(self, source_lang: str, target_lang: str) -> None:
        """Set NLLB language codes."""
        src_code = get_nllb_code(source_lang)
        tgt_code = get_nllb_code(target_lang)

        self.tokenizer.src_lang = src_code
        self.generation_config["forced_bos_token_id"] = self.tokenizer.convert_tokens_to_ids(tgt_code)


def load_seq2seq_model(
    model_name: str,
    model_type: Optional[str] = None,
    device: str = "auto",
    precision: str = "fp16",
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Seq2SeqModel:
    """
    Load a seq2seq translation model.

    Args:
        model_name: Model name or path
        model_type: Model type ("m2m100", "nllb"). Auto-detected if None.
        device: Device to use
        precision: Precision (fp32, fp16, bf16)
        cache_dir: Cache directory
        **kwargs: Additional config options

    Returns:
        Loaded model wrapper
    """
    # Auto-detect model type
    if model_type is None:
        model_name_lower = model_name.lower()
        if "m2m100" in model_name_lower:
            model_type = "m2m100"
        elif "nllb" in model_name_lower:
            model_type = "nllb"
        else:
            raise ValueError(f"Cannot auto-detect model type for: {model_name}")

    config = ModelConfig(
        name=model_name,
        model_type=model_type,
        device=device,
        precision=precision,
        cache_dir=cache_dir,
        extra=kwargs,
    )

    if model_type == "m2m100":
        model = M2M100Model(config)
    elif model_type == "nllb":
        model = NLLBModel(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load()
    return model
