"""Multilingual training utilities for N-to-N translation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler

from ..data.dataset import AfriScienceDataset
from ..utils.language_utils import LANGUAGE_CODES, get_language_name
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MultilingualDataset(Dataset):
    """
    Dataset that combines multiple language pairs for N-to-N training.

    Supports various data mixing strategies:
    - proportional: Sample proportionally to dataset size
    - equal: Equal samples from each language pair
    - temperature_sampling: Temperature-based sampling for balance
    - weighted: Custom weights per direction type
    """

    def __init__(
        self,
        datasets: Dict[str, AfriScienceDataset],
        mixing_strategy: str = "temperature_sampling",
        temperature: float = 5.0,
        direction_weights: Optional[Dict[str, float]] = None,
        use_language_tags: bool = True,
        tag_format: str = "[{lang}]",
        tag_position: str = "prefix",
        african_languages: Optional[List[str]] = None,
    ):
        """
        Initialize multilingual dataset.

        Args:
            datasets: Dictionary mapping language pairs to datasets
            mixing_strategy: How to mix datasets
            temperature: Temperature for temperature-based sampling
            direction_weights: Weights for different translation directions
            use_language_tags: Whether to add language tags to inputs
            tag_format: Format string for language tags
            tag_position: Where to add tags ("prefix" or "suffix")
            african_languages: List of African language codes for weighting
        """
        self.datasets = datasets
        self.mixing_strategy = mixing_strategy
        self.temperature = temperature
        self.direction_weights = direction_weights or {}
        self.use_language_tags = use_language_tags
        self.tag_format = tag_format
        self.tag_position = tag_position
        self.african_languages = african_languages or ["amh", "hau", "lug", "nso", "yor", "zul"]

        # Compute sampling weights
        self.lang_pairs = list(datasets.keys())
        self.weights = self._compute_weights()

        # Build flat index
        self.samples = self._build_sample_index()

        # Statistics
        self._log_statistics()

    def _compute_weights(self) -> Dict[str, float]:
        """Compute sampling weights for each language pair."""
        sizes = {lp: len(ds) for lp, ds in self.datasets.items()}
        total = sum(sizes.values())

        if self.mixing_strategy == "equal":
            # Equal weight for all language pairs
            n = len(self.lang_pairs)
            return {lp: 1.0 / n for lp in self.lang_pairs}

        elif self.mixing_strategy == "temperature_sampling":
            # Temperature-scaled sampling
            probs = {}
            for lp, size in sizes.items():
                base_prob = (size / total) ** (1.0 / self.temperature)

                # Apply direction weights
                direction_weight = self._get_direction_weight(lp)
                probs[lp] = base_prob * direction_weight

            # Normalize
            total_prob = sum(probs.values())
            return {lp: p / total_prob for lp, p in probs.items()}

        elif self.mixing_strategy == "weighted":
            # Use direction weights directly
            probs = {}
            for lp, size in sizes.items():
                direction_weight = self._get_direction_weight(lp)
                probs[lp] = (size / total) * direction_weight

            total_prob = sum(probs.values())
            return {lp: p / total_prob for lp, p in probs.items()}

        else:  # proportional
            return {lp: size / total for lp, size in sizes.items()}

    def _get_direction_weight(self, lang_pair: str) -> float:
        """Get weight for a translation direction."""
        src, tgt = lang_pair.split("-")

        # Check explicit weights for this pair
        if lang_pair in self.direction_weights:
            return self.direction_weights[lang_pair]

        # Check direction type weights
        src_is_african = src in self.african_languages
        tgt_is_african = tgt in self.african_languages

        if src_is_african and tgt_is_african:
            return self.direction_weights.get("african_to_african", 1.0)
        elif src == "eng" and tgt_is_african:
            return self.direction_weights.get("eng_to_african", 1.0)
        elif src_is_african and tgt == "eng":
            return self.direction_weights.get("african_to_eng", 1.0)

        return 1.0

    def _build_sample_index(self) -> List[Tuple[str, int]]:
        """Build a flat list of (lang_pair, index) tuples."""
        samples = []
        for lang_pair, dataset in self.datasets.items():
            for i in range(len(dataset)):
                samples.append((lang_pair, i))
        return samples

    def _log_statistics(self) -> None:
        """Log dataset statistics."""
        logger.info(f"Created multilingual dataset:")
        logger.info(f"  Total samples: {len(self.samples)}")
        logger.info(f"  Language pairs: {len(self.lang_pairs)}")
        logger.info(f"  Mixing strategy: {self.mixing_strategy}")

        # Count by direction type
        eng_to_afr = sum(1 for lp in self.lang_pairs if lp.startswith("eng-"))
        afr_to_eng = sum(1 for lp in self.lang_pairs if lp.endswith("-eng") and not lp.startswith("eng"))
        afr_to_afr = len(self.lang_pairs) - eng_to_afr - afr_to_eng

        logger.info(f"  Directions: eng->afr={eng_to_afr}, afr->eng={afr_to_eng}, afr->afr={afr_to_afr}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lang_pair, sample_idx = self.samples[idx]
        sample = self.datasets[lang_pair][sample_idx].copy()

        source = sample["source"]
        target = sample["target"]
        src_lang = sample["source_lang"]
        tgt_lang = sample["target_lang"]

        # Add language tags if enabled
        if self.use_language_tags:
            src_tag = self.tag_format.format(lang=src_lang)
            tgt_tag = self.tag_format.format(lang=tgt_lang)

            if self.tag_position == "prefix":
                source = f"{src_tag} {source}"
                target = f"{tgt_tag} {target}"
            else:
                source = f"{source} {src_tag}"
                target = f"{target} {tgt_tag}"

            sample["source"] = source
            sample["target"] = target

        sample["lang_pair"] = lang_pair
        return sample

    def get_sampler_weights(self) -> List[float]:
        """Get per-sample weights for weighted sampling."""
        weights = []
        for lang_pair, _ in self.samples:
            weights.append(self.weights[lang_pair])
        return weights

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Create a weighted random sampler."""
        weights = self.get_sampler_weights()
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(self.samples),
            replacement=True,
        )

    def get_pairs_by_direction(self) -> Dict[str, List[str]]:
        """Group language pairs by direction type."""
        directions = {
            "eng_to_african": [],
            "african_to_eng": [],
            "african_to_african": [],
        }

        for lp in self.lang_pairs:
            src, tgt = lp.split("-")
            if src == "eng" and tgt in self.african_languages:
                directions["eng_to_african"].append(lp)
            elif src in self.african_languages and tgt == "eng":
                directions["african_to_eng"].append(lp)
            elif src in self.african_languages and tgt in self.african_languages:
                directions["african_to_african"].append(lp)

        return directions


def generate_all_language_pairs(
    languages: List[str],
    exclude_same: bool = True,
    mode: str = "n_to_n",
    source_languages: Optional[List[str]] = None,
    target_languages: Optional[List[str]] = None,
) -> List[str]:
    """
    Generate language pairs based on mode.

    Args:
        languages: List of all languages
        exclude_same: Exclude same-language pairs (e.g., eng-eng)
        mode: Generation mode:
            - "n_to_n": All combinations (N*(N-1) pairs)
            - "one_to_many": One source to all targets
            - "many_to_one": All sources to one target
            - "many_to_many": Custom source/target lists
        source_languages: Source languages for many_to_many mode
        target_languages: Target languages for many_to_many mode

    Returns:
        List of language pair strings
    """
    pairs = []

    if mode == "n_to_n":
        # All combinations
        for src in languages:
            for tgt in languages:
                if exclude_same and src == tgt:
                    continue
                pairs.append(f"{src}-{tgt}")

    elif mode == "one_to_many":
        # English to all others (common setup)
        src = "eng"
        for tgt in languages:
            if tgt != src:
                pairs.append(f"{src}-{tgt}")

    elif mode == "many_to_one":
        # All others to English
        tgt = "eng"
        for src in languages:
            if src != tgt:
                pairs.append(f"{src}-{tgt}")

    elif mode == "many_to_many":
        # Custom source and target lists
        src_langs = source_languages or languages
        tgt_langs = target_languages or languages

        for src in src_langs:
            for tgt in tgt_langs:
                if exclude_same and src == tgt:
                    continue
                pairs.append(f"{src}-{tgt}")

    return list(set(pairs))  # Remove duplicates


def get_all_language_pairs(
    source_langs: List[str] = ["eng"],
    target_langs: Optional[List[str]] = None,
    bidirectional: bool = True,
) -> List[str]:
    """
    Generate all language pairs (backward compatible function).

    Args:
        source_langs: Source languages
        target_langs: Target languages (defaults to all African languages)
        bidirectional: Include reverse direction

    Returns:
        List of language pair strings
    """
    if target_langs is None:
        target_langs = [l for l in LANGUAGE_CODES if l != "eng"]

    pairs = []

    for src in source_langs:
        for tgt in target_langs:
            if src != tgt:
                pairs.append(f"{src}-{tgt}")
                if bidirectional:
                    pairs.append(f"{tgt}-{src}")

    return list(set(pairs))  # Remove duplicates


def create_multilingual_dataset(
    data_path: Union[str, Path],
    languages: Optional[List[str]] = None,
    language_pairs: Optional[List[str]] = None,
    split: str = "train",
    mode: str = "n_to_n",
    mixing_strategy: str = "temperature_sampling",
    temperature: float = 5.0,
    direction_weights: Optional[Dict[str, float]] = None,
    use_language_tags: bool = True,
    **kwargs,
) -> MultilingualDataset:
    """
    Create a multilingual dataset from multiple language pairs.

    Args:
        data_path: Path to data directory
        languages: List of languages (generates all pairs if language_pairs not provided)
        language_pairs: Explicit list of language pairs (e.g., ["eng-hau", "eng-yor"])
        split: Data split to load
        mode: Pair generation mode for N-to-N training
        mixing_strategy: Data mixing strategy
        temperature: Temperature for temperature sampling
        direction_weights: Weights for translation directions
        use_language_tags: Whether to use language tags
        **kwargs: Additional arguments for AfriScienceDataset

    Returns:
        MultilingualDataset instance
    """
    # Generate language pairs if not provided
    if language_pairs is None:
        if languages is None:
            languages = ["eng", "amh", "hau", "lug", "nso", "yor", "zul"]
        language_pairs = generate_all_language_pairs(
            languages=languages,
            mode=mode,
        )

    logger.info(f"Creating multilingual dataset with {len(language_pairs)} language pairs")

    datasets = {}
    for lang_pair in language_pairs:
        src, tgt = lang_pair.split("-")
        try:
            dataset = AfriScienceDataset(
                data_path=data_path,
                source_lang=src,
                target_lang=tgt,
                split=split,
                **kwargs,
            )
            datasets[lang_pair] = dataset
            logger.info(f"  {lang_pair}: {len(dataset)} samples")
        except FileNotFoundError as e:
            logger.warning(f"  {lang_pair}: Not found - {e}")

    if not datasets:
        raise ValueError("No datasets could be loaded")

    return MultilingualDataset(
        datasets=datasets,
        mixing_strategy=mixing_strategy,
        temperature=temperature,
        direction_weights=direction_weights,
        use_language_tags=use_language_tags,
    )


def create_n2n_dataset(
    data_path: Union[str, Path],
    languages: List[str] = ["eng", "amh", "hau", "lug", "nso", "yor", "zul"],
    split: str = "train",
    temperature: float = 5.0,
    upweight_african_to_african: float = 1.5,
    **kwargs,
) -> MultilingualDataset:
    """
    Convenience function for N-to-N multilingual training.

    Creates a dataset with all language pair combinations,
    with optional upweighting for African-to-African pairs.

    Args:
        data_path: Path to data directory
        languages: List of all languages
        split: Data split
        temperature: Sampling temperature
        upweight_african_to_african: Weight multiplier for afr->afr pairs
        **kwargs: Additional arguments

    Returns:
        MultilingualDataset for N-to-N training
    """
    direction_weights = {
        "eng_to_african": 1.0,
        "african_to_eng": 1.0,
        "african_to_african": upweight_african_to_african,
    }

    return create_multilingual_dataset(
        data_path=data_path,
        languages=languages,
        split=split,
        mode="n_to_n",
        mixing_strategy="temperature_sampling",
        temperature=temperature,
        direction_weights=direction_weights,
        use_language_tags=True,
        **kwargs,
    )


class MultilingualCollator:
    """Collator for multilingual datasets."""

    def __init__(
        self,
        tokenizer,
        max_source_length: int = 256,
        max_target_length: int = 256,
        model_type: str = "seq2seq",
        padding: str = "max_length",
    ):
        """
        Initialize collator.

        Args:
            tokenizer: Tokenizer instance
            max_source_length: Max source sequence length
            max_target_length: Max target sequence length
            model_type: Model type ("seq2seq" or "llm")
            padding: Padding strategy
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.padding = padding

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        sources = [item["source"] for item in batch]
        targets = [item["target"] for item in batch]

        if self.model_type == "seq2seq":
            model_inputs = self.tokenizer(
                sources,
                max_length=self.max_source_length,
                truncation=True,
                padding=self.padding,
                return_tensors="pt",
            )

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=self.max_target_length,
                    truncation=True,
                    padding=self.padding,
                    return_tensors="pt",
                )

            model_inputs["labels"] = labels["input_ids"]
            model_inputs["labels"][model_inputs["labels"] == self.tokenizer.pad_token_id] = -100

            return model_inputs

        else:  # LLM
            # Format as instruction-response pairs
            texts = []
            for src, tgt in zip(sources, targets):
                text = f"### Source:\n{src}\n\n### Translation:\n{tgt}"
                texts.append(text)

            model_inputs = self.tokenizer(
                texts,
                max_length=self.max_source_length + self.max_target_length,
                truncation=True,
                padding=self.padding,
                return_tensors="pt",
            )

            model_inputs["labels"] = model_inputs["input_ids"].clone()
            model_inputs["labels"][model_inputs["labels"] == self.tokenizer.pad_token_id] = -100

            return model_inputs


def get_zero_shot_pairs(
    all_languages: List[str],
    training_pairs: List[str],
) -> List[str]:
    """
    Get language pairs not seen during training (for zero-shot evaluation).

    Args:
        all_languages: All languages in the system
        training_pairs: Pairs used for training

    Returns:
        List of zero-shot pairs
    """
    all_possible = generate_all_language_pairs(all_languages, mode="n_to_n")
    training_set = set(training_pairs)

    return [p for p in all_possible if p not in training_set]


def get_african_to_african_pairs(
    african_languages: List[str] = ["amh", "hau", "lug", "nso", "yor", "zul"],
) -> List[str]:
    """Get all African-to-African language pairs."""
    return generate_all_language_pairs(african_languages, mode="n_to_n")
