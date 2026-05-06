"""Hugging Face model upload utilities with Collection support."""

import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from huggingface_hub import (
    HfApi,
    create_repo,
    upload_folder,
    login,
    whoami,
)

from .logging import get_logger

logger = get_logger(__name__)


# Collection definitions for organizing models
COLLECTION_CONFIGS = {
    "seq2seq": {
        "title": "AfriScience-MT Seq2Seq Models",
        "description": "Fine-tuned M2M100 and NLLB models for scientific machine translation between English and African languages (Amharic, Hausa, Luganda, Northern Sotho, Yoruba, isiZulu).",
        "namespace": "seq2seq",
    },
    "lora": {
        "title": "AfriScience-MT LoRA Adapters",
        "description": "LoRA adapters for Gemma and Llama models, fine-tuned for scientific machine translation between English and African languages.",
        "namespace": "lora",
    },
    "dora": {
        "title": "AfriScience-MT DoRA Adapters",
        "description": "DoRA (Weight-Decomposed Low-Rank Adaptation) adapters for Gemma and Llama models, fine-tuned for scientific machine translation between English and African languages.",
        "namespace": "dora",
    },
    "multilingual": {
        "title": "AfriScience-MT Multilingual N-to-N",
        "description": "Multilingual models supporting translation between any pair of 7 languages (English + 6 African languages) in a single model.",
        "namespace": "multilingual",
    },
    "all": {
        "title": "AfriScience-MT Complete Collection",
        "description": "Complete collection of all AfriScience-MT models for scientific machine translation involving African languages.",
        "namespace": "all",
    },
}


class HFCollectionManager:
    """Manage HuggingFace Collections for organizing models."""

    def __init__(
        self,
        organization: str,
        token: Optional[str] = None,
    ):
        """
        Initialize collection manager.

        Args:
            organization: HF organization/username
            token: HF API token
        """
        self.organization = organization
        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)
        self._collection_cache = {}

    def get_or_create_collection(
        self,
        collection_type: str = "all",
    ) -> Optional[str]:
        """
        Get or create a collection for the given type.

        Args:
            collection_type: Type of collection (seq2seq, lora, multilingual, all)

        Returns:
            Collection slug if successful
        """
        if collection_type in self._collection_cache:
            return self._collection_cache[collection_type]

        config = COLLECTION_CONFIGS.get(collection_type, COLLECTION_CONFIGS["all"])

        try:
            # Try to create the collection
            slug = f"{self.organization}/afriscience-mt-{config['namespace']}"

            collection = self.api.create_collection(
                title=config["title"],
                description=config["description"],
                namespace=self.organization,
                private=False,
                exists_ok=True,
                token=self.token,
            )

            self._collection_cache[collection_type] = collection.slug
            logger.info(f"Collection ready: {collection.slug}")
            return collection.slug

        except Exception as e:
            logger.warning(f"Could not create/get collection: {e}")
            return None

    def add_model_to_collection(
        self,
        repo_id: str,
        collection_type: str = "all",
        note: Optional[str] = None,
    ) -> bool:
        """
        Add a model to a collection.

        Args:
            repo_id: Full repository ID (org/repo-name)
            collection_type: Type of collection to add to
            note: Optional note about the model

        Returns:
            True if successful
        """
        collection_slug = self.get_or_create_collection(collection_type)

        if not collection_slug:
            return False

        try:
            self.api.add_collection_item(
                collection_slug=collection_slug,
                item_id=repo_id,
                item_type="model",
                note=note,
                token=self.token,
                exists_ok=True,
            )
            logger.info(f"Added {repo_id} to collection {collection_slug}")
            return True

        except Exception as e:
            logger.warning(f"Could not add to collection: {e}")
            return False

    def add_to_multiple_collections(
        self,
        repo_id: str,
        collection_types: List[str],
        note: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Add a model to multiple collections.

        Args:
            repo_id: Full repository ID
            collection_types: List of collection types
            note: Optional note

        Returns:
            Dictionary mapping collection types to success status
        """
        results = {}
        for ctype in collection_types:
            results[ctype] = self.add_model_to_collection(repo_id, ctype, note)
        return results


class HFUploader:
    """Upload trained models to Hugging Face Hub with Collection support."""

    def __init__(
        self,
        organization: Optional[str] = None,
        token: Optional[str] = None,
        private: bool = False,
        add_to_collection: bool = True,
    ):
        """
        Initialize HF uploader.

        Args:
            organization: HF organization/username to upload to
            token: HF API token (defaults to HF_TOKEN env var)
            private: Whether to create private repos
            add_to_collection: Whether to add models to collections
        """
        self.token = token or os.environ.get("HF_TOKEN")
        self.private = private
        self.add_to_collection = add_to_collection
        self.api = HfApi(token=self.token)

        # Get organization/username
        if organization:
            self.organization = organization
        else:
            try:
                user_info = whoami(token=self.token)
                self.organization = user_info.get("name", user_info.get("username"))
            except Exception as e:
                logger.warning(f"Could not get HF username: {e}")
                self.organization = None

        # Initialize collection manager
        if self.organization and self.add_to_collection:
            self.collection_manager = HFCollectionManager(
                organization=self.organization,
                token=self.token,
            )
        else:
            self.collection_manager = None

    def create_model_card(
        self,
        model_name: str,
        model_type: str,
        source_lang: str,
        target_lang: str,
        base_model: str,
        metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a comprehensive model card (README.md) for the uploaded model."""

        # Language mapping
        lang_names = {
            "eng": "English",
            "amh": "Amharic",
            "hau": "Hausa",
            "lug": "Luganda",
            "nso": "Northern Sotho",
            "yor": "Yoruba",
            "zul": "isiZulu",
        }

        lang_codes = {
            "eng": "en", "amh": "am", "hau": "ha",
            "lug": "lg", "nso": "nso", "yor": "yo", "zul": "zu"
        }

        src_name = lang_names.get(source_lang, source_lang)
        tgt_name = lang_names.get(target_lang, target_lang)
        is_multilingual = source_lang == "multilingual" or target_lang == "multilingual"

        # Determine model family for usage examples
        is_nllb = "nllb" in base_model.lower()
        is_m2m = "m2m" in base_model.lower()

        # Build YAML frontmatter
        card = f"""---
language:
"""
        if is_multilingual:
            card += "- en\n- am\n- ha\n- lg\n- nso\n- yo\n- zu\n"
        else:
            card += f"- {lang_codes.get(source_lang, source_lang)}\n"
            card += f"- {lang_codes.get(target_lang, target_lang)}\n"

        card += f"""tags:
- translation
- african-languages
- scientific-translation
- afriscience-mt
- {'nllb' if is_nllb else 'm2m100' if is_m2m else 'seq2seq'}
license: apache-2.0
base_model: {base_model}
datasets:
- afriscience-mt
pipeline_tag: translation
"""

        # Add metrics to YAML for HF model card display
        if metrics:
            card += """model-index:
- name: """ + model_name + """
  results:
  - task:
      type: translation
    metrics:
"""
            if metrics.get("test_bleu") is not None:
                card += f"""    - name: BLEU (test)
      type: bleu
      value: {metrics.get('test_bleu', 0):.2f}
"""
            if metrics.get("test_chrf") is not None:
                card += f"""    - name: chrF (test)
      type: chrf
      value: {metrics.get('test_chrf', 0):.2f}
"""
            if metrics.get("test_ssa_comet") is not None:
                card += f"""    - name: SSA-COMET (test)
      type: comet
      value: {metrics.get('test_ssa_comet', 0) * 100:.2f}
"""

        card += """---

"""
        # Model title and badges
        card += f"""# {model_name}

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/raw/main/model-on-hf-sm.svg)](https://huggingface.co/{self.organization}/{model_name})

This model is part of the **AfriScience-MT** project, focused on machine translation of scientific texts for African languages.

## Model Description

"""
        if is_multilingual:
            card += f"""| Property | Value |
|----------|-------|
| **Model Type** | {model_type} (Multilingual N-to-N) |
| **Languages** | English, Amharic, Hausa, Luganda, Northern Sotho, Yoruba, isiZulu |
| **Base Model** | [{base_model}](https://huggingface.co/{base_model}) |
| **Domain** | Scientific/Academic texts |
| **Training** | Full fine-tuning on AfriScience-MT dataset |
"""
        else:
            card += f"""| Property | Value |
|----------|-------|
| **Model Type** | Seq2Seq Translation |
| **Translation Direction** | {src_name} → {tgt_name} |
| **Base Model** | [{base_model}](https://huggingface.co/{base_model}) |
| **Domain** | Scientific/Academic texts |
| **Training** | Full fine-tuning on AfriScience-MT dataset |
"""

        # Evaluation Results
        if metrics:
            card += """
## Evaluation Results

Performance on the AfriScience-MT test set:

| Split | BLEU | chrF | SSA-COMET |
|-------|------|------|-----------|
"""
            val_bleu = metrics.get('val_bleu', '-')
            val_chrf = metrics.get('val_chrf', '-')
            val_comet = metrics.get('val_ssa_comet', '-')
            test_bleu = metrics.get('test_bleu', '-')
            test_chrf = metrics.get('test_chrf', '-')
            test_comet = metrics.get('test_ssa_comet', '-')

            # Format COMET as percentage (multiply by 100)
            def fmt_comet(v):
                return f'{v * 100:.2f}' if isinstance(v, float) else v

            if val_bleu != '-' or val_chrf != '-':
                card += f"| Validation | {f'{val_bleu:.2f}' if isinstance(val_bleu, float) else val_bleu} | {f'{val_chrf:.2f}' if isinstance(val_chrf, float) else val_chrf} | {fmt_comet(val_comet)} |\n"
            card += f"| **Test** | **{f'{test_bleu:.2f}' if isinstance(test_bleu, float) else test_bleu}** | **{f'{test_chrf:.2f}' if isinstance(test_chrf, float) else test_chrf}** | **{fmt_comet(test_comet)}** |\n"

            card += """
**Metrics explanation:**
- **BLEU**: Measures n-gram overlap with reference translations (0-100, higher is better)
- **chrF**: Character-level F-score, robust for morphologically rich languages (0-100, higher is better)
- **SSA-COMET**: Neural metric trained for Sub-Saharan African languages, shown as percentage (0-100, higher is better) ([McGill-NLP/ssa-comet-stl](https://huggingface.co/McGill-NLP/ssa-comet-stl))
"""

        # Usage section with proper code examples
        card += """
## Usage

### Quick Start

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_id = \"""" + f"{self.organization}/{model_name}" + """\"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
"""

        if is_nllb:
            # NLLB-specific usage
            nllb_src = {"eng": "eng_Latn", "amh": "amh_Ethi", "hau": "hau_Latn",
                       "lug": "lug_Latn", "nso": "nso_Latn", "yor": "yor_Latn", "zul": "zul_Latn"}
            nllb_tgt = nllb_src.get(target_lang, "eng_Latn")
            nllb_src_code = nllb_src.get(source_lang, "eng_Latn")

            card += f"""
# Set source language
tokenizer.src_lang = "{nllb_src_code}"

# Translate
text = "The mitochondria is the powerhouse of the cell."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

# Generate with target language
forced_bos_token_id = tokenizer.convert_tokens_to_ids("{nllb_tgt}")
outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=256, num_beams=5)
translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(translation)
```
"""
        elif is_m2m:
            # M2M100-specific usage
            m2m_codes = {"eng": "en", "amh": "am", "hau": "ha", "lug": "lg",
                        "nso": "ns", "yor": "yo", "zul": "zu"}
            m2m_src = m2m_codes.get(source_lang, "en")
            m2m_tgt = m2m_codes.get(target_lang, "en")

            card += f"""
# Set source language
tokenizer.src_lang = "{m2m_src}"

# Translate
text = "The mitochondria is the powerhouse of the cell."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)

# Generate with target language
forced_bos_token_id = tokenizer.get_lang_id("{m2m_tgt}")
outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=256, num_beams=5)
translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(translation)
```
"""
        else:
            card += """
# Translate
text = "The mitochondria is the powerhouse of the cell."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
outputs = model.generate(**inputs, max_length=256, num_beams=5)
translation = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
print(translation)
```
"""

        card += """
### Batch Translation

```python
texts = [
    "Climate change affects agricultural productivity.",
    "The study analyzed genetic markers in the population.",
    "Renewable energy sources are essential for sustainable development."
]

inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
"""
        if is_nllb:
            card += f"""outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=256, num_beams=5)
"""
        elif is_m2m:
            card += f"""outputs = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=256, num_beams=5)
"""
        else:
            card += """outputs = model.generate(**inputs, max_length=256, num_beams=5)
"""
        card += """translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)
for src, tgt in zip(texts, translations):
    print(f"{src}\\n→ {tgt}\\n")
```
"""

        # Training configuration
        if training_config:
            card += """
## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
"""
            config_display = {
                "num_epochs": "Epochs",
                "batch_size": "Batch Size",
                "learning_rate": "Learning Rate",
                "max_length": "Max Sequence Length",
                "warmup_steps": "Warmup Steps",
                "weight_decay": "Weight Decay",
                "gradient_accumulation_steps": "Gradient Accumulation",
                "fp16": "Mixed Precision (FP16)",
                "bf16": "Mixed Precision (BF16)",
            }

            for key, display_name in config_display.items():
                if key in training_config:
                    value = training_config[key]
                    if isinstance(value, float) and value < 0.01:
                        card += f"| {display_name} | {value:.0e} |\n"
                    else:
                        card += f"| {display_name} | {value} |\n"

            card += """
### Training Data

- **Dataset**: AfriScience-MT
- **Domain**: Scientific abstracts and papers
- **Languages**: English and 6 African languages (Amharic, Hausa, Luganda, Northern Sotho, Yoruba, isiZulu)

"""

        # Reproducibility section
        card += """
## Reproducibility

To reproduce this model:

```bash
# Clone the AfriScience-MT repository
git clone https://github.com/afriscience-mt/afriscience-mt.git
cd afriscience-mt

# Install dependencies
pip install -r requirements.txt

# Run training
python -m afriscience_mt.scripts.run_seq2seq_training \\
    --data_dir ./data \\
    --source_lang """ + source_lang + """ \\
    --target_lang """ + target_lang + """ \\
    --model_name """ + base_model + """ \\
    --model_type """ + ("nllb" if is_nllb else "m2m100" if is_m2m else "seq2seq") + """ \\
    --output_dir ./output \\
    --num_epochs 10 \\
    --batch_size 16 \\
    --learning_rate 2e-5
```

## Limitations

- **Domain Specificity**: This model is optimized for scientific/academic texts and may perform poorly on colloquial or informal text.
- **Language Coverage**: Only supports the specific language pair indicated.
- **Input Length**: Maximum input length is 256 tokens; longer texts should be split into segments.

## Citation

If you use this model, please cite the AfriScience-MT project:

```bibtex
@inproceedings{afriscience-mt-2025,
  title={AfriScience-MT: Machine Translation for African Scientific Literature},
  author={AfriScience-MT Team},
  year={2025},
  url={https://github.com/afriscience-mt/afriscience-mt}
}
```

## License

This model is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgments

- Built on top of [{base_model}](https://huggingface.co/{base_model})
- Evaluation using [SSA-COMET](https://huggingface.co/McGill-NLP/ssa-comet-stl) for African language assessment
"""
        return card

    def upload_model(
        self,
        model_path: str,
        repo_name: str,
        model_type: str = "seq2seq",
        source_lang: str = "eng",
        target_lang: str = "hau",
        base_model: str = "facebook/nllb-200-1.3B",
        metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        delete_local: bool = True,
        repo_prefix: str = "afriscience-mt",
        collection_types: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Upload a trained model to Hugging Face Hub.

        Args:
            model_path: Path to the model directory
            repo_name: Name for the repository
            model_type: Type of model (seq2seq, lora, etc.)
            source_lang: Source language code
            target_lang: Target language code
            base_model: Base model name
            metrics: Evaluation metrics
            training_config: Training configuration
            delete_local: Whether to delete local files after upload
            repo_prefix: Prefix for repository names
            collection_types: List of collection types to add model to

        Returns:
            Repository URL if successful, None otherwise
        """
        model_path = Path(model_path)

        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return None

        if not self.organization:
            logger.error("No HF organization/username configured")
            return None

        # Create full repo name (organization already has the project name, so no prefix needed)
        full_repo_name = f"{self.organization}/{repo_name}"

        logger.info(f"Uploading model to {full_repo_name}...")

        try:
            # Create repository
            create_repo(
                repo_id=full_repo_name,
                token=self.token,
                private=self.private,
                exist_ok=True,
            )

            # Create model card
            model_card = self.create_model_card(
                model_name=repo_name,
                model_type=model_type,
                source_lang=source_lang,
                target_lang=target_lang,
                base_model=base_model,
                metrics=metrics,
                training_config=training_config,
            )

            # Write model card
            readme_path = model_path / "README.md"
            with open(readme_path, "w") as f:
                f.write(model_card)

            # Upload folder
            upload_folder(
                folder_path=str(model_path),
                repo_id=full_repo_name,
                token=self.token,
                commit_message=f"Upload {repo_name} model",
            )

            repo_url = f"https://huggingface.co/{full_repo_name}"
            logger.info(f"Successfully uploaded to {repo_url}")

            # Add to collections
            if self.collection_manager and self.add_to_collection:
                # Determine collection types
                if collection_types is None:
                    collection_types = ["all"]
                    if "multilingual" in repo_name.lower() or source_lang == "multilingual":
                        collection_types.append("multilingual")
                    elif model_type == "seq2seq":
                        collection_types.append("seq2seq")

                # Create note for collection
                if source_lang == "multilingual":
                    note = f"Multilingual N-to-N model based on {base_model}"
                else:
                    note = f"{source_lang} → {target_lang} based on {base_model}"

                self.collection_manager.add_to_multiple_collections(
                    repo_id=full_repo_name,
                    collection_types=collection_types,
                    note=note,
                )

            # Delete local files if requested
            if delete_local:
                logger.info(f"Deleting local model files: {model_path}")
                shutil.rmtree(model_path)
                logger.info("Local files deleted")

            return repo_url

        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return None

    def upload_lora_adapter(
        self,
        adapter_path: str,
        repo_name: str,
        base_model: str,
        source_lang: str = "eng",
        target_lang: str = "hau",
        lora_rank: int = 32,
        metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        delete_local: bool = True,
        repo_prefix: str = "afriscience-mt",
        collection_types: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Upload a LoRA adapter to Hugging Face Hub.

        Args:
            adapter_path: Path to the LoRA adapter directory
            repo_name: Name for the repository
            base_model: Base model the adapter was trained on
            source_lang: Source language code
            target_lang: Target language code
            lora_rank: LoRA rank used
            metrics: Evaluation metrics
            training_config: Training configuration
            delete_local: Whether to delete local files after upload
            repo_prefix: Prefix for repository names
            collection_types: List of collection types to add model to

        Returns:
            Repository URL if successful, None otherwise
        """
        adapter_path = Path(adapter_path)

        if not adapter_path.exists():
            logger.error(f"Adapter path does not exist: {adapter_path}")
            return None

        if not self.organization:
            logger.error("No HF organization/username configured")
            return None

        # Create full repo name
        # Organization already has the project name, so no prefix needed
        full_repo_name = f"{self.organization}/{repo_name}"

        logger.info(f"Uploading LoRA adapter to {full_repo_name}...")

        try:
            # Create repository
            create_repo(
                repo_id=full_repo_name,
                token=self.token,
                private=self.private,
                exist_ok=True,
            )

            # Create adapter card
            adapter_card = self._create_lora_card(
                repo_name=repo_name,
                base_model=base_model,
                source_lang=source_lang,
                target_lang=target_lang,
                lora_rank=lora_rank,
                metrics=metrics,
                training_config=training_config,
            )

            # Write adapter card
            readme_path = adapter_path / "README.md"
            with open(readme_path, "w") as f:
                f.write(adapter_card)

            # Upload folder
            upload_folder(
                folder_path=str(adapter_path),
                repo_id=full_repo_name,
                token=self.token,
                commit_message=f"Upload {repo_name} LoRA adapter",
            )

            repo_url = f"https://huggingface.co/{full_repo_name}"
            logger.info(f"Successfully uploaded to {repo_url}")

            # Add to collections
            if self.collection_manager and self.add_to_collection:
                # Determine collection types
                if collection_types is None:
                    collection_types = ["all", "lora"]
                    if "multilingual" in repo_name.lower() or source_lang == "multilingual":
                        collection_types.append("multilingual")

                # Create note for collection
                adapter_method = "DoRA" if (collection_types and "dora" in collection_types) else "LoRA"
                if source_lang == "multilingual":
                    note = f"Multilingual {adapter_method} r{lora_rank} based on {base_model}"
                else:
                    note = f"{source_lang} → {target_lang} {adapter_method} r{lora_rank} based on {base_model}"

                self.collection_manager.add_to_multiple_collections(
                    repo_id=full_repo_name,
                    collection_types=collection_types,
                    note=note,
                )

            # Delete local files if requested
            if delete_local:
                logger.info(f"Deleting local adapter files: {adapter_path}")
                shutil.rmtree(adapter_path)
                logger.info("Local files deleted")

            return repo_url

        except Exception as e:
            logger.error(f"Failed to upload adapter: {e}")
            return None

    def _create_lora_card(
        self,
        repo_name: str,
        base_model: str,
        source_lang: str,
        target_lang: str,
        lora_rank: int,
        metrics: Optional[Dict[str, float]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a comprehensive model card for LoRA adapter."""

        lang_names = {
            "eng": "English",
            "amh": "Amharic",
            "hau": "Hausa",
            "lug": "Luganda",
            "nso": "Northern Sotho",
            "yor": "Yoruba",
            "zul": "isiZulu",
            "multilingual": "Multilingual",
        }

        lang_codes = {
            "eng": "en", "amh": "am", "hau": "ha",
            "lug": "lg", "nso": "nso", "yor": "yo", "zul": "zu"
        }

        src_name = lang_names.get(source_lang, source_lang)
        tgt_name = lang_names.get(target_lang, target_lang)
        is_multilingual = source_lang == "multilingual" or target_lang == "multilingual"

        # Determine model family
        is_gemma = "gemma" in base_model.lower()
        is_llama = "llama" in base_model.lower()

        # Build YAML frontmatter
        card = f"""---
library_name: peft
base_model: {base_model}
"""
        if is_multilingual:
            card += "language:\n- en\n- am\n- ha\n- lg\n- nso\n- yo\n- zu\n"
        else:
            card += f"language:\n- {lang_codes.get(source_lang, source_lang)}\n- {lang_codes.get(target_lang, target_lang)}\n"

        card += f"""tags:
- translation
- african-languages
- scientific-translation
- afriscience-mt
- lora
- peft
- {'gemma' if is_gemma else 'llama' if is_llama else 'causal-lm'}
license: apache-2.0
pipeline_tag: translation
"""

        # Add metrics to YAML — only if at least one eval metric exists
        eval_metric_entries = []
        if metrics:
            if metrics.get("test_bleu") is not None:
                eval_metric_entries.append(f"""    - name: BLEU (test)
      type: bleu
      value: {metrics.get('test_bleu', 0):.2f}
""")
            if metrics.get("test_chrf") is not None:
                eval_metric_entries.append(f"""    - name: chrF (test)
      type: chrf
      value: {metrics.get('test_chrf', 0):.2f}
""")
            if metrics.get("test_ssa_comet") is not None:
                eval_metric_entries.append(f"""    - name: SSA-COMET (test)
      type: comet
      value: {metrics.get('test_ssa_comet', 0) * 100:.2f}
""")

        if eval_metric_entries:
            card += f"""model-index:
- name: {repo_name}
  results:
  - task:
      type: translation
    metrics:
"""
            card += "".join(eval_metric_entries)

        card += f"""---

# {repo_name}

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/raw/main/model-on-hf-sm.svg)](https://huggingface.co/{self.organization}/{repo_name})

This is a **LoRA adapter** for the AfriScience-MT project, enabling efficient scientific machine translation for African languages.

## Adapter Description

| Property | Value |
|----------|-------|
| **Base Model** | [{base_model}](https://huggingface.co/{base_model}) |
"""
        if is_multilingual:
            card += """| **Languages** | English ↔ Amharic, Hausa, Luganda, Northern Sotho, Yoruba, isiZulu |
"""
        else:
            card += f"""| **Translation Direction** | {src_name} → {tgt_name} |
"""

        card += f"""| **LoRA Rank (r)** | {lora_rank} |
| **LoRA Alpha** | {lora_rank * 2} |
| **Training Method** | QLoRA (4-bit quantization) |
| **Domain** | Scientific/Academic texts |

### Why LoRA?

LoRA (Low-Rank Adaptation) enables efficient fine-tuning by training only a small number of additional parameters. This adapter adds only **~{lora_rank * 0.5:.1f}M parameters** to the base model while achieving strong translation performance.
"""

        # Evaluation Results
        if metrics:
            card += """
## Evaluation Results

Performance on the AfriScience-MT test set:

| Split | BLEU | chrF | SSA-COMET |
|-------|------|------|-----------|
"""
            val_bleu = metrics.get('val_bleu', '-')
            val_chrf = metrics.get('val_chrf', '-')
            val_comet = metrics.get('val_ssa_comet', '-')
            test_bleu = metrics.get('test_bleu', '-')
            test_chrf = metrics.get('test_chrf', '-')
            test_comet = metrics.get('test_ssa_comet', '-')

            # Format COMET as percentage (multiply by 100)
            def fmt_comet(v):
                return f'{v * 100:.2f}' if isinstance(v, float) else v

            if val_bleu != '-' or val_chrf != '-':
                card += f"| Validation | {f'{val_bleu:.2f}' if isinstance(val_bleu, float) else val_bleu} | {f'{val_chrf:.2f}' if isinstance(val_chrf, float) else val_chrf} | {fmt_comet(val_comet)} |\n"
            card += f"| **Test** | **{f'{test_bleu:.2f}' if isinstance(test_bleu, float) else test_bleu}** | **{f'{test_chrf:.2f}' if isinstance(test_chrf, float) else test_chrf}** | **{fmt_comet(test_comet)}** |\n"

            card += """
**Metrics explanation:**
- **BLEU**: Measures n-gram overlap with reference translations (0-100, higher is better)
- **chrF**: Character-level F-score, robust for morphologically rich languages (0-100, higher is better)
- **SSA-COMET**: Neural metric trained for Sub-Saharan African languages, shown as percentage (0-100, higher is better) ([McGill-NLP/ssa-comet-stl](https://huggingface.co/McGill-NLP/ssa-comet-stl))
"""

        # Usage section
        card += f"""
## Usage

### Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Configure 4-bit quantization (recommended for memory efficiency)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "{base_model}",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained("{base_model}")

# Load LoRA adapter
adapter_name = "{self.organization}/{repo_name}"
model = PeftModel.from_pretrained(base_model, adapter_name)
model.eval()

# Prepare translation prompt
source_text = "Climate change significantly impacts agricultural productivity in sub-Saharan Africa."
instruction = "Translate the following {src_name} scientific text to {tgt_name}."
"""

        if is_gemma:
            card += f"""
# Format for Gemma chat template
messages = [{{"role": "user", "content": f"{{instruction}}\\n\\n{{source_text}}"}}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
"""
        else:
            card += """
# Format prompt
prompt = f\"\"\"### Instruction:
{instruction}

### Input:
{source_text}

### Response:
\"\"\"
"""

        card += """
# Generate translation
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        num_beams=5,
        early_stopping=True,
        pad_token_id=tokenizer.pad_token_id,
    )

# Decode only the generated part
generated = outputs[0][inputs["input_ids"].shape[1]:]
translation = tokenizer.decode(generated, skip_special_tokens=True)
print(translation)
```

### Without Quantization (Full Precision)

```python
# For GPUs with sufficient memory (>24GB for larger models)
base_model = AutoModelForCausalLM.from_pretrained(
    \"""" + base_model + """\",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = PeftModel.from_pretrained(base_model, \"""" + f"{self.organization}/{repo_name}" + """\")
```
"""

        # Training configuration
        if training_config:
            card += """
## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
"""
            card += f"| LoRA Rank (r) | {lora_rank} |\n"
            card += f"| LoRA Alpha | {lora_rank * 2} |\n"
            card += "| LoRA Dropout | 0.05 |\n"
            card += "| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |\n"

            config_display = {
                "num_epochs": "Epochs",
                "batch_size": "Batch Size",
                "learning_rate": "Learning Rate",
                "max_length": "Max Sequence Length",
                "warmup_steps": "Warmup Steps",
                "weight_decay": "Weight Decay",
                "gradient_accumulation_steps": "Gradient Accumulation",
            }

            for key, display_name in config_display.items():
                if key in training_config:
                    value = training_config[key]
                    if isinstance(value, float) and value < 0.01:
                        card += f"| {display_name} | {value:.0e} |\n"
                    else:
                        card += f"| {display_name} | {value} |\n"

        card += """
### Hardware Requirements

| Configuration | VRAM Required |
|---------------|---------------|
| 4-bit (QLoRA) | ~8-12 GB |
| 8-bit | ~16-20 GB |
| Full precision | ~24-40 GB |

"""

        # Reproducibility section
        card += f"""
## Reproducibility

To reproduce this adapter:

```bash
# Clone the AfriScience-MT repository
git clone https://github.com/afriscience-mt/afriscience-mt.git
cd afriscience-mt

# Install dependencies
pip install -r requirements.txt

# Run LoRA training
python -m afriscience_mt.scripts.run_lora_training \\
    --data_dir ./data \\
    --source_lang {source_lang} \\
    --target_lang {target_lang} \\
    --model_name {base_model} \\
    --model_type {'gemma' if is_gemma else 'llama' if is_llama else 'causal'} \\
    --lora_rank {lora_rank} \\
    --output_dir ./output \\
    --num_epochs 3 \\
    --batch_size 4 \\
    --load_in_4bit
```

## Limitations

- **Domain Specificity**: Optimized for scientific/academic texts; may underperform on casual or colloquial language.
- **Language Direction**: Only supports {src_name} → {tgt_name} translation.
- **Base Model Required**: Must be used with the [{base_model}](https://huggingface.co/{base_model}) base model.
- **Context Length**: Maximum context is model-dependent; longer texts should be chunked.

## Citation

If you use this adapter, please cite the AfriScience-MT project:

```bibtex
@inproceedings{{afriscience-mt-2025,
  title={{AfriScience-MT: Machine Translation for African Scientific Literature}},
  author={{AfriScience-MT Team}},
  year={{2025}},
  url={{https://github.com/afriscience-mt/afriscience-mt}}
}}
```

## License

This adapter is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgments

- Base model: [{base_model}](https://huggingface.co/{base_model})
- LoRA implementation: [PEFT](https://github.com/huggingface/peft)
- Evaluation: [SSA-COMET](https://huggingface.co/McGill-NLP/ssa-comet-stl) for African language assessment
"""
        return card

    def batch_upload(
        self,
        models_dir: str,
        model_type: str = "seq2seq",
        delete_local: bool = True,
        repo_prefix: str = "afriscience-mt",
    ) -> Dict[str, str]:
        """
        Upload all models in a directory.

        Args:
            models_dir: Directory containing model subdirectories
            model_type: Type of models
            delete_local: Whether to delete local files after upload
            repo_prefix: Prefix for repository names

        Returns:
            Dictionary mapping model names to repository URLs
        """
        models_dir = Path(models_dir)
        results = {}

        for model_path in models_dir.iterdir():
            if model_path.is_dir() and (model_path / "config.json").exists():
                # Parse model info from directory name
                parts = model_path.name.split("-")
                if len(parts) >= 2:
                    source_lang = parts[0] if len(parts[0]) == 3 else "eng"
                    target_lang = parts[1] if len(parts) > 1 and len(parts[1]) == 3 else "hau"
                else:
                    source_lang, target_lang = "eng", "hau"

                repo_name = model_path.name
                url = self.upload_model(
                    model_path=str(model_path),
                    repo_name=repo_name,
                    model_type=model_type,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    delete_local=delete_local,
                    repo_prefix=repo_prefix,
                )

                if url:
                    results[repo_name] = url

        return results


def upload_after_training(
    model_path: str,
    model_name: str,
    model_type: str,
    source_lang: str,
    target_lang: str,
    base_model: str,
    metrics: Optional[Dict[str, float]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    organization: Optional[str] = None,
    delete_local: bool = True,
    lora_rank: int = 32,
    add_to_collection: bool = True,
) -> Optional[str]:
    """
    Convenience function to upload a model after training.

    Args:
        model_path: Path to trained model
        model_name: Name for the model
        model_type: Type of model (seq2seq, lora)
        source_lang: Source language
        target_lang: Target language
        base_model: Base model used
        metrics: Evaluation metrics
        training_config: Training configuration
        organization: HF organization (uses env var if not provided)
        delete_local: Whether to delete local files
        lora_rank: LoRA rank (for LoRA models)
        add_to_collection: Whether to add to HF collections

    Returns:
        Repository URL if successful
    """
    uploader = HFUploader(
        organization=organization,
        add_to_collection=add_to_collection,
    )

    if "lora" in model_type.lower() or "dora" in model_type.lower():
        return uploader.upload_lora_adapter(
            adapter_path=model_path,
            repo_name=model_name,
            base_model=base_model,
            source_lang=source_lang,
            target_lang=target_lang,
            lora_rank=lora_rank,
            metrics=metrics,
            training_config=training_config,
            delete_local=delete_local,
            # Pass DoRA-specific collection type so the model lands in the right collection
            collection_types=["dora", "all"] if "dora" in model_type.lower() else None,
        )
    else:
        return uploader.upload_model(
            model_path=model_path,
            repo_name=model_name,
            model_type=model_type,
            source_lang=source_lang,
            target_lang=target_lang,
            base_model=base_model,
            metrics=metrics,
            training_config=training_config,
            delete_local=delete_local,
        )


def setup_all_collections(organization: str, token: Optional[str] = None) -> Dict[str, str]:
    """
    Create all collections for the AfriScience-MT project.

    Args:
        organization: HF organization/username
        token: HF API token

    Returns:
        Dictionary mapping collection types to slugs
    """
    manager = HFCollectionManager(organization=organization, token=token)

    results = {}
    for ctype in COLLECTION_CONFIGS.keys():
        slug = manager.get_or_create_collection(ctype)
        if slug:
            results[ctype] = slug

    return results
