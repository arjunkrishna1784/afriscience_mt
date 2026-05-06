"""Single source of truth for model display names, language sets, domains."""

from __future__ import annotations

# 6 African languages in DCS
SUPPORTED_LANGS: tuple[str, ...] = ("amh", "hau", "lug", "nso", "yor", "zul")

# 4 languages overlapping with MAFAND (no amh, no nso)
MAFAND_LANGS: tuple[str, ...] = ("hau", "lug", "yor", "zul")

# Scientific domains in DCS
DOMAINS: tuple[str, ...] = (
    "agriculture", "biochemistry", "biology", "chemistry",
    "computer_science", "engineering", "geography", "health",
    "indigenous_knowledge", "sociology", "statistics",
)

# model id (HF) -> human-friendly display
MODEL_DISPLAY: dict[str, str] = {
    "facebook/nllb-200-600M": "NLLB-600M",
    "facebook/nllb-200-1.3B": "NLLB-1.3B",
    "facebook/m2m100_418M": "M2M100-418M",
    "facebook/m2m100_1.2B": "M2M100-1.2B",
    "google/gemma-2-9b-it": "Gemma-2-9B-IT",
    "google/gemma-3-4b-it": "Gemma-3-4B",
    "google/gemma-3-12b-it": "Gemma-3-12B",
    "google/translategemma-12b": "TranslateGemma-12B",
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama-3-8B",
    "meta-llama/Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "McGill-NLP/AfriqueLlama-8B": "AfriqueLlama-8B",
    "McGill-NLP/AfriqueQwen-8B": "AfriqueQwen-8B",
    "CohereForAI/aya-expanse-8b": "Aya-Expanse-8B",
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-5": "GPT-5",
    "google/gemini-1.5-flash": "Gemini-1.5-Flash",
}

MODEL_TYPE: dict[str, str] = {
    "facebook/nllb-200-600M": "seq2seq",
    "facebook/nllb-200-1.3B": "seq2seq",
    "facebook/m2m100_418M": "seq2seq",
    "facebook/m2m100_1.2B": "seq2seq",
}


def short_name(model_name: str) -> str:
    """Stable directory-safe short id for a model."""
    base = model_name.split("/")[-1]
    return (base.lower()
            .replace("-it", "_it")
            .replace(".", "_")
            .replace("-", "_"))
