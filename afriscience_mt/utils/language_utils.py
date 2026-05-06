"""Language code mappings and utilities for African languages."""

from typing import Dict, Optional, List, Tuple

# ISO 639-3 codes used in the project
LANGUAGE_CODES = ["eng", "amh", "hau", "lug", "nso", "yor", "zul"]

# Full language names
LANGUAGE_NAMES: Dict[str, str] = {
    "eng": "English",
    "amh": "Amharic",
    "hau": "Hausa",
    "lug": "Luganda",
    "nso": "Northern Sotho",
    "yor": "Yoruba",
    "zul": "isiZulu",
}

# Language families
LANGUAGE_FAMILIES: Dict[str, str] = {
    "eng": "Indo-European",
    "amh": "Afro-Asiatic",
    "hau": "Niger-Congo",
    "lug": "Niger-Congo",
    "nso": "Niger-Congo",
    "yor": "Niger-Congo",
    "zul": "Niger-Congo",
}

# Geographic regions
LANGUAGE_REGIONS: Dict[str, str] = {
    "eng": "Global",
    "amh": "East Africa (Ethiopia)",
    "hau": "West Africa (Nigeria, Niger)",
    "lug": "East Africa (Uganda)",
    "nso": "Southern Africa (South Africa)",
    "yor": "West Africa (Nigeria, Benin)",
    "zul": "Southern Africa (South Africa)",
}

# NLLB language codes (BCP-47 format with script)
NLLB_CODES: Dict[str, str] = {
    "eng": "eng_Latn",
    "amh": "amh_Ethi",
    "hau": "hau_Latn",
    "lug": "lug_Latn",
    "nso": "nso_Latn",
    "yor": "yor_Latn",
    "zul": "zul_Latn",
}

# M2M100 language codes
M2M100_CODES: Dict[str, str] = {
    "eng": "en",
    "amh": "am",
    "hau": "ha",
    "lug": "lg",
    "nso": "ns",  # M2M100 uses the ISO 639-3 code directly for Northern Sotho
    "yor": "yo",
    "zul": "zu",
}

# Languages with significant diacritics/tone marks
DIACRITIC_LANGUAGES: Dict[str, Dict] = {
    "yor": {
        "has_diacritics": True,
        "diacritic_types": ["tone_marks", "underdots"],
        "special_chars": {"ẹ": "e", "ọ": "o", "ṣ": "s"},
        "description": "Yoruba uses tone marks (acute, grave, macron) and underdots",
    },
    "amh": {
        "has_diacritics": False,
        "script": "Ge'ez",
        "description": "Amharic uses Ge'ez script (abugida), not Latin with diacritics",
    },
    "hau": {
        "has_diacritics": True,
        "diacritic_types": ["hooked_letters"],
        "special_chars": {"ɓ": "b", "ɗ": "d", "ƙ": "k", "ƴ": "y"},
        "description": "Hausa uses hooked letters for implosives",
    },
    "nso": {
        "has_diacritics": True,
        "diacritic_types": ["caron"],
        "special_chars": {"š": "s"},
        "description": "Northern Sotho uses š (s-caron)",
    },
    "zul": {
        "has_diacritics": False,
        "description": "isiZulu uses standard Latin alphabet",
    },
    "lug": {
        "has_diacritics": False,
        "description": "Luganda uses standard Latin alphabet",
    },
}

# Speaker populations (approximate, from Ethnologue)
SPEAKER_POPULATIONS: Dict[str, int] = {
    "eng": 1_500_000_000,  # L1 + L2 speakers
    "amh": 57_000_000,
    "hau": 77_000_000,
    "lug": 7_000_000,
    "nso": 4_600_000,
    "yor": 47_000_000,
    "zul": 27_000_000,
}


def get_language_name(code: str) -> str:
    """Get full language name from code."""
    return LANGUAGE_NAMES.get(code, code)


def get_nllb_code(code: str) -> str:
    """Get NLLB-format language code."""
    return NLLB_CODES.get(code, code)


def get_m2m100_code(code: str) -> str:
    """Get M2M100-format language code."""
    return M2M100_CODES.get(code, code)


def get_language_pair(src: str, tgt: str) -> str:
    """Get standardized language pair string."""
    return f"{src}-{tgt}"


def parse_language_pair(pair: str) -> Tuple[str, str]:
    """Parse language pair string into source and target."""
    parts = pair.split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid language pair format: {pair}")
    return parts[0], parts[1]


def get_all_language_pairs(
    source: str = "eng",
    bidirectional: bool = True
) -> List[str]:
    """
    Get all language pairs involving a source language.

    Args:
        source: Source language code (default: "eng")
        bidirectional: If True, include both directions (eng-xxx and xxx-eng)

    Returns:
        List of language pair strings
    """
    targets = [lang for lang in LANGUAGE_CODES if lang != source]
    pairs = [f"{source}-{tgt}" for tgt in targets]

    if bidirectional:
        pairs.extend([f"{tgt}-{source}" for tgt in targets])

    return pairs


def get_african_to_african_pairs() -> List[str]:
    """Get all African language to African language pairs (n-to-n)."""
    african_langs = [lang for lang in LANGUAGE_CODES if lang != "eng"]
    pairs = []

    for src in african_langs:
        for tgt in african_langs:
            if src != tgt:
                pairs.append(f"{src}-{tgt}")

    return pairs


def has_diacritics(lang: str) -> bool:
    """Check if a language uses significant diacritics."""
    if lang in DIACRITIC_LANGUAGES:
        return DIACRITIC_LANGUAGES[lang].get("has_diacritics", False)
    return False


def get_special_chars(lang: str) -> Dict[str, str]:
    """Get special character mappings for denormalization."""
    if lang in DIACRITIC_LANGUAGES:
        return DIACRITIC_LANGUAGES[lang].get("special_chars", {})
    return {}


def validate_language_code(code: str) -> bool:
    """Validate that a language code is supported."""
    return code in LANGUAGE_CODES


def validate_language_pair(pair: str) -> bool:
    """Validate that a language pair is valid."""
    try:
        src, tgt = parse_language_pair(pair)
        return validate_language_code(src) and validate_language_code(tgt) and src != tgt
    except ValueError:
        return False


# =============================================================================
# Scientific Domain Utilities
# =============================================================================

# Canonical domain names (using underscores)
SCIENTIFIC_DOMAINS = [
    "agriculture",
    "biochemistry",
    "biology",
    "chemistry",
    "computer_science",
    "engineering",
    "geography",
    "health",
    "indigenous_knowledge",
    "sociology",
    "statistics",
]

# Domain name mappings (for normalization)
DOMAIN_ALIASES = {
    # Spaces to underscores
    "computer science": "computer_science",
    "indigenous knowledge": "indigenous_knowledge",
    # Common variations
    "cs": "computer_science",
    "compsci": "computer_science",
    "comp_sci": "computer_science",
    "bio": "biology",
    "biochem": "biochemistry",
    "chem": "chemistry",
    "eng": "engineering",
    "agri": "agriculture",
    "agric": "agriculture",
    "geo": "geography",
    "geog": "geography",
    "stats": "statistics",
    "stat": "statistics",
    "soc": "sociology",
    "socio": "sociology",
    "ik": "indigenous_knowledge",
    "indigenous": "indigenous_knowledge",
}


def normalize_domain(domain: str) -> str:
    """
    Normalize domain name to canonical form (lowercase, underscores).

    Args:
        domain: Domain name (may have spaces, different case, etc.)

    Returns:
        Normalized domain name

    Examples:
        >>> normalize_domain("Computer Science")
        'computer_science'
        >>> normalize_domain("computer-science")
        'computer_science'
        >>> normalize_domain("CS")
        'computer_science'
    """
    if not domain:
        return ""

    # Lowercase and normalize separators
    normalized = domain.lower().strip()
    normalized = normalized.replace("-", "_").replace(" ", "_")

    # Check aliases
    if normalized in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[normalized]

    # Check if it's already canonical
    if normalized in SCIENTIFIC_DOMAINS:
        return normalized

    # Return as-is if not recognized (preserves custom domains)
    return normalized


def validate_domain(domain: str) -> bool:
    """Check if domain is a recognized scientific domain."""
    normalized = normalize_domain(domain)
    return normalized in SCIENTIFIC_DOMAINS


def get_domain_display_name(domain: str) -> str:
    """
    Get display name for a domain (Title Case with spaces).

    Args:
        domain: Domain name (any format)

    Returns:
        Display name (e.g., "Computer Science")
    """
    normalized = normalize_domain(domain)
    return normalized.replace("_", " ").title()


def get_all_domains() -> List[str]:
    """Get list of all canonical domain names."""
    return SCIENTIFIC_DOMAINS.copy()
