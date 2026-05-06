"""Glossary management for terminology-aware translation."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd


class Glossary:
    """
    Terminology glossary for a language pair.

    Manages domain-specific terminology mappings.
    """

    def __init__(
        self,
        source_lang: str = "eng",
        target_lang: str = None,
        data_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize glossary.

        Args:
            source_lang: Source language code
            target_lang: Target language code
            data_path: Path to glossary file or directory
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.terms: Dict[str, str] = {}
        self.reverse_terms: Dict[str, str] = {}  # For reverse lookup
        self.domains: Dict[str, Set[str]] = {}  # Domain -> terms

        if data_path:
            self.load(data_path)

    def load(self, path: Union[str, Path]) -> None:
        """
        Load glossary from file.

        Supports:
        - TSV: eng<tab>target_lang
        - JSON: {"eng_term": "target_term", ...}

        Args:
            path: Path to glossary file
        """
        path = Path(path)

        if path.is_dir():
            # Look for glossary file in directory
            if self.target_lang:
                tsv_path = path / "glossary" / f"{self.target_lang}.tsv"
                json_path = path / "glossary" / f"{self.target_lang}.json"
            else:
                raise ValueError("target_lang required when loading from directory")

            if tsv_path.exists():
                path = tsv_path
            elif json_path.exists():
                path = json_path
            else:
                raise FileNotFoundError(f"No glossary found for {self.target_lang}")

        if not path.exists():
            raise FileNotFoundError(f"Glossary file not found: {path}")

        if path.suffix == ".tsv":
            self._load_tsv(path)
        elif path.suffix == ".json":
            self._load_json(path)
        else:
            # Try TSV format
            self._load_tsv(path)

    def _load_tsv(self, path: Path) -> None:
        """Load from TSV file."""
        df = pd.read_csv(path, sep="\t")

        # Detect column names
        if "eng" in df.columns:
            src_col = "eng"
            tgt_col = self.target_lang if self.target_lang in df.columns else df.columns[1]
        else:
            src_col = df.columns[0]
            tgt_col = df.columns[1]

        for _, row in df.iterrows():
            src_term = str(row[src_col]).strip()
            tgt_term = str(row[tgt_col]).strip()

            if src_term and tgt_term and src_term != "nan" and tgt_term != "nan":
                self.terms[src_term.lower()] = tgt_term
                self.reverse_terms[tgt_term.lower()] = src_term

                # Track domain if available
                if "domain" in df.columns:
                    domain = row["domain"]
                    if domain not in self.domains:
                        self.domains[domain] = set()
                    self.domains[domain].add(src_term.lower())

    def _load_json(self, path: Path) -> None:
        """Load from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for src_term, tgt_term in data.items():
            if isinstance(tgt_term, str):
                self.terms[src_term.lower()] = tgt_term
                self.reverse_terms[tgt_term.lower()] = src_term
            elif isinstance(tgt_term, dict):
                # Extended format with metadata
                self.terms[src_term.lower()] = tgt_term.get("translation", "")
                if "domain" in tgt_term:
                    domain = tgt_term["domain"]
                    if domain not in self.domains:
                        self.domains[domain] = set()
                    self.domains[domain].add(src_term.lower())

    def lookup(self, term: str, reverse: bool = False) -> Optional[str]:
        """
        Look up a term translation.

        Args:
            term: Term to look up
            reverse: If True, look up target->source

        Returns:
            Translation or None if not found
        """
        term_lower = term.lower()

        if reverse:
            return self.reverse_terms.get(term_lower)
        return self.terms.get(term_lower)

    def get_terms_for_text(
        self,
        text: str,
        max_terms: Optional[int] = None,
        reverse: bool = False,
    ) -> List[Tuple[str, str]]:
        """
        Find glossary terms present in text.

        Args:
            text: Text to search
            max_terms: Maximum number of terms to return
            reverse: If True, search target language terms

        Returns:
            List of (source_term, target_term) tuples
        """
        text_lower = text.lower()
        terms_dict = self.reverse_terms if reverse else self.terms

        found_terms = []
        for src_term, tgt_term in terms_dict.items() if not reverse else self.reverse_terms.items():
            # Look for the source term in text
            if src_term in text_lower:
                if reverse:
                    found_terms.append((tgt_term, src_term))  # (src, tgt) order
                else:
                    found_terms.append((src_term, self.terms.get(src_term, "")))

        # Sort by length (longer terms first to prioritize specificity)
        found_terms.sort(key=lambda x: len(x[0]), reverse=True)

        if max_terms:
            found_terms = found_terms[:max_terms]

        return found_terms

    def format_for_prompt(
        self,
        terms: Optional[List[Tuple[str, str]]] = None,
        text: Optional[str] = None,
        max_terms: int = 20,
        format_string: str = "{src} → {tgt}",
        separator: str = "\n",
    ) -> str:
        """
        Format glossary terms for inclusion in a prompt.

        Args:
            terms: List of term tuples, or None to use all
            text: If provided, only include terms found in text
            max_terms: Maximum number of terms
            format_string: Format for each term
            separator: Separator between terms

        Returns:
            Formatted glossary string
        """
        if text:
            terms = self.get_terms_for_text(text, max_terms=max_terms)
        elif terms is None:
            terms = list(self.terms.items())[:max_terms]

        formatted = []
        for src, tgt in terms:
            formatted.append(format_string.format(src=src, tgt=tgt))

        return separator.join(formatted)

    def get_domain_terms(self, domain: str) -> List[Tuple[str, str]]:
        """Get all terms for a specific domain."""
        if domain not in self.domains:
            return []

        return [(term, self.terms[term]) for term in self.domains[domain]]

    def __len__(self) -> int:
        return len(self.terms)

    def __contains__(self, term: str) -> bool:
        return term.lower() in self.terms


def load_glossary(
    data_path: Union[str, Path],
    target_lang: str,
    source_lang: str = "eng",
) -> Glossary:
    """
    Load glossary for a language pair.

    Args:
        data_path: Path to data directory
        target_lang: Target language code
        source_lang: Source language code

    Returns:
        Glossary instance
    """
    return Glossary(
        source_lang=source_lang,
        target_lang=target_lang,
        data_path=data_path,
    )


def get_relevant_terms(
    glossary: Glossary,
    text: str,
    max_terms: int = 10,
    min_term_length: int = 3,
) -> List[Tuple[str, str]]:
    """
    Get glossary terms relevant to a text.

    Args:
        glossary: Glossary instance
        text: Source text
        max_terms: Maximum terms to return
        min_term_length: Minimum term length to consider

    Returns:
        List of (source, target) term tuples
    """
    terms = glossary.get_terms_for_text(text, max_terms=max_terms * 2)

    # Filter by length
    terms = [(s, t) for s, t in terms if len(s) >= min_term_length]

    return terms[:max_terms]
