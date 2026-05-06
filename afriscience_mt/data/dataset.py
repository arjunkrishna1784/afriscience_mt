"""Dataset classes for AfriScience-MT."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from torch.utils.data import Dataset

from ..utils.language_utils import (
    LANGUAGE_CODES,
    get_language_name,
    validate_language_pair,
    normalize_domain,
)


class AfriScienceDataset(Dataset):
    """
    Dataset for sentence-level translation tasks.

    Supports loading from various formats:
    - JSON (litgpt format): instruction, input, output
    - TSV: parallel text files
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        source_lang: str,
        target_lang: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        include_metadata: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to data directory
            source_lang: Source language code
            target_lang: Target language code
            split: Data split (train, val, test)
            max_samples: Maximum number of samples to load
            include_metadata: Whether to include paper/domain metadata
        """
        self.data_path = Path(data_path)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.split = split
        self.lang_pair = f"{source_lang}-{target_lang}"
        self.include_metadata = include_metadata

        # Validate language pair
        if not validate_language_pair(self.lang_pair):
            raise ValueError(f"Invalid language pair: {self.lang_pair}")

        # Load data
        self.samples = self._load_data()

        if max_samples and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from appropriate format."""
        samples = []

        # Try litgpt JSON format first
        json_path = self.data_path / "processed_files" / "litgpt" / self.lang_pair / f"{self.split}.json"
        if json_path.exists():
            samples = self._load_json(json_path)
        else:
            # Fall back to TSV format
            samples = self._load_tsv()

        return samples

    def _load_json(self, json_path: Path) -> List[Dict[str, Any]]:
        """Load from litgpt JSON format."""
        samples = []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            sample = {
                "source": item.get("input", ""),
                "target": item.get("output", ""),
                "instruction": item.get("instruction", ""),
            }

            # Add metadata if available
            if self.include_metadata:
                sample["paper"] = item.get("paper", "")
                # Normalize domain name (handles spaces, different cases, etc.)
                raw_domain = item.get("category", item.get("domain", ""))
                sample["domain"] = normalize_domain(raw_domain)
                sample["segment_id"] = item.get("segment", "")

            samples.append(sample)

        return samples

    def _load_tsv(self) -> List[Dict[str, Any]]:
        """Load from parallel TSV files."""
        samples = []

        # Load source and target TSV files
        src_path = self.data_path / "processed_files" / self.split / f"{self.source_lang}.tsv"
        tgt_path = self.data_path / "processed_files" / self.split / f"{self.target_lang}.tsv"

        if not src_path.exists() or not tgt_path.exists():
            raise FileNotFoundError(
                f"TSV files not found: {src_path} or {tgt_path}"
            )

        src_df = pd.read_csv(src_path, sep="\t")
        tgt_df = pd.read_csv(tgt_path, sep="\t")

        # Align by paper and segment
        for _, src_row in src_df.iterrows():
            # Find matching target row
            tgt_row = tgt_df[
                (tgt_df["paper"] == src_row["paper"]) &
                (tgt_df["segment"] == src_row["segment"])
            ]

            if len(tgt_row) > 0:
                tgt_row = tgt_row.iloc[0]
                sample = {
                    "source": src_row["text"],
                    "target": tgt_row["text"],
                }

                if self.include_metadata:
                    sample["paper"] = src_row["paper"]
                    # Normalize domain name
                    raw_domain = src_row.get("category", "")
                    sample["domain"] = normalize_domain(raw_domain)
                    sample["segment_id"] = src_row["segment"]

                samples.append(sample)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        return {
            "source": sample["source"],
            "target": sample["target"],
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "metadata": {
                k: v for k, v in sample.items()
                if k not in ["source", "target"]
            } if self.include_metadata else {},
        }

    def get_sources(self) -> List[str]:
        """Get all source texts."""
        return [s["source"] for s in self.samples]

    def get_targets(self) -> List[str]:
        """Get all target texts."""
        return [s["target"] for s in self.samples]

    def get_by_domain(self, domain: str) -> List[Dict[str, Any]]:
        """Get samples from a specific domain (handles different formats like 'Computer Science' or 'computer_science')."""
        normalized = normalize_domain(domain)
        return [s for s in self.samples if s.get("domain") == normalized]

    def get_domains(self) -> List[str]:
        """Get list of unique domains (normalized)."""
        return list(set(s.get("domain", "") for s in self.samples if s.get("domain")))


class AfriScienceDocumentDataset(Dataset):
    """
    Dataset for document-level translation tasks.

    Groups sentences by paper to form documents.  The litgpt JSON files lack
    paper/segment/domain metadata, so this class enriches each sentence by
    cross-referencing the raw CSV files (data/raw_files/{paper}_{domain}.csv).
    Three column-name variants exist across the 230 raw files:

      Format A  Paper#   Seg. #     Source    (papers ≈ 60–150)
      Format B  Paper #  Segment #  English   (papers ≈ 1–50)
      Format C  Paper #  Segment    English   (papers ≈ 151–230)
    """

    # Candidate column names for each semantic field
    _PAPER_COLS  = ("Paper#", "Paper #")
    _SEG_COLS    = ("Seg. #", "Segment #", "Segment")
    _SOURCE_COLS = ("Source", "English")

    def __init__(
        self,
        data_path: Union[str, Path],
        source_lang: str,
        target_lang: str,
        split: str = "test",
        max_sentences_per_doc: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.split = split
        self.lang_pair = f"{source_lang}-{target_lang}"
        self.max_sentences = max_sentences_per_doc

        # Build metadata lookup from raw CSVs: source_text -> (paper, segment, domain)
        self._meta: Dict[str, Tuple] = self._build_raw_metadata_lookup()

        # Load sentence-level data and enrich with paper/segment/domain
        sentence_dataset = AfriScienceDataset(
            data_path=data_path,
            source_lang=source_lang,
            target_lang=target_lang,
            split=split,
            include_metadata=True,
        )
        enriched = self._enrich_samples(sentence_dataset.samples)

        # Group by paper
        self.documents = self._group_by_paper(enriched)

    # ------------------------------------------------------------------
    # Raw-CSV metadata helpers
    # ------------------------------------------------------------------

    def _build_raw_metadata_lookup(self) -> Dict[str, Tuple]:
        """
        Scan data/raw_files/*.csv and return a dict mapping
        source_text (stripped) -> (paper_num: int, segment: int, domain: str).
        Handles all three known column-name variants silently.
        """
        import re as _re

        raw_dir = self.data_path / "raw_files"
        lookup: Dict[str, Tuple] = {}

        if not raw_dir.exists():
            return lookup

        for csv_file in sorted(raw_dir.glob("*.csv")):
            m = _re.match(r"^(\d+)_(.+)$", csv_file.stem)
            if not m:
                continue
            paper_num = int(m.group(1))
            domain = normalize_domain(m.group(2))

            try:
                df = pd.read_csv(csv_file)
                col_set = set(df.columns)

                src_col = next((c for c in self._SOURCE_COLS if c in col_set), None)
                seg_col = next((c for c in self._SEG_COLS    if c in col_set), None)

                if src_col is None or seg_col is None:
                    continue

                for _, row in df.iterrows():
                    src = str(row[src_col]).strip().rstrip(":")
                    try:
                        seg = int(float(row[seg_col]))
                    except (ValueError, TypeError):
                        seg = 0
                    if src and src.lower() != "nan":
                        lookup[src] = (paper_num, seg, domain)
            except Exception:
                pass  # skip malformed CSVs silently

        return lookup

    def _enrich_samples(self, samples: List[Dict]) -> List[Dict]:
        """
        Add paper / segment_id / domain to each sample by looking up the
        raw-CSV metadata table.

        The lookup is keyed on English text.  For eng→xxx the English text is
        in "source"; for xxx→eng it is in "target".  We try both so that
        reverse-direction pairs (amh→eng, hau→eng, …) are enriched correctly.
        Samples that cannot be matched keep whatever metadata the JSON already
        provided.
        """
        enriched = []
        for s in samples:
            s = dict(s)  # shallow copy
            # Try source first (eng→xxx), then target (xxx→eng)
            for field in ("source", "target"):
                candidate = s.get(field, "").strip().rstrip(":")
                if candidate in self._meta:
                    paper_num, seg, domain = self._meta[candidate]
                    s["paper"]      = str(paper_num)
                    s["segment_id"] = seg
                    s["domain"]     = domain
                    break
            enriched.append(s)
        return enriched

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _group_by_paper(self, samples: List[Dict]) -> List[Dict[str, Any]]:
        """Group sentences into documents by paper, ordered by segment id."""
        paper_groups: Dict[str, Dict] = {}

        for sample in samples:
            paper = sample.get("paper") or "unknown"
            if paper not in paper_groups:
                paper_groups[paper] = {
                    "paper":     paper,
                    "domain":    sample.get("domain", ""),
                    "sentences": [],
                }
            paper_groups[paper]["sentences"].append({
                "source":     sample["source"],
                "target":     sample["target"],
                "segment_id": sample.get("segment_id", 0),
            })

        documents = []
        for paper_data in paper_groups.values():
            sentences = paper_data["sentences"]

            # Sort by segment id so sentences appear in original paper order
            try:
                sentences.sort(key=lambda x: int(x.get("segment_id", 0)))
            except (ValueError, TypeError):
                pass

            if self.max_sentences:
                sentences = sentences[: self.max_sentences]

            documents.append({
                "paper":           paper_data["paper"],
                "domain":          paper_data["domain"],
                "source_document": " ".join(s["source"] for s in sentences),
                "target_document": " ".join(s["target"] for s in sentences),
                "sentences":       sentences,
                "num_sentences":   len(sentences),
            })

        return documents

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        doc = self.documents[idx]
        return {
            "source_document": doc["source_document"],
            "target_document": doc["target_document"],
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "metadata": {
                "paper": doc["paper"],
                "domain": doc["domain"],
                "num_sentences": doc["num_sentences"],
            },
            "sentences": doc["sentences"],
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Return dataset statistics useful for experiment reporting.

        Keys
        ----
        num_documents           : number of paper-level documents
        total_sentences         : total sentences across all documents (after
                                  max_sentences_per_doc truncation if set)
        avg_sentences_per_doc   : mean sentences per document (float)
        min_sentences_per_doc   : min sentences in any document
        max_sentences_per_doc   : max sentences in any document
        max_sentences_per_doc_setting : value of max_sentences_per_doc param
                                  (None = no per-document sentence truncation)
        num_unknown_paper       : sentences that could not be matched to a raw
                                  CSV and were grouped under "unknown"
        metadata_coverage       : fraction of sentences that were matched
                                  (0.0–1.0; 1.0 means all matched)
        domains                 : sorted list of unique domain strings
        """
        sent_counts = [doc["num_sentences"] for doc in self.documents]
        total = sum(sent_counts)

        unknown_doc = next(
            (d for d in self.documents if d["paper"] == "unknown"), None
        )
        n_unknown = unknown_doc["num_sentences"] if unknown_doc else 0
        n_known   = total - n_unknown
        coverage  = round(n_known / total, 4) if total > 0 else 0.0

        domains = sorted({d["domain"] for d in self.documents if d.get("domain")})

        return {
            "num_documents":                len(self.documents),
            "total_sentences":              total,
            "avg_sentences_per_doc":        round(total / len(self.documents), 2) if self.documents else 0.0,
            "min_sentences_per_doc":        min(sent_counts) if sent_counts else 0,
            "max_sentences_per_doc":        max(sent_counts) if sent_counts else 0,
            "max_sentences_per_doc_setting": self.max_sentences,
            "num_unknown_paper":            n_unknown,
            "metadata_coverage":            coverage,
            "domains":                      domains,
        }


def load_dataset_split(
    data_path: Union[str, Path],
    source_lang: str,
    target_lang: str,
    split: str,
    document_level: bool = False,
    **kwargs,
) -> Union[AfriScienceDataset, AfriScienceDocumentDataset]:
    """
    Load a dataset split.

    Args:
        data_path: Path to data directory
        source_lang: Source language code
        target_lang: Target language code
        split: Data split (train, val, test)
        document_level: Whether to load document-level data
        **kwargs: Additional arguments for dataset class

    Returns:
        Dataset instance
    """
    if document_level:
        return AfriScienceDocumentDataset(
            data_path=data_path,
            source_lang=source_lang,
            target_lang=target_lang,
            split=split,
            **kwargs,
        )
    else:
        return AfriScienceDataset(
            data_path=data_path,
            source_lang=source_lang,
            target_lang=target_lang,
            split=split,
            **kwargs,
        )


def load_all_splits(
    data_path: Union[str, Path],
    source_lang: str,
    target_lang: str,
    splits: List[str] = ["train", "val", "test"],
    **kwargs,
) -> Dict[str, AfriScienceDataset]:
    """
    Load all dataset splits.

    Args:
        data_path: Path to data directory
        source_lang: Source language code
        target_lang: Target language code
        splits: List of splits to load
        **kwargs: Additional arguments for dataset class

    Returns:
        Dictionary mapping split names to datasets
    """
    datasets = {}

    for split in splits:
        try:
            datasets[split] = load_dataset_split(
                data_path=data_path,
                source_lang=source_lang,
                target_lang=target_lang,
                split=split,
                **kwargs,
            )
        except FileNotFoundError:
            pass  # Skip missing splits

    return datasets
