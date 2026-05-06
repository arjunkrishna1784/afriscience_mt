"""Centralized path configuration for the framework.

Override defaults via environment variables (see Paths) or by passing values
explicitly when constructing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    repo_root: Path = _REPO_ROOT
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "DCSMT_DATA_DIR", _REPO_ROOT / "data")))
    experiments_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "DCSMT_EXPERIMENTS_DIR", _REPO_ROOT / "experiments")))
    hf_cache_dir: Path = field(default_factory=lambda: Path(os.environ.get(
        "DCSMT_HF_CACHE", _REPO_ROOT / "hf_cache")))
    configs_dir: Path = field(default_factory=lambda: _REPO_ROOT / "configs")

    def __post_init__(self) -> None:
        # Resolve symlinks once so downstream code sees absolute paths
        self.data_dir = Path(self.data_dir).resolve()
        self.experiments_dir = Path(self.experiments_dir).resolve()
        self.hf_cache_dir = Path(self.hf_cache_dir).resolve()
        self.configs_dir = Path(self.configs_dir).resolve()

    def experiment_dir(self, etype: str, model_short: str, lang_pair: str,
                       *, dataset: str | None = None) -> Path:
        """Where one experiment instance's results.json lives."""
        parts = [self.experiments_dir, etype, model_short]
        if dataset and dataset != "dcs":
            parts.append(dataset)
        parts.append(lang_pair)
        return Path(*parts)
