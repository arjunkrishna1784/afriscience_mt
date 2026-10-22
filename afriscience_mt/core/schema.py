"""Canonical result schema for all DCS-MT experiments.

One shape across zero-shot, seq2seq, lora, and LLM (icl/document) runs so that
aggregation, table generation, and plotting never need to special-case file
layouts.

Layout on disk (per experiment instance):
    experiments/<experiment_type>/<model_short>/<src-tgt>/
        results.json
        predictions.json   # optional; format below
        training.log       # optional

For aggregated experiments (zero-shot evaluator that scores many pairs in one
run), per-pair files are still written; an extra per-model `aggregate.json`
summarizes averages.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


SCHEMA_VERSION = "1.0"

EXPERIMENT_TYPES = {
    "zero_shot",          # no training, run inference on a base seq2seq model
    "seq2seq_finetune",   # full fine-tuning of NLLB / M2M100
    "lora_finetune",      # LoRA fine-tuning of an LLM
    "llm_zero_shot",      # decoder-only LLM, sentence-level zero-shot prompt
    "llm_icl",            # LLM with k-shot in-context examples
    "llm_doc",            # LLM at document level (zero or k-shot)
    "llm_eval",           # LLM-as-judge metric run
}


@dataclass
class Metrics:
    bleu: Optional[float] = None
    chrf: Optional[float] = None
    ssa_comet: Optional[float] = None
    num_samples: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "extra" and v is not None}
        d.update(self.extra)
        return d


@dataclass
class ModelInfo:
    name: str                       # full hub id, e.g. facebook/nllb-200-1.3B
    type: str                       # seq2seq | llm | lora
    short_name: Optional[str] = None  # e.g. nllb_1.3b
    checkpoint: Optional[str] = None  # local path or HF revision
    base_model: Optional[str] = None  # for lora: the underlying llm

    def asdict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ExperimentResult:
    """Canonical result file. One per (experiment_type, model, lang_pair, dataset)."""

    experiment_id: str
    experiment_type: str            # one of EXPERIMENT_TYPES
    timestamp: str
    model: ModelInfo
    source_lang: str
    target_lang: str
    split: str                      # test | dev | train
    dataset: str                    # dcs | mafand | dcs_mafand | other
    config: dict[str, Any]
    metrics: Metrics
    val_metrics: Optional[Metrics] = None
    train_metrics: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    environment: Optional[dict[str, Any]] = None
    git: Optional[dict[str, Any]] = None
    schema_version: str = SCHEMA_VERSION

    @property
    def lang_pair(self) -> str:
        return f"{self.source_lang}-{self.target_lang}"

    def asdict(self) -> dict[str, Any]:
        out = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "timestamp": self.timestamp,
            "model": self.model.asdict(),
            "source_lang": self.source_lang,
            "target_lang": self.target_lang,
            "lang_pair": self.lang_pair,
            "split": self.split,
            "dataset": self.dataset,
            "config": self.config,
            "metrics": self.metrics.asdict(),
            "metadata": self.metadata,
        }
        if self.val_metrics is not None:
            out["val_metrics"] = self.val_metrics.asdict()
        if self.train_metrics is not None:
            out["train_metrics"] = self.train_metrics
        if self.environment is not None:
            out["environment"] = self.environment
        if self.git is not None:
            out["git"] = self.git
        return out


@dataclass
class PredictionFile:
    """Canonical predictions list for one experiment instance.

    Each sample carries source, reference, hypothesis. ``doc_id`` is set for
    document-level runs so we can reconstruct doc-level scoring later.
    """

    experiment_id: str
    source_lang: str
    target_lang: str
    samples: list[dict[str, Any]]   # [{"src", "ref", "hyp", "doc_id"?, "domain"?}]
    schema_version: str = SCHEMA_VERSION
    reference_fingerprint: Optional[str] = None

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


# ---------- IO ----------

def save_result(result: ExperimentResult, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as f:
            temporary = Path(f.name)
            json.dump(result.asdict(), f, indent=2, ensure_ascii=False, default=_jsonable)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_result(path: str | Path) -> ExperimentResult:
    with open(path) as f:
        d = json.load(f)
    return _result_from_dict(d)


def save_predictions(preds: PredictionFile, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(preds.asdict(), f, indent=2, ensure_ascii=False, default=_jsonable)


def load_predictions(path: str | Path) -> PredictionFile:
    with open(path) as f:
        d = json.load(f)
    return PredictionFile(**d)


def _result_from_dict(d: dict[str, Any]) -> ExperimentResult:
    metrics_d = d.get("metrics", {})
    extra = {k: v for k, v in metrics_d.items()
             if k not in {"bleu", "chrf", "ssa_comet", "num_samples"}}
    metrics = Metrics(
        bleu=metrics_d.get("bleu"),
        chrf=metrics_d.get("chrf"),
        ssa_comet=metrics_d.get("ssa_comet"),
        num_samples=metrics_d.get("num_samples"),
        extra=extra,
    )
    val_metrics = None
    if "val_metrics" in d:
        vm = d["val_metrics"]
        val_metrics = Metrics(
            bleu=vm.get("bleu"), chrf=vm.get("chrf"),
            ssa_comet=vm.get("ssa_comet"), num_samples=vm.get("num_samples"),
            extra={k: v for k, v in vm.items()
                   if k not in {"bleu", "chrf", "ssa_comet", "num_samples"}},
        )
    return ExperimentResult(
        experiment_id=d["experiment_id"],
        experiment_type=d["experiment_type"],
        timestamp=d["timestamp"],
        model=ModelInfo(**d["model"]),
        source_lang=d["source_lang"],
        target_lang=d["target_lang"],
        split=d.get("split", "test"),
        dataset=d.get("dataset", "dcs"),
        config=d.get("config", {}),
        metrics=metrics,
        val_metrics=val_metrics,
        train_metrics=d.get("train_metrics"),
        metadata=d.get("metadata", {}),
        environment=d.get("environment"),
        git=d.get("git"),
        schema_version=d.get("schema_version", SCHEMA_VERSION),
    )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (datetime, Path)):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


# ---------- canonical migration (legacy → canonical) ----------

def canonicalize(legacy: dict[str, Any], *,
                 experiment_type: Optional[str] = None,
                 dataset: str = "dcs",
                 source: str = "migrated",
                 original_path: Optional[str] = None) -> ExperimentResult:
    """Convert a legacy results.json dict into the canonical ExperimentResult.

    Handles three legacy shapes seen in DCS-MT-Updated:
    1. Per-pair structured (seq2seq / lora / llm/*): top-level keys
       experiment_name, experiment_type, source_lang, target_lang, model_name,
       model_type, metrics{test_bleu, test_chrf, test_ssa_comet, ...}.
    2. Aggregated zero-shot (zero_shot/<model>/results.json): no per-pair
       result; caller should iterate `per_pair_results` and call this once per
       pair before passing.
    3. LLM aggregate (llm/<phase>/<model>/aggregate_results.json): summary
       only, not a per-pair result. Migration tool skips these.
    """
    etype = experiment_type or _infer_etype(legacy)
    metrics_d = legacy.get("metrics", {}) or {}

    test_metrics = _extract_metrics(metrics_d, prefix="test_")
    val_metrics = _extract_metrics(metrics_d, prefix="val_")

    train_metrics = None
    train_keys = {k: v for k, v in metrics_d.items()
                  if k.startswith("train_") or k in {"hf_url", "lora_rank"}}
    if train_keys:
        train_metrics = train_keys

    model_name = (legacy.get("model_name") or legacy.get("model")
                  or legacy.get("base_model") or "unknown")
    raw_type = legacy.get("model_type") or ""
    model_type = _normalize_model_type(raw_type, model_name, etype)
    base_model = legacy.get("base_model") if etype == "lora_finetune" else None

    src = legacy.get("source_lang", "")
    tgt = legacy.get("target_lang", "")
    metadata = dict(legacy.get("metadata") or {})
    metadata.setdefault("source", source)
    if original_path:
        metadata.setdefault("original_path", original_path)
    if "comet_metric" not in metadata:
        metadata["comet_metric"] = "ssa-comet"

    config = dict(legacy.get("config") or {})

    eid = legacy.get("experiment_name") or f"{etype}__{model_name.split('/')[-1]}__{src}-{tgt}"

    from afriscience_mt.core.registry import short_name as _short
    return ExperimentResult(
        experiment_id=eid,
        experiment_type=etype,
        timestamp=legacy.get("timestamp", datetime.utcnow().isoformat()),
        model=ModelInfo(name=model_name, type=model_type,
                        short_name=_short(model_name) if model_name != "unknown" else None,
                        base_model=base_model),
        source_lang=src,
        target_lang=tgt,
        split="test",
        dataset=dataset,
        config=config,
        metrics=test_metrics,
        val_metrics=val_metrics if (val_metrics.bleu or val_metrics.chrf
                                    or val_metrics.ssa_comet) else None,
        train_metrics=train_metrics,
        metadata=metadata,
        environment=legacy.get("environment"),
        git=legacy.get("git_info") or legacy.get("git"),
    )


def _extract_metrics(metrics_d: dict[str, Any], *, prefix: str) -> Metrics:
    """Pull bleu/chrf/ssa_comet (and extras) for a given prefix (`test_`/`val_`)."""
    keys = ("bleu", "chrf", "ssa_comet", "num_samples")
    primary = {k: metrics_d.get(f"{prefix}{k}") for k in keys}
    extra = {k.removeprefix(prefix): v for k, v in metrics_d.items()
             if k.startswith(prefix) and k.removeprefix(prefix) not in keys}
    return Metrics(
        bleu=primary["bleu"],
        chrf=primary["chrf"],
        ssa_comet=primary["ssa_comet"],
        num_samples=primary["num_samples"],
        extra=extra,
    )


def _infer_etype(legacy: dict[str, Any]) -> str:
    raw = legacy.get("experiment_type", "")
    mapping = {
        "lora_finetune": "lora_finetune",
        "seq2seq_finetune": "seq2seq_finetune",
        "zero_shot": "llm_zero_shot",   # llm/zero_shot uses bare 'zero_shot'
        "doc_zero_shot": "llm_doc",
        "doc_1shot": "llm_doc",
        "icl_5shot": "llm_icl",
        "icl_0shot": "llm_icl",
        "icl_5shot_semantic": "llm_icl",
    }
    if raw in mapping:
        return mapping[raw]
    if raw in EXPERIMENT_TYPES:
        return raw
    return raw or "unknown"


def _guess_model_type(model_name: str) -> str:
    n = model_name.lower()
    if "nllb" in n or "m2m100" in n:
        return "seq2seq"
    if "lora" in n:
        return "lora"
    return "llm"


def _normalize_model_type(raw_type: str, model_name: str, etype: str) -> str:
    """Map legacy free-form `model_type` to canonical {seq2seq, llm, lora}."""
    if etype == "lora_finetune":
        return "lora"
    if etype in {"seq2seq_finetune", "zero_shot"}:
        return "seq2seq"
    seq2seq_aliases = {"nllb", "m2m100", "seq2seq"}
    if raw_type.lower() in seq2seq_aliases:
        return "seq2seq"
    if raw_type.lower() in {"lora"}:
        return "lora"
    if etype.startswith("llm_"):
        return "llm"
    return _guess_model_type(model_name)
