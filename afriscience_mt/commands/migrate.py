"""Migrate legacy DCS-MT result files to the canonical schema.

The legacy layout has three top-level shapes; this tool collapses all of them
into one consistent on-disk layout under ``experiments/``:

    experiments/<canonical_etype>/<model_short>/<src-tgt>/results.json
                                                          predictions.json
    experiments/<canonical_etype>/<model_short>/aggregate.json   (optional)

Original files are preserved; the canonical ones are written alongside (with
the ``schema_version`` field) so the harness can verify and roll back.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from afriscience_mt.core.schema import (
    EXPERIMENT_TYPES,
    ExperimentResult,
    Metrics,
    ModelInfo,
    PredictionFile,
    canonicalize,
    save_predictions,
    save_result,
)
from afriscience_mt.core.registry import short_name


# Top-level legacy directories → (canonical experiment_type, prompt_strategy)
# strategy is None when there is only one variant for that type
LEGACY_DIR_TO_TYPE: dict[str, tuple[str, str | None]] = {
    "zero_shot": ("zero_shot", None),
    "seq2seq": ("seq2seq_finetune", None),
    "lora": ("lora_finetune", None),
    "llm/zero_shot": ("llm_zero_shot", None),
    "llm/icl_0shot": ("llm_icl", "icl_0shot"),
    "llm/icl_5shot": ("llm_icl", "icl_5shot"),
    "llm/icl_5shot_semantic": ("llm_icl", "icl_5shot_semantic"),
    "llm/doc_zero_shot": ("llm_doc", "doc_0shot"),
    "llm/doc_1shot": ("llm_doc", "doc_1shot"),
    "llm/docs_0shot": ("llm_doc", "doc_0shot_legacy"),
    "llm/docs_1shot": ("llm_doc", "doc_1shot_legacy"),
    "icl": ("llm_icl", "icl_legacy"),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="afriscience-mt migrate")
    p.add_argument("--src", default="experiments",
                   help="root of legacy results (default: experiments/)")
    p.add_argument("--dst", default="experiments_canonical",
                   help="output root for canonical files")
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be migrated without writing")
    p.add_argument("--copy-predictions", action="store_true", default=True,
                   help="copy predictions.json alongside results")
    p.add_argument("--in-place", action="store_true",
                   help="write canonical files into the same dirs (overwrites results.json after backing up to results.original.json)")
    args = p.parse_args(argv)

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve() if not args.in_place else src_root

    n_seq, n_zs, n_skip = 0, 0, 0
    for legacy_dir, (etype, strategy) in LEGACY_DIR_TO_TYPE.items():
        for path in (src_root / legacy_dir).rglob("results.json"):
            try:
                count = _migrate_file(path, src_root, dst_root, etype,
                                      strategy=strategy,
                                      dry_run=args.dry_run,
                                      in_place=args.in_place,
                                      copy_predictions=args.copy_predictions)
                n_seq += count
            except Exception as e:
                print(f"  SKIP {path}: {e}")
                n_skip += 1

        # zero_shot has aggregated per-model results to fan out, but
        # only when per-pair files don't already exist (those are richer)
        if etype == "zero_shot":
            for path in (src_root / legacy_dir).glob("*/results.json"):
                model_dir = path.parent
                has_per_pair = any(p.is_dir() and (p / "results.json").exists()
                                   for p in model_dir.iterdir() if p.is_dir())
                if has_per_pair:
                    continue
                fanned = _migrate_zero_shot_aggregate(
                    path, src_root, dst_root,
                    dry_run=args.dry_run, in_place=args.in_place,
                )
                n_zs += fanned

    print(f"\n{'DRY RUN: ' if args.dry_run else ''}migrated "
          f"{n_seq} per-pair results, fanned {n_zs} zero-shot pair results, "
          f"skipped {n_skip}.")
    return 0


def _migrate_file(path: Path, src_root: Path, dst_root: Path,
                  etype: str, *, strategy: str | None,
                  dry_run: bool, in_place: bool,
                  copy_predictions: bool) -> int:
    """Migrate one per-pair results.json. Returns 1 if a result was written."""
    legacy = json.load(open(path))

    # Per-pair records under llm/<phase>/<model>/<src-tgt>/ have source/target
    # at top level; the per-model aggregate_results.json doesn't, so skip those.
    if not legacy.get("source_lang"):
        return 0

    canonical = canonicalize(legacy, experiment_type=etype,
                             dataset="dcs",
                             original_path=str(path.relative_to(src_root)))
    if strategy:
        canonical.metadata["prompt_strategy"] = strategy

    # Override short_name with the legacy directory name for LLM-type runs:
    # the on-disk dir name disambiguates variants (gemini_15_flash vs
    # gemini_15_flash_temp_0.6) better than the model_name field, which is
    # often shared between sibling directories.
    if etype.startswith("llm_") or etype == "lora_finetune":
        legacy_model_dir = _legacy_model_dir(path, etype)
        if legacy_model_dir:
            from afriscience_mt.core.registry import short_name as _short
            canonical.model.short_name = _short(legacy_model_dir)

    out_dir = _output_dir(dst_root, canonical, strategy=strategy) if not in_place else path.parent
    out_path = out_dir / "results.json"

    if dry_run:
        print(f"  would write {out_path}")
        return 1

    if in_place and out_path.exists():
        backup = out_path.with_name("results.legacy.json")
        if not backup.exists():
            shutil.copy2(out_path, backup)

    save_result(canonical, out_path)

    if copy_predictions:
        _maybe_copy_predictions(path.parent, out_dir, canonical)

    return 1


def _migrate_zero_shot_aggregate(path: Path, src_root: Path, dst_root: Path,
                                 *, dry_run: bool, in_place: bool) -> int:
    """zero_shot/<model>/results.json has `per_pair_results`. Fan out to per-pair."""
    legacy = json.load(open(path))
    pair_results = legacy.get("per_pair_results")
    if not pair_results:
        return 0

    model_name = legacy.get("model", "unknown")
    model_type = "seq2seq"   # zero_shot in this project is always NLLB/M2M100
    timestamp = legacy.get("timestamp", datetime.utcnow().isoformat())
    base_metadata = {
        "source": "migrated",
        "comet_metric": "ssa-comet",
        "original_path": str(path.relative_to(src_root)),
        "directions": legacy.get("directions"),
    }

    n = 0
    for pair, body in pair_results.items():
        src, tgt = pair.split("-", 1)
        m = body.get("metrics", {})
        canonical = ExperimentResult(
            experiment_id=f"zero_shot__{short_name(model_name)}__{pair}",
            experiment_type="zero_shot",
            timestamp=timestamp,
            model=ModelInfo(name=model_name, type=model_type,
                            short_name=short_name(model_name)),
            source_lang=src,
            target_lang=tgt,
            split=legacy.get("split", "test"),
            dataset="dcs",
            config={"directions": legacy.get("directions")},
            metrics=Metrics(
                bleu=m.get("bleu"), chrf=m.get("chrf"),
                ssa_comet=m.get("ssa_comet"),
                num_samples=body.get("num_samples"),
            ),
            metadata=dict(base_metadata),
        )
        out_dir = _output_dir(dst_root, canonical)
        if dry_run:
            print(f"  would write {out_dir / 'results.json'}")
        else:
            save_result(canonical, out_dir / "results.json")
        n += 1
    return n


def _legacy_model_dir(path: Path, etype: str) -> str | None:
    """Return the legacy model dir name (e.g. 'gemini_15_flash_temp_0.6').

    Layout assumed:
      lora_finetune: experiments/lora/<MODEL>/r<N>/<src-tgt>/results.json
      llm_*:         experiments/llm/<phase>/<MODEL>/<src-tgt>/results.json
    """
    parts = path.parts
    pair = path.parent.name  # src-tgt
    if "-" not in pair:
        return None
    if etype == "lora_finetune":
        # parents: <pair>/r<N>/<MODEL>/lora/...
        try:
            return path.parent.parent.parent.name
        except IndexError:
            return None
    return path.parent.parent.name


def _output_dir(dst_root: Path, r: ExperimentResult, *,
                strategy: str | None = None) -> Path:
    short = r.model.short_name or short_name(r.model.name)
    parts: list = [dst_root, r.experiment_type, short]
    if r.dataset and r.dataset != "dcs":
        parts.append(r.dataset)
    # LoRA: preserve rank so r4/r8/r16/... don't overwrite
    if r.experiment_type == "lora_finetune":
        rank = r.config.get("lora_rank")
        if rank is not None:
            parts.append(f"r{rank}")
    # LLM doc/icl: preserve prompt strategy so doc_0shot/doc_1shot/icl_5shot/etc don't collide
    if strategy:
        parts.append(strategy)
    parts.append(r.lang_pair)
    return Path(*parts)


def _maybe_copy_predictions(src_dir: Path, out_dir: Path,
                            canonical: ExperimentResult) -> None:
    """Convert legacy predictions to canonical PredictionFile if present."""
    legacy = src_dir / "predictions.json"
    if not legacy.exists():
        return
    try:
        raw = json.load(open(legacy))
    except json.JSONDecodeError:
        return

    samples: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                samples.append({"hyp": item})
            elif isinstance(item, dict):
                samples.append({
                    "src": item.get("src") or item.get("source"),
                    "ref": item.get("ref") or item.get("reference"),
                    "hyp": item.get("hyp") or item.get("hypothesis") or item.get("translation"),
                    "doc_id": item.get("doc_id"),
                    "domain": item.get("domain"),
                })
    elif isinstance(raw, dict) and "samples" in raw:
        samples = raw["samples"]
    else:
        return

    pf = PredictionFile(
        experiment_id=canonical.experiment_id,
        source_lang=canonical.source_lang,
        target_lang=canonical.target_lang,
        samples=samples,
    )
    save_predictions(pf, out_dir / "predictions.json")


if __name__ == "__main__":
    raise SystemExit(main())
