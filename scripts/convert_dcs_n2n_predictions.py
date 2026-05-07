"""Convert legacy ``dcs_n2n/<pair>/generated_predictions.txt`` files into the
canonical ``dcs/<pair>/predictions.json`` + ``results.json`` schema.

Source layout (one file per pair, plain text — one prediction per line):
    experiments/seq2seq/finetune/<model>/dcs_n2n/<src>-<tgt>/generated_predictions.txt

Target layout (canonical, position-aligned with data/processed_files/test/<src>.tsv):
    experiments/seq2seq/finetune/<model>/dcs/<src>-<tgt>/predictions.json
    experiments/seq2seq/finetune/<model>/dcs/<src>-<tgt>/results.json

Rules:
  * Skip eng↔X pairs entirely — those are already populated under ``dcs/``.
  * Pad with empty ``hyp`` if the legacy txt is short (the canonical test has
    970 rows; the legacy txt usually has 969 — root cause unknown, possibly
    a missing first/last sentence). The rescore CLI computes chrF/COMET by
    index against the canonical test; a missing hyp scores 0 for that row.
  * Refuse to overwrite an existing ``predictions.json`` unless ``--force``.
  * Compute chrF + BLEU at convert time (cheap); leave SSA-COMET to a later
    ``afriscience-mt evaluate --mode rescore`` pass.

Usage:
    python3 scripts/convert_dcs_n2n_predictions.py
    python3 scripts/convert_dcs_n2n_predictions.py --models nllb_200_distilled_600m
    python3 scripts/convert_dcs_n2n_predictions.py --force
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "seq2seq" / "finetune"
TEST_DIR = ROOT / "data" / "processed_files" / "test"

DEFAULT_MODELS = ["m2m100_418m", "nllb_200_distilled_600m"]
EXPECTED_N = 970          # test split row count
LANGS = {"amh", "hau", "lug", "nso", "yor", "zul"}


def load_test(lang: str) -> list[str]:
    """Return list of `text` strings for `lang.tsv` (970 rows, header skipped)."""
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


def load_legacy(path: Path) -> list[str]:
    """Read a generated_predictions.txt file. Each line = one prediction."""
    return path.read_text().splitlines()


def score_chrf(hyps: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score


def score_bleu(hyps: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def model_full_name(model_short: str) -> str:
    return {
        "m2m100_418m": "facebook/m2m100_418M",
        "m2m100_1_2b": "facebook/m2m100_1.2B",
        "nllb_200_distilled_600m": "facebook/nllb-200-distilled-600M",
        "nllb_200_1_3b": "facebook/nllb-200-1.3B",
    }.get(model_short, model_short)


def convert_pair(model_short: str, pair: str, *, force: bool) -> tuple[Path, dict] | None:
    src_lang, tgt_lang = pair.split("-")
    if src_lang == "eng" or tgt_lang == "eng":
        return None  # eng↔X already in canonical dcs/
    if src_lang not in LANGS or tgt_lang not in LANGS:
        return None

    legacy = EXP_DIR / model_short / "dcs_n2n" / pair / "generated_predictions.txt"
    target_dir = EXP_DIR / model_short / "dcs" / pair
    pred_path = target_dir / "predictions.json"
    res_path = target_dir / "results.json"

    if not legacy.exists():
        return None
    if pred_path.exists() and not force:
        return ("skip", {"pair": pair, "reason": "predictions.json exists"})

    hyps = load_legacy(legacy)
    n_legacy = len(hyps)
    if n_legacy < EXPECTED_N:
        hyps = hyps + [""] * (EXPECTED_N - n_legacy)
        pad = EXPECTED_N - n_legacy
    elif n_legacy > EXPECTED_N:
        hyps = hyps[:EXPECTED_N]
        pad = -(n_legacy - EXPECTED_N)
    else:
        pad = 0

    refs = load_test(tgt_lang)
    chrf = score_chrf(hyps, refs)
    bleu = score_bleu(hyps, refs)

    target_dir.mkdir(parents=True, exist_ok=True)

    predictions = {
        "schema_version": "1.0",
        "experiment_id": f"seq2seq_finetune__{model_short}__{pair}",
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "samples": [{"hyp": h} for h in hyps],
    }
    pred_path.write_text(json.dumps(predictions, indent=2, ensure_ascii=False))

    results = {
        "schema_version": "1.0",
        "experiment_id": f"seq2seq_finetune__{model_short}__{pair}",
        "experiment_type": "seq2seq_finetune",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": model_full_name(model_short),
            "type": "seq2seq",
            "short_name": model_short,
        },
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "lang_pair": pair,
        "split": "test",
        "dataset": "dcs",
        "config": {},
        "metrics": {
            "bleu": bleu,
            "chrf": chrf,
            # ssa_comet computed by `afriscience-mt evaluate --mode rescore`
            "num_samples": len(hyps),
        },
        "metadata": {
            "source": "imported_from_dcs_n2n",
            "original_path": str(legacy.relative_to(ROOT)),
            "is_ablation": False,
            "import_note": (
                f"converted from generated_predictions.txt; padded {pad} empty "
                f"row(s) to reach EXPECTED_N={EXPECTED_N}"
                if pad > 0 else
                (f"truncated {-pad} extra row(s) to EXPECTED_N={EXPECTED_N}"
                 if pad < 0 else
                 "1:1 conversion, no padding/truncation needed")
            ),
        },
    }
    res_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    return ("write", {"pair": pair, "chrf": chrf, "bleu": bleu, "pad": pad})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                    help="model_short names (default: m2m100_418m nllb_200_distilled_600m)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing predictions.json/results.json")
    args = ap.parse_args()

    summary = {"written": 0, "skipped": 0, "missing": 0}
    for model_short in args.models:
        legacy_root = EXP_DIR / model_short / "dcs_n2n"
        if not legacy_root.is_dir():
            print(f"[{model_short}] no dcs_n2n directory; skipping")
            continue
        print(f"[{model_short}] converting from {legacy_root}")
        for pair_dir in sorted(legacy_root.iterdir()):
            if not pair_dir.is_dir():
                continue
            res = convert_pair(model_short, pair_dir.name, force=args.force)
            if res is None:
                summary["missing"] += 1
                continue
            kind, info = res
            if kind == "write":
                summary["written"] += 1
                tag = (f" pad={info['pad']:+d}" if info["pad"] else "")
                print(f"  ✓ {info['pair']:10s}  chrF={info['chrf']:5.2f}  "
                      f"BLEU={info['bleu']:5.2f}{tag}")
            else:
                summary["skipped"] += 1
                print(f"  · {info['pair']:10s}  skipped: {info['reason']}")
    print()
    print(f"summary: written={summary['written']}  "
          f"skipped={summary['skipped']}  "
          f"missing/eng-pair={summary['missing']}")


if __name__ == "__main__":
    main()
