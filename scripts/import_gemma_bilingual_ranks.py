"""Import Gemma-2-9B (non-instruct) bilingual LoRA rank-sweep predictions
from DCS-MT for the eng->xxx direction.

Source:
  /Users/MacbookPro/Documents/GitHub/DCS-MT/models/unsloth/lora/gemma-2-9b/
    rank-{8,16,32,64,128}/<eng-xxx>/lora_model/predictions_10e.txt

Each predictions_10e.txt contains one translation per line, in the same order
as data/processed_files/test/<tgt_lang>.tsv.

Destination:
  experiments/llm/open/finetune/gemma2_9b/r<N>/<pair>/
    predictions.json   {samples: [{src,ref,hyp}, ...]}
    results.json       canonical, is_ablation tagged True for r != 64

We use model_short=gemma2_9b (non-instruct base) to avoid mixing with the
existing gemma2_9b_it (instruct) bilingual data at experiments/llm/open/
finetune/gemma2_9b_it/. The MMT counterpart at experiments/llm/open/
multilingual/gemma2_9b/ uses the same non-instruct base, so the bilingual
imports here pair apples-to-apples with the existing MMT for the
multilingual-vs-bilingual comparison.

Run from repo root:
    python3 scripts/import_gemma_bilingual_ranks.py
    python3 -m afriscience_mt evaluate --mode rescore \
        --src experiments/llm/open/finetune/gemma2_9b
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_BASE = Path("/Users/MacbookPro/Documents/GitHub/DCS-MT/models/unsloth/lora/gemma-2-9b")
DST_BASE = REPO / "experiments/llm/open/finetune/gemma2_9b"
TEST_DIR = REPO / "data/processed_files/test"

RANKS = [8, 16, 32, 64, 128]
PAIRS_E2X = ["eng-amh", "eng-hau", "eng-lug", "eng-nso", "eng-yor", "eng-zul"]
MODEL_NAME = "google/gemma-2-9b"
MODEL_SHORT = "gemma2_9b"


def read_text_column(tsv_path: Path) -> list[str]:
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [r["text"] for r in reader]


def read_pred_lines(p: Path) -> list[str]:
    with open(p, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_pair(pair: str, rank: int, hyps: list[str]) -> None:
    src, tgt = pair.split("-")
    src_text = read_text_column(TEST_DIR / f"{src}.tsv")
    tgt_text = read_text_column(TEST_DIR / f"{tgt}.tsv")
    n = min(len(hyps), len(src_text), len(tgt_text))
    samples = [
        {"src": src_text[i], "ref": tgt_text[i], "hyp": hyps[i]}
        for i in range(n)
    ]
    if n != len(src_text):
        print(f"  warn {pair} r{rank}: {n} hyps vs {len(src_text)} test rows; truncated to {n}")

    out_dir = DST_BASE / f"r{rank}" / pair
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_id = f"lora_r{rank}_gemma2_9b_{src}_{tgt}"

    (out_dir / "predictions.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "source_lang": src, "target_lang": tgt,
        "samples": samples,
    }, indent=2, ensure_ascii=False))

    (out_dir / "results.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "experiment_type": "lora_finetune",
        "timestamp": datetime.utcnow().isoformat(),
        "source_lang": src, "target_lang": tgt, "lang_pair": pair,
        "split": "test", "dataset": "dcs",
        "model": {"name": MODEL_NAME, "type": "lora", "short_name": MODEL_SHORT,
                  "base_model": MODEL_NAME},
        "config": {
            "lora_rank": rank,
            "lora_alpha": rank,
            "num_epochs": 10,
            "training_setup": "bilingual",
        },
        "metrics": {"bleu": None, "chrf": None, "ssa_comet": None,
                    "num_samples": n},
        "metadata": {
            "source": "imported_from_dcs_mt",
            "original_path": (
                f"DCS-MT/models/unsloth/lora/gemma-2-9b/rank-{rank}/{pair}"
                "/lora_model/predictions_10e.txt"
            ),
            "is_ablation": rank != 64,
            "needs_eval": True,
            "comet_metric": "ssa-comet",
        },
    }, indent=2, ensure_ascii=False))


def main() -> int:
    if not SRC_BASE.is_dir():
        print(f"error: source dir not found: {SRC_BASE}", file=sys.stderr)
        return 2

    n_imported = 0
    n_missing = 0
    for rank in RANKS:
        for pair in PAIRS_E2X:
            base = SRC_BASE / f"rank-{rank}" / pair / "lora_model"
            # Prefer the 10-epoch variant; fall back to predictions.txt
            # (rank-128 has only the latter for Gemma).
            pred = base / "predictions_10e.txt"
            if not pred.is_file():
                pred = base / "predictions.txt"
            if not pred.is_file():
                print(f"  miss r{rank} {pair}: no predictions(_10e).txt under {base}")
                n_missing += 1
                continue
            hyps = read_pred_lines(pred)
            write_pair(pair, rank, hyps)
            n_imported += 1
            print(f"  ok   r{rank:>3} {pair} ({len(hyps)} lines, file={pred.name})")

    print(f"\nImported {n_imported}, missing {n_missing}")
    print("\nNext: rescore to populate chrF + SSA-COMET (needs GPU for COMET):")
    print("  python -m afriscience_mt evaluate --mode rescore \\")
    print("    --src experiments/llm/open/finetune/gemma2_9b")
    print("Then re-aggregate:")
    print("  python -m afriscience_mt aggregate --out experiments/summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
