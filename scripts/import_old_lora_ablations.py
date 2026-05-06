"""Import additional LoRA-rank runs for llama3_8b from DCS-MT-Updated.

Source: /ext_data/idris/DCS-MT-Updated/old_experiments/lora/
        Meta-Llama-3.1-8B-Instruct/rank-{8,16,32,128}/<pair>/lora_model/

Old format:
  predictions.txt       (one translation per line)
  predictions_10e.txt   (10-epoch variant; used for xx-eng pairs that only
                         have this file)
  results.json          (sacrebleu BLEU only; no chrF / no SSA-COMET)

We rebuild canonical predictions.json + results.json under
  experiments/llm/open/finetune/llama3_8b/r<N>/<pair>/
populating samples with src/ref/hyp from data/processed_files/test/<lang>.tsv
so that `afriscience-mt evaluate --mode rescore` can later compute
BLEU / chrF / SSA-COMET in our canonical setup.

All imported runs are flagged metadata.is_ablation = true (main focus is r4 / r64).
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_BASE = Path("/ext_data/idris/DCS-MT-Updated/old_experiments/lora/Meta-Llama-3.1-8B-Instruct")
DST_BASE = REPO / "experiments/llm/open/finetune/llama3_8b"
TEST_DIR = REPO / "data/processed_files/test"

RANKS = [8, 16, 32, 128]
PAIRS_3E = ["eng-amh", "eng-hau", "eng-lug", "eng-nso", "eng-yor", "eng-zul"]  # have predictions.txt
PAIRS_10E = ["amh-eng", "hau-eng", "lug-eng", "nso-eng", "yor-eng", "zul-eng"]  # have predictions_10e.txt only


def read_text_column(tsv_path: Path) -> list[str]:
    """Return the `text` column of a test tsv (skipping header)."""
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [r["text"] for r in reader]


def import_pair(rank: int, src: str, tgt: str, preds_file: str, num_epochs: int) -> tuple[Path, int] | None:
    preds_path = SRC_BASE / f"rank-{rank}" / f"{src}-{tgt}" / "lora_model" / preds_file
    if not preds_path.exists():
        return None

    hyps = preds_path.read_text().splitlines()
    # Cap to the test split length so the predictions index is meaningful when
    # rescore reads src/ref on the fly from data/processed_files/test/.
    ref_lines = read_text_column(TEST_DIR / f"{tgt}.tsv")
    n = min(len(hyps), len(ref_lines))
    samples = [{"hyp": hyps[i]} for i in range(n)]

    out_dir = DST_BASE / f"r{rank}" / f"{src}-{tgt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_doc = {
        "schema_version": "1.0",
        "experiment_id": f"lora_r{rank}_llama_{src}_{tgt}",
        "source_lang": src,
        "target_lang": tgt,
        "samples": samples,
    }
    (out_dir / "predictions.json").write_text(
        json.dumps(pred_doc, indent=2, ensure_ascii=False))

    result = {
        "schema_version": "1.0",
        "experiment_id": f"lora_r{rank}_llama_{src}_{tgt}",
        "experiment_type": "lora_finetune",
        "timestamp": datetime.utcnow().isoformat(),
        "model": {
            "name": "meta-llama/Llama-3.1-8B-Instruct",
            "type": "lora",
            "short_name": "llama3_8b",
        },
        "source_lang": src,
        "target_lang": tgt,
        "lang_pair": f"{src}-{tgt}",
        "split": "test",
        "dataset": "dcs",
        "config": {
            "lora_rank": rank,
            "lora_alpha": rank,
            "quantization": "4bit",
            "num_epochs": num_epochs,
            "base_model": "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
        },
        "metrics": {"num_samples": n},
        "metadata": {
            "source": "imported_from_old_experiments",
            "original_path": str(preds_path.parent.relative_to(Path("/ext_data/idris/DCS-MT-Updated"))),
            "is_ablation": True,
            "import_note": (
                "Imported from DCS-MT-Updated/old_experiments. Original results had "
                "only sacrebleu BLEU and possibly AfriCOMET; metrics will be recomputed "
                "with canonical BLEU / chrF / SSA-COMET via `evaluate --mode rescore`."
            ),
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))

    return out_dir, n


def main() -> int:
    n_imported = 0
    n_missing = 0
    for rank in RANKS:
        for pair in PAIRS_3E:
            src, tgt = pair.split("-")
            r = import_pair(rank, src, tgt, "predictions.txt", num_epochs=3)
            if r is None:
                print(f"  miss r{rank} {pair}")
                n_missing += 1
            else:
                d, n = r
                print(f"  r{rank} {pair} -> {d.relative_to(REPO)} ({n} samples)")
                n_imported += 1
        for pair in PAIRS_10E:
            src, tgt = pair.split("-")
            r = import_pair(rank, src, tgt, "predictions_10e.txt", num_epochs=10)
            if r is None:
                print(f"  miss r{rank} {pair}")
                n_missing += 1
            else:
                d, n = r
                print(f"  r{rank} {pair} -> {d.relative_to(REPO)} ({n} samples) [10e]")
                n_imported += 1
    print(f"\nimported: {n_imported}  missing: {n_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
