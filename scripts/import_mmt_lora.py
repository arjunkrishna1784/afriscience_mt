"""Import multilingual LoRA (mmt) runs for Meta-Llama-3.1-8B-Instruct.

Source:
  /ext_data/idris/DCS-MT-Old/models/unsloth/lora/Meta-Llama-3.1-8B-Instruct/
    rank-{8,16,32,64,128}/mmt/lora_model/
      predictions_eng-{amh,hau,lug,nso,yor,zul}.txt   one translation per line
      results_eng-{src}.json                           sacrebleu BLEU only

Single LoRA adapter trained jointly on all eng->african pairs (mmt =
multilingual MT), then evaluated per-pair. Only eng-{tgt} directions exist;
no {src}-eng for this setup.

Destination:
  experiments/llm/open/multilingual/llama3_8b/r<N>/<pair>/
    predictions.json   {samples: [{hyp: "..."}]}      # hyp-only
    results.json       canonical, is_ablation=true

Run rescore after to populate metrics:
  afriscience-mt evaluate --mode rescore \\
      --src experiments/llm/open/multilingual/llama3_8b --metric bleu
  (then chrf, ssa_comet)
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_BASE = Path("/ext_data/idris/DCS-MT-Old/models/unsloth/lora/Meta-Llama-3.1-8B-Instruct")
DST_BASE = REPO / "experiments/llm/open/multilingual/llama3_8b"
TEST_DIR = REPO / "data/processed_files/test"

RANKS = [8, 16, 32, 64, 128]
PAIRS = ["eng-amh", "eng-hau", "eng-lug", "eng-nso", "eng-yor", "eng-zul"]


def cap_to_test_split(hyps: list[str], tgt: str) -> list[str]:
    tgt_tsv = TEST_DIR / f"{tgt}.tsv"
    if not tgt_tsv.exists():
        return hyps
    with open(tgt_tsv) as f:
        n_ref = sum(1 for _ in csv.DictReader(f, delimiter="\t"))
    return hyps[:min(len(hyps), n_ref)]


def import_pair(rank: int, src: str, tgt: str) -> tuple[Path, int] | None:
    preds_path = SRC_BASE / f"rank-{rank}" / "mmt" / "lora_model" / f"predictions_{src}-{tgt}.txt"
    if not preds_path.exists():
        return None

    hyps = cap_to_test_split(preds_path.read_text().splitlines(), tgt)

    out_dir = DST_BASE / f"r{rank}" / f"{src}-{tgt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_doc = {
        "schema_version": "1.0",
        "experiment_id": f"mmt_lora_r{rank}_llama_{src}_{tgt}",
        "source_lang": src,
        "target_lang": tgt,
        "samples": [{"hyp": h} for h in hyps],
    }
    (out_dir / "predictions.json").write_text(
        json.dumps(pred_doc, indent=2, ensure_ascii=False))

    result = {
        "schema_version": "1.0",
        "experiment_id": f"mmt_lora_r{rank}_llama_{src}_{tgt}",
        "experiment_type": "multilingual_lora",
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
            "training_mode": "mmt",
            "base_model": "unsloth/meta-llama-3.1-8b-instruct-bnb-4bit",
        },
        "metrics": {"num_samples": len(hyps)},
        "metadata": {
            "source": "imported_from_dcs_mt_old_mmt",
            "original_path": str(preds_path.parent.relative_to(Path("/ext_data/idris/DCS-MT-Old"))),
            "is_ablation": True,
            "import_note": (
                "Multilingual LoRA (single adapter trained on all eng->african "
                "pairs jointly). Imported from DCS-MT-Old/models/unsloth/lora/. "
                "Original results had only sacrebleu BLEU; recompute via "
                "`evaluate --mode rescore` for canonical BLEU / chrF / SSA-COMET. "
                "Only eng-{tgt} directions; no {src}-eng for this run."
            ),
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))

    return out_dir, len(hyps)


def main() -> int:
    n_imported = 0
    n_missing = 0
    for rank in RANKS:
        for pair in PAIRS:
            src, tgt = pair.split("-")
            r = import_pair(rank, src, tgt)
            if r is None:
                print(f"  miss r{rank} {pair}")
                n_missing += 1
            else:
                d, n = r
                print(f"  r{rank} {pair} -> {d.relative_to(REPO)} ({n} samples)")
                n_imported += 1
    print(f"\nimported: {n_imported}  missing: {n_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
