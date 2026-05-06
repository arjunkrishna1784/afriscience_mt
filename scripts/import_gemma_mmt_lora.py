"""Import multilingual LoRA (mmt) runs for Gemma-2-9B base.

Source:
  /ext_data/idris/DCS-MT-Old/models/unsloth/lora/gemma-2-9b/
    rank-{8,16,32,64,128}/mmt/lora_model/
      predictions_eng-{amh,hau,lug,nso,yor,zul}.txt   one translation per line
      results_eng-{tgt}.json                           sacrebleu BLEU only
      adapter_config.json                              base = unsloth/gemma-2-9b-bnb-4bit

Single LoRA adapter trained jointly on all eng->african pairs (mmt). 6 pairs
per rank. Sibling block to the Meta-Llama-3.1-8B-Instruct mmt already in
experiments/llm/open/multilingual/llama3_8b/.

Caveat: base is `unsloth/gemma-2-9b-bnb-4bit` — non-instruct variant of
Gemma-2-9B. The existing per-pair LoRA at experiments/llm/open/finetune/
gemma2_9b_it/ uses the *instruct* variant; we use a different model_short
("gemma2_9b") here to avoid mixing them in the same column.

Destination:
  experiments/llm/open/multilingual/gemma2_9b/r<N>/<pair>/
    predictions.json   {samples: [{hyp: "..."}]}      # hyp-only
    results.json       canonical, is_ablation=true
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_BASE = Path("/ext_data/idris/DCS-MT-Old/models/unsloth/lora/gemma-2-9b")
DST_BASE = REPO / "experiments/llm/open/multilingual/gemma2_9b"
TEST_DIR = REPO / "data/processed_files/test"

RANKS = [8, 16, 32, 64, 128]
PAIRS = ["eng-amh", "eng-hau", "eng-lug", "eng-nso", "eng-yor", "eng-zul"]


def cap_to_test(hyps: list[str], tgt: str) -> list[str]:
    tgt_tsv = TEST_DIR / f"{tgt}.tsv"
    if not tgt_tsv.exists():
        return hyps
    with open(tgt_tsv) as f:
        n_ref = sum(1 for _ in csv.DictReader(f, delimiter="\t"))
    return hyps[:min(len(hyps), n_ref)]


def import_pair(rank: int, src: str, tgt: str) -> tuple[Path, int] | None:
    preds = SRC_BASE / f"rank-{rank}" / "mmt" / "lora_model" / f"predictions_{src}-{tgt}.txt"
    if not preds.exists():
        return None
    hyps = cap_to_test(preds.read_text().splitlines(), tgt)

    out_dir = DST_BASE / f"r{rank}" / f"{src}-{tgt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_doc = {
        "schema_version": "1.0",
        "experiment_id": f"mmt_lora_r{rank}_gemma_{src}_{tgt}",
        "source_lang": src,
        "target_lang": tgt,
        "samples": [{"hyp": h} for h in hyps],
    }
    (out_dir / "predictions.json").write_text(
        json.dumps(pred_doc, indent=2, ensure_ascii=False))

    result = {
        "schema_version": "1.0",
        "experiment_id": f"mmt_lora_r{rank}_gemma_{src}_{tgt}",
        "experiment_type": "multilingual_lora",
        "timestamp": datetime.utcnow().isoformat(),
        "model": {
            "name": "google/gemma-2-9b",
            "type": "lora",
            "short_name": "gemma2_9b",
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
            "base_model": "unsloth/gemma-2-9b-bnb-4bit",
            "is_instruct_base": False,
        },
        "metrics": {"num_samples": len(hyps)},
        "metadata": {
            "source": "imported_from_dcs_mt_old_mmt",
            "original_path": str(preds.parent.relative_to(Path("/ext_data/idris/DCS-MT-Old"))),
            "is_ablation": True,
            "import_note": (
                "Gemma-2-9B (base, NOT instruct; 4-bit Unsloth) multilingual "
                "LoRA. Sibling to llama3_8b mmt for the gemma-vs-llama comparison "
                "but with different base capability tier (llama3_8b mmt uses "
                "instruct base). Recompute BLEU/chrF/SSA-COMET via "
                "`evaluate --mode rescore`."
            ),
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))
    return out_dir, len(hyps)


def main() -> int:
    n = 0
    miss = 0
    for r in RANKS:
        for pair in PAIRS:
            src, tgt = pair.split("-")
            res = import_pair(r, src, tgt)
            if res is None:
                print(f"  miss r{r} {pair}")
                miss += 1
            else:
                d, c = res
                print(f"  r{r} {pair} -> {d.relative_to(REPO)} ({c} samples)")
                n += 1
    print(f"\nimported: {n}  missing: {miss}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
