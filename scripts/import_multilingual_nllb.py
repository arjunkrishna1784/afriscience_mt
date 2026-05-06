"""Import multilingual NLLB results from DCS-MT-Old.

Source:
  /ext_data/idris/DCS-MT-Old/results/nllb/multilingual/<N>_updates/processed/<src>_<tgt>.txt
where N ∈ {40k, 80k, 200k, 400k} (training updates) and predictions are
1-line-per-test-segment, aligned to data/processed_files/test/<lang>.tsv.

Base model: facebook/nllb-200-distilled-600M (single multilingual checkpoint
trained on all DCS pairs simultaneously, fairseq-style).

Destination:
  experiments/seq2seq/multilingual/nllb_200_distilled_600m/<N>/<src>-<tgt>/
    predictions.json   {samples: [{hyp: "..."}]}     # hyp-only; rescore looks
    results.json       canonical, is_ablation=true   # up src/ref from test/

All 4 checkpoints are flagged as ablation by default. The 400k checkpoint is
typically the main result — promote it later by setting metadata.is_ablation
to false if you want it in the actual table.

Run rescore after to populate BLEU / chrF / SSA-COMET:
  afriscience-mt evaluate --mode rescore \\
      --src experiments/seq2seq/multilingual --metric bleu
  (then chrf, ssa_comet)
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_BASE = Path("/ext_data/idris/DCS-MT-Old/results/nllb/multilingual")
DST_BASE = REPO / "experiments/seq2seq/multilingual/nllb_200_distilled_600m"

CHECKPOINTS = ["40k", "80k", "200k", "400k"]
PAIRS = [
    "eng-amh", "eng-hau", "eng-lug", "eng-nso", "eng-yor", "eng-zul",
    "amh-eng", "hau-eng", "lug-eng", "nso-eng", "yor-eng", "zul-eng",
]


def import_pair(checkpoint: str, src: str, tgt: str) -> tuple[Path, int] | None:
    preds_path = SRC_BASE / f"{checkpoint}_updates" / "processed" / f"{src}_{tgt}.txt"
    if not preds_path.exists():
        return None

    hyps = preds_path.read_text().splitlines()
    out_dir = DST_BASE / checkpoint / f"{src}-{tgt}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_doc = {
        "schema_version": "1.0",
        "experiment_id": f"multilingual_nllb_{checkpoint}_{src}_{tgt}",
        "source_lang": src,
        "target_lang": tgt,
        "samples": [{"hyp": h} for h in hyps],
    }
    (out_dir / "predictions.json").write_text(
        json.dumps(pred_doc, indent=2, ensure_ascii=False))

    result = {
        "schema_version": "1.0",
        "experiment_id": f"multilingual_nllb_{checkpoint}_{src}_{tgt}",
        "experiment_type": "multilingual_finetune",
        "timestamp": datetime.utcnow().isoformat(),
        "model": {
            "name": "facebook/nllb-200-distilled-600M",
            "type": "seq2seq",
            "short_name": "nllb_200_distilled_600m",
        },
        "source_lang": src,
        "target_lang": tgt,
        "lang_pair": f"{src}-{tgt}",
        "split": "test",
        "dataset": "dcs",
        "config": {
            "training_framework": "fairseq",
            "training_updates": checkpoint,
            "checkpoint": checkpoint,
        },
        "metrics": {"num_samples": len(hyps)},
        "metadata": {
            "source": "imported_from_dcs_mt_old_multilingual",
            "original_path": str(preds_path.relative_to(Path("/ext_data/idris/DCS-MT-Old"))),
            "is_ablation": True,
            "import_note": (
                "Imported from DCS-MT-Old/results/nllb/multilingual. Original "
                "metrics file had only BLEU/chrF; recompute via "
                "`afriscience-mt evaluate --mode rescore` for canonical "
                "BLEU / chrF / SSA-COMET. 400k typically the main checkpoint."
            ),
        },
    }
    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False))

    return out_dir, len(hyps)


def main() -> int:
    n_imported = 0
    n_missing = 0
    for ck in CHECKPOINTS:
        for pair in PAIRS:
            src, tgt = pair.split("-")
            r = import_pair(ck, src, tgt)
            if r is None:
                print(f"  miss {ck} {pair}")
                n_missing += 1
            else:
                d, n = r
                print(f"  {ck} {pair} -> {d.relative_to(REPO)} ({n} samples)")
                n_imported += 1
    print(f"\nimported: {n_imported}  missing: {n_missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
