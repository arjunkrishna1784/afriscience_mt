"""Convert legacy Llama-3.1-8B template/shot ablation TSVs into the canonical
schema (predictions.json + results.json), then score chrF + BLEU.

Mirrors ``convert_closed_llm_templates.py`` but for open-source models. The
upstream legacy files live in DCS-MT-Old and split by direction:

    eng→X (6 pairs):
        DCS-MT-Old/LLMS/translations/Meta-Llama-3.1-8B-Instruct/
            eng-to-{tgt}-template{N}-shots{K}.tsv
    X→eng (6 pairs):
        DCS-MT-Old/LLMS/translations/meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo-128K_temp_0.0/
            {src}-to-eng-template{N}-shots{K}.tsv

Each TSV has columns ``prompt`` and ``translation`` and exactly 970 rows.
N ∈ 1..5, K ∈ {0,5,10,20,50}; only the (template, shots) pairs actually run
exist on disk (zero-shot for templates 1/2/4, ICL shots for templates 3/5).

Target layout (canonical, matches existing open ICL runs):
    experiments/llm/open/icl/<model_short>/<strategy>/<pair>/predictions.json
    experiments/llm/open/icl/<model_short>/<strategy>/<pair>/results.json
where <strategy> = ``template{N}_{K}shot``.

Per-row metadata distinguishes the two upstream sources so we can audit:
    metadata.import_source = "Meta-Llama-3.1-8B-Instruct"   (eng→X, local FP16)
                          | "Meta-Llama-3.1-8B-Instruct-Turbo-128K"  (X→eng, Together FP8)

Both share the same architecture; the Turbo variant differs only in
inference engine + quantization. Flagged in metadata so we can sanity-check
later if the cross-direction comparison looks suspicious.

Each generated row gets:
    experiment_type = llm_icl
    prompt_strategy = template{N}_{K}shot
    n_shots         = K
    temp_setting    = t0
    is_ablation     = True

Run:
    python scripts/convert_open_llm_templates.py
    python scripts/rescore_missing_ssa.py --src experiments/llm/open/icl
    afriscience-mt aggregate
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
OPEN_ICL_DIR = ROOT / "experiments" / "llm" / "open" / "icl"
TEST_DIR = ROOT / "data" / "processed_files" / "test"

LEGACY_ROOT = Path("/home/idris/ext_data/DCS-MT-Old/LLMS/translations")

# (legacy_dirname, direction_label, source_tag)
LEGACY_SOURCES = [
    ("Meta-Llama-3.1-8B-Instruct",
     "eng_to_X",
     "Meta-Llama-3.1-8B-Instruct"),
    ("meta-llama_Meta-Llama-3.1-8B-Instruct-Turbo-128K_temp_0.0",
     "X_to_eng",
     "Meta-Llama-3.1-8B-Instruct-Turbo-128K"),
]

MODEL_SHORT = "llama3_8b"
MODEL_FULL = "meta-llama/Llama-3.1-8B-Instruct"
TEMP_SETTING = "t0"
EXPECTED_N = 970

LANGS = {"amh", "hau", "lug", "nso", "yor", "zul"}

# Filenames look like  eng-to-amh-template1-shots0.tsv
FNAME_RE = re.compile(
    r"^(?P<src>[a-z]+)-to-(?P<tgt>[a-z]+)-template(?P<n>\d+)-shots(?P<k>\d+)(?P<suffix>(-old)?)\.tsv$"
)


def load_test_refs(lang: str) -> list[str]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


def read_tsv_translations(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return [(r.get("translation") or "").strip()
                for r in csv.DictReader(f, delimiter="\t")]


def normalise_hyps(hyps: list[str]) -> tuple[list[str], int]:
    """Pad/truncate to EXPECTED_N. Return (hyps, pad). pad>0 means padded
    with empties; pad<0 means truncated."""
    n = len(hyps)
    if n < EXPECTED_N:
        return hyps + [""] * (EXPECTED_N - n), EXPECTED_N - n
    if n > EXPECTED_N:
        return hyps[:EXPECTED_N], -(n - EXPECTED_N)
    return hyps, 0


def write_canonical(*, out_dir: Path, pair: str, src_lang: str, tgt_lang: str,
                    hyps: list[str], pad: int, prompt_strategy: str,
                    n_shots: int, source_path: Path, source_tag: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.json"
    res_path = out_dir / "results.json"

    refs = load_test_refs(tgt_lang)
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

    exp_id = f"llm_icl__{MODEL_SHORT}__{prompt_strategy}__{pair}"
    pred_path.write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "samples": [{"hyp": h} for h in hyps],
    }, indent=2, ensure_ascii=False))

    res_path.write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "experiment_type": "llm_icl",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": MODEL_FULL,
            "type": "llm",
            "short_name": MODEL_SHORT,
        },
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "lang_pair": pair,
        "split": "test",
        "dataset": "dcs",
        "config": {
            "prompt_strategy": prompt_strategy,
            "n_shots": n_shots,
            "temp_setting": TEMP_SETTING,
        },
        "metrics": {
            "bleu": bleu,
            "chrf": chrf,
            "num_samples": len(hyps),
        },
        "metadata": {
            "source": "imported_from_open_template_dump",
            "import_source": source_tag,
            "original_path": str(source_path),
            "is_ablation": True,
            "prompt_strategy": prompt_strategy,
            "n_shots": n_shots,
            "temp_setting": TEMP_SETTING,
            "import_note": (
                f"converted from {source_path.name}; padded {pad} empty rows"
                if pad > 0 else
                (f"truncated {-pad} extra rows" if pad < 0 else
                 "1:1 conversion, no padding/truncation needed")
            ),
        },
    }, indent=2, ensure_ascii=False))

    return {"chrf": chrf, "bleu": bleu, "pad": pad}


def convert_file(tsv_path: Path, source_tag: str, *, force: bool,
                 summary: Counter) -> None:
    m = FNAME_RE.match(tsv_path.name)
    if not m:
        return
    if m.group("suffix"):
        # skip *-old.tsv duplicates
        summary["skipped_old_dup"] += 1
        return
    src, tgt = m.group("src"), m.group("tgt")
    n_shots = int(m.group("k"))
    tmpl_n = int(m.group("n"))
    if src == tgt:
        return
    if src not in (LANGS | {"eng"}) or tgt not in (LANGS | {"eng"}):
        return

    pair = f"{src}-{tgt}"
    prompt_strategy = f"template{tmpl_n}_{n_shots}shot"
    out_dir = OPEN_ICL_DIR / MODEL_SHORT / prompt_strategy / pair

    if (out_dir / "predictions.json").exists() and not force:
        summary["skipped"] += 1
        return

    hyps_raw = read_tsv_translations(tsv_path)
    hyps, pad = normalise_hyps(hyps_raw)
    info = write_canonical(
        out_dir=out_dir, pair=pair, src_lang=src, tgt_lang=tgt,
        hyps=hyps, pad=pad, prompt_strategy=prompt_strategy,
        n_shots=n_shots, source_path=tsv_path, source_tag=source_tag,
    )
    summary["written"] += 1
    tag = (f" pad={pad:+d}" if pad else "")
    print(f"  ✓ {MODEL_SHORT}/{prompt_strategy}/{pair}: "
          f"chrF={info['chrf']:.2f} BLEU={info['bleu']:.2f}{tag}  ({source_tag})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-convert even if predictions.json already exists")
    args = ap.parse_args()

    if not LEGACY_ROOT.is_dir():
        raise SystemExit(f"legacy root not found: {LEGACY_ROOT}")

    summary: Counter = Counter()
    for dirname, direction, source_tag in LEGACY_SOURCES:
        legacy_dir = LEGACY_ROOT / dirname
        if not legacy_dir.is_dir():
            print(f"[skip] missing legacy dir: {legacy_dir}")
            continue
        print(f"[{direction}] {dirname}")
        for tsv in sorted(legacy_dir.glob("*.tsv")):
            convert_file(tsv, source_tag, force=args.force, summary=summary)

    print()
    print(f"summary: written={summary['written']} "
          f"skipped={summary['skipped']} "
          f"skipped_old_dup={summary['skipped_old_dup']}")


if __name__ == "__main__":
    main()
