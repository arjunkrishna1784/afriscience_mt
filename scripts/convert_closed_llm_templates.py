"""Convert raw GPT-4o / Gemini-1.5-Flash template predictions into the
canonical schema (predictions.json + results.json), then score chrF + BLEU.

Source layout (mirrors the recent import commit):
    experiments/llm/closed/imports/<model>/t0/<pair>/template{N}-{K}shots.txt
    experiments/llm/closed/imports/<model>/t06/<pair>/template{N}-{K}shots.txt
    experiments/llm/closed/imports/<model>/semantic/<pair>/{K}shots.txt
where <model> ∈ {gpt4o, gemini_15_flash}, N ∈ 1..5, K ∈ {0,5,10,20,50}.

Target layout (canonical):
    experiments/llm/closed/<temp>/icl/<model>/template{N}_{K}shot/<pair>/predictions.json
    experiments/llm/closed/<temp>/icl/<model>/template{N}_{K}shot/<pair>/results.json
where <temp> ∈ {t0, t06}.
Semantic variants land at:
    experiments/llm/closed/t0/icl/<model>/semantic_{K}shot/<pair>/predictions.json

Each generated row gets:
    experiment_type = llm_icl
    prompt_strategy = template{N}_{K}shot   (or semantic_{K}shot)
    n_shots         = K
    temp_setting    = t0 / t06
    is_ablation     = True   (these are template-sweep ablations of the
                              canonical 5shot / 5shot_semantic runs)

Document-level imports under ``imports/<model>/docs/...`` are left for a
later pass — they're 1-file-per-paper and need a per-paper concat step.
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
IMPORTS_DIR = ROOT / "experiments" / "llm" / "closed" / "imports"
CLOSED_DIR = ROOT / "experiments" / "llm" / "closed"
TEST_DIR = ROOT / "data" / "processed_files" / "test"

EXPECTED_N = 970

MODELS = ["gpt4o", "gemini_15_flash"]
LANGS = {"amh", "hau", "lug", "nso", "yor", "zul"}

# template{N}-{K}shots.txt or {K}shots.txt for semantic
TEMPLATE_RE = re.compile(r"^template(\d+)-(\d+)shots\.txt$")
SEMANTIC_RE = re.compile(r"^(\d+)shots\.txt$")


def load_test(lang: str) -> list[str]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


def model_full_name(short: str) -> str:
    return {"gpt4o": "gpt-4o", "gemini_15_flash": "gemini-1.5-flash"}.get(short, short)


def normalise_hyps(path: Path) -> tuple[list[str], int]:
    """Read a .txt prediction file and pad/truncate to EXPECTED_N."""
    hyps = path.read_text().splitlines()
    n = len(hyps)
    if n < EXPECTED_N:
        return hyps + [""] * (EXPECTED_N - n), EXPECTED_N - n
    if n > EXPECTED_N:
        return hyps[:EXPECTED_N], -(n - EXPECTED_N)
    return hyps, 0


def write_canonical(*, out_dir: Path, model_short: str, pair: str,
                    src_lang: str, tgt_lang: str, hyps: list[str],
                    pad: int, prompt_strategy: str, n_shots: int,
                    temp_setting: str, source_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.json"
    res_path = out_dir / "results.json"

    refs = load_test(tgt_lang)
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

    exp_id = f"llm_icl__{model_short}__{prompt_strategy}__{pair}"
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
            "name": model_full_name(model_short),
            "type": "llm",
            "short_name": model_short,
        },
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "lang_pair": pair,
        "split": "test",
        "dataset": "dcs",
        "config": {
            "prompt_strategy": prompt_strategy,
            "n_shots": n_shots,
            "temp_setting": temp_setting,
        },
        "metrics": {
            "bleu": bleu,
            "chrf": chrf,
            "num_samples": len(hyps),
        },
        "metadata": {
            "source": "imported_from_closed_template_dump",
            "original_path": str(source_path.relative_to(ROOT)),
            "is_ablation": True,
            "prompt_strategy": prompt_strategy,
            "n_shots": n_shots,
            "temp_setting": temp_setting,
            "import_note": (
                f"converted from {source_path.name}; padded {pad} empty rows"
                if pad > 0 else
                (f"truncated {-pad} extra rows" if pad < 0 else
                 "1:1 conversion, no padding/truncation needed")
            ),
        },
    }, indent=2, ensure_ascii=False))

    return {"chrf": chrf, "bleu": bleu, "pad": pad}


def convert_template_dir(model_short: str, temp_setting: str, pair_dir: Path,
                         force: bool, summary: Counter):
    """Iterate template{N}-{K}shots.txt within one pair dir."""
    src_lang, tgt_lang = pair_dir.name.split("-")
    for f in sorted(pair_dir.iterdir()):
        m = TEMPLATE_RE.match(f.name)
        if not m:
            continue
        tmpl_n, n_shots = int(m.group(1)), int(m.group(2))
        prompt_strategy = f"template{tmpl_n}_{n_shots}shot"
        out_dir = (CLOSED_DIR / temp_setting / "icl" / model_short
                   / prompt_strategy / pair_dir.name)
        if (out_dir / "predictions.json").exists() and not force:
            summary["skipped"] += 1
            continue
        hyps, pad = normalise_hyps(f)
        info = write_canonical(
            out_dir=out_dir, model_short=model_short, pair=pair_dir.name,
            src_lang=src_lang, tgt_lang=tgt_lang, hyps=hyps, pad=pad,
            prompt_strategy=prompt_strategy, n_shots=n_shots,
            temp_setting=temp_setting, source_path=f,
        )
        summary["written"] += 1
        tag = (f" pad={pad:+d}" if pad else "")
        print(f"  ✓ {temp_setting}/{model_short}/{prompt_strategy}/{pair_dir.name}: "
              f"chrF={info['chrf']:.2f}{tag}")


def convert_semantic_dir(model_short: str, pair_dir: Path,
                         force: bool, summary: Counter):
    """Iterate {K}shots.txt within one semantic pair dir (always t0)."""
    src_lang, tgt_lang = pair_dir.name.split("-")
    for f in sorted(pair_dir.iterdir()):
        m = SEMANTIC_RE.match(f.name)
        if not m:
            continue
        n_shots = int(m.group(1))
        prompt_strategy = f"semantic_{n_shots}shot"
        out_dir = (CLOSED_DIR / "t0" / "icl" / model_short
                   / prompt_strategy / pair_dir.name)
        if (out_dir / "predictions.json").exists() and not force:
            summary["skipped"] += 1
            continue
        hyps, pad = normalise_hyps(f)
        info = write_canonical(
            out_dir=out_dir, model_short=model_short, pair=pair_dir.name,
            src_lang=src_lang, tgt_lang=tgt_lang, hyps=hyps, pad=pad,
            prompt_strategy=prompt_strategy, n_shots=n_shots,
            temp_setting="t0", source_path=f,
        )
        summary["written"] += 1
        tag = (f" pad={pad:+d}" if pad else "")
        print(f"  ✓ t0/{model_short}/{prompt_strategy}/{pair_dir.name}: "
              f"chrF={info['chrf']:.2f}{tag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    summary = Counter()
    for model_short in args.models:
        base = IMPORTS_DIR / model_short
        if not base.is_dir():
            print(f"[{model_short}] no imports dir; skipping")
            continue
        for temp_setting in ("t0", "t06"):
            tdir = base / temp_setting
            if not tdir.is_dir():
                continue
            print(f"[{model_short}/{temp_setting}] template ablation")
            for pair_dir in sorted(tdir.iterdir()):
                if not pair_dir.is_dir():
                    continue
                src, tgt = pair_dir.name.split("-")
                if src not in (LANGS | {"eng"}) or tgt not in (LANGS | {"eng"}):
                    continue
                if src == tgt:
                    continue
                convert_template_dir(model_short, temp_setting, pair_dir,
                                     args.force, summary)
        sdir = base / "semantic"
        if sdir.is_dir():
            print(f"[{model_short}/semantic] shot-count ablation")
            for pair_dir in sorted(sdir.iterdir()):
                if not pair_dir.is_dir():
                    continue
                convert_semantic_dir(model_short, pair_dir, args.force, summary)

    print()
    print(f"summary: written={summary['written']}  "
          f"skipped={summary['skipped']}")


if __name__ == "__main__":
    main()
