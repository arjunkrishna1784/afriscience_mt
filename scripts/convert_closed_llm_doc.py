"""Convert raw GPT-4o / Gemini-1.5-Flash document-level template predictions
into the canonical schema (one ``predictions.json`` per (pair × strategy ×
template)), score chrF + BLEU, and embed `src`/`ref` directly so the
existing ``afriscience-mt evaluate --mode rescore`` step can compute
SSA-COMET without index lookup against the sentence-level test split.

Source layout:
    experiments/llm/closed/imports/<model>/docs/<strategy>/<pair>/template_{N}/<paper_id>_<domain>.txt
where <model> ∈ {gpt4o, gemini_15_flash},
      <strategy> ∈ {0shot, 1shot, 5shot},
      N ∈ {1, 2, 3}.
Each .txt file is the full document-level translation of one paper.

Target layout (one predictions.json per cell):
    experiments/llm/closed/doc/<model>/<strategy>_template{N}/<pair>/predictions.json
    experiments/llm/closed/doc/<model>/<strategy>_template{N}/<pair>/results.json

`predictions.json.samples` is a list of 28 entries (one per test paper),
each with `{src, ref, hyp, paper_id}`. Embedding `src`/`ref` lets the
rescore CLI score directly without trying to align by index against the
970-row sentence-level test split.

`experiment_type = llm_doc` so it shows up alongside the existing 0shot/1shot
doc rows; `is_ablation = True` (these are template ablations).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parents[1]
IMPORTS_DIR = ROOT / "experiments" / "llm" / "closed" / "imports"
CLOSED_DIR = ROOT / "experiments" / "llm" / "closed"
TEST_DIR = ROOT / "data" / "processed_files" / "test"

MODELS = ["gpt4o", "gemini_15_flash"]
STRATEGIES = ["0shot", "1shot", "5shot"]
TEMPLATES = ["template_1", "template_2", "template_3"]
LANGS = {"amh", "hau", "lug", "nso", "yor", "zul"}
PAPER_FILE_RE = re.compile(r"^(\d+)_([^.]+)\.txt$")  # 102_geography.txt


def model_full_name(short: str) -> str:
    return {"gpt4o": "gpt-4o",
            "gemini_15_flash": "gemini-1.5-flash"}.get(short, short)


def load_papers(lang: str) -> dict[str, list[dict]]:
    """Return {paper_id: [row, row, ...]} for the test split, sorted by segment."""
    papers: dict[str, list[dict]] = defaultdict(list)
    with open(TEST_DIR / f"{lang}.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            papers[r["paper"]].append(r)
    for pid in papers:
        # Segment is numeric in the corpus; sort numerically when possible.
        try:
            papers[pid].sort(key=lambda r: int(r["segment"]))
        except ValueError:
            papers[pid].sort(key=lambda r: r["segment"])
    return papers


def concat_paper_sents(rows: list[dict]) -> str:
    """Join all `text` strings for a paper with single spaces.

    sacrebleu's chrF/BLEU works on document strings — we treat each test
    paper as one long document for both hypothesis and reference."""
    return " ".join(r["text"].strip() for r in rows if r["text"])


def read_doc_translation(path: Path) -> str:
    """Read a per-paper .txt file and return one whitespace-joined string."""
    return " ".join(line.strip() for line in path.read_text().splitlines()
                    if line.strip())


def write_canonical(*, out_dir: Path, model_short: str, pair: str,
                    src_lang: str, tgt_lang: str, samples: list[dict],
                    prompt_strategy: str, n_shots: int,
                    source_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.json"
    res_path = out_dir / "results.json"

    hyps = [s["hyp"] for s in samples]
    refs = [s["ref"] for s in samples]
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

    exp_id = f"llm_doc__{model_short}__{prompt_strategy}__{pair}"
    pred_path.write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "samples": samples,
    }, indent=2, ensure_ascii=False))

    res_path.write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "experiment_type": "llm_doc",
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
            "level": "document",
        },
        "metrics": {
            "bleu": bleu,
            "chrf": chrf,
            "num_samples": len(samples),
        },
        "metadata": {
            "source": "imported_from_closed_doc_template_dump",
            "original_path": str(source_dir.relative_to(ROOT)),
            "is_ablation": True,
            "prompt_strategy": prompt_strategy,
            "n_shots": n_shots,
            "level": "document",
        },
    }, indent=2, ensure_ascii=False))

    return {"chrf": chrf, "bleu": bleu, "n_papers": len(samples)}


def convert_one_cell(model_short: str, strategy: str, pair: str,
                     template: str, *, force: bool) -> tuple[str, dict]:
    src_lang, tgt_lang = pair.split("-")
    src_dir = (IMPORTS_DIR / model_short / "docs" / strategy / pair / template)
    if not src_dir.is_dir():
        return ("missing", {})

    template_n = int(template.split("_")[1])
    prompt_strategy = f"doc_{strategy}_template{template_n}"
    out_dir = (CLOSED_DIR / "doc" / model_short / prompt_strategy / pair)
    if (out_dir / "predictions.json").exists() and not force:
        return ("skip", {})

    src_papers = load_papers(src_lang)
    tgt_papers = load_papers(tgt_lang)

    samples = []
    for f in sorted(src_dir.iterdir()):
        m = PAPER_FILE_RE.match(f.name)
        if not m:
            continue
        paper_id, domain = m.group(1), m.group(2)
        if paper_id not in src_papers or paper_id not in tgt_papers:
            continue
        hyp = read_doc_translation(f)
        ref = concat_paper_sents(tgt_papers[paper_id])
        src = concat_paper_sents(src_papers[paper_id])
        samples.append({
            "paper_id": paper_id,
            "domain": domain,
            "src": src,
            "ref": ref,
            "hyp": hyp,
        })
    if not samples:
        return ("missing", {})
    samples.sort(key=lambda s: int(s["paper_id"]))

    n_shots = {"0shot": 0, "1shot": 1, "5shot": 5}[strategy]
    info = write_canonical(
        out_dir=out_dir, model_short=model_short, pair=pair,
        src_lang=src_lang, tgt_lang=tgt_lang, samples=samples,
        prompt_strategy=prompt_strategy, n_shots=n_shots,
        source_dir=src_dir,
    )
    return ("write", info)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES)
    ap.add_argument("--templates", nargs="*", default=TEMPLATES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    summary = Counter()
    for model_short in args.models:
        base = IMPORTS_DIR / model_short / "docs"
        if not base.is_dir():
            print(f"[{model_short}] no docs/ imports; skipping")
            continue
        print(f"[{model_short}/docs] template ablation")
        for strategy in args.strategies:
            sdir = base / strategy
            if not sdir.is_dir():
                continue
            for pair_dir in sorted(sdir.iterdir()):
                if not pair_dir.is_dir():
                    continue
                pair = pair_dir.name
                src, tgt = pair.split("-")
                if src not in (LANGS | {"eng"}) or tgt not in (LANGS | {"eng"}):
                    continue
                if src == tgt:
                    continue
                for template in args.templates:
                    kind, info = convert_one_cell(
                        model_short, strategy, pair, template,
                        force=args.force)
                    if kind == "write":
                        summary["written"] += 1
                        print(f"  ✓ doc/{model_short}/{strategy}_template"
                              f"{template.split('_')[1]}/{pair}: "
                              f"chrF={info['chrf']:.2f} "
                              f"papers={info['n_papers']}")
                    elif kind == "skip":
                        summary["skipped"] += 1
                    else:
                        summary["missing"] += 1

    print()
    print(f"summary: written={summary['written']}  "
          f"skipped={summary['skipped']}  "
          f"missing={summary['missing']}")


if __name__ == "__main__":
    main()
