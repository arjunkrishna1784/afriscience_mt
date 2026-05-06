#!/usr/bin/env python3
"""Train NLLB with three data setups and compare on the AfriScience-MT test set.

This is the single entry point that fills the audit gap: it produces the
"with vs without AfriScience" comparison for the 4 MAFAND-supported African
languages (hau, lug, yor, zul) by training the same architecture on:

    --setup mafand        : MAFAND only
    --setup dcs_mafand    : MAFAND + DCS (AfriScience)
    --setup dcs           : DCS only baseline (only if processed_files/nllb/ is present)

All setups evaluate on the **DCS test split** by default (--test-set dcs) so
the new numbers are directly comparable with the existing seq2seq_finetune
results in experiments/. Pass --test-set mafand if you also want the
news-domain MAFAND test for ablation.

Results land under ``experiments/seq2seq_finetune/<model>/<dataset>/<pair>/``
in the canonical schema, so ``afriscience-mt aggregate`` / ``afriscience-mt table`` pick them up
without further work.

Examples
--------
Train one model on one setup for one pair (debug):

    python scripts/train_mafand_comparison.py \
        --model facebook/nllb-200-distilled-600M \
        --setup mafand \
        --pairs eng-hau \
        --num-epochs 5 --batch-size 16

Run the full comparison (3 setups × 4 pairs both directions × 1 model):

    python scripts/train_mafand_comparison.py \
        --model facebook/nllb-200-distilled-600M \
        --setup all \
        --langs hau lug yor zul \
        --directions both \
        --num-epochs 5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

# Make `afriscience_mt` importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from afriscience_mt.core.schema import (
    ExperimentResult,
    Metrics,
    ModelInfo,
    PredictionFile,
    save_predictions,
    save_result,
)
from afriscience_mt.core.registry import MAFAND_LANGS, short_name


SETUPS = ("mafand", "dcs", "dcs_mafand")
DEFAULT_MODELS_BY_TYPE = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "m2m100": "facebook/m2m100_418M",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="facebook/nllb-200-distilled-600M",
                   help="HuggingFace model id (NLLB or M2M100)")
    p.add_argument("--setup", choices=[*SETUPS, "all"], default="all",
                   help="dataset(s) to train: mafand | dcs | dcs_mafand | all")
    p.add_argument("--pairs", nargs="*",
                   help="explicit lang pairs (eng-hau hau-eng ...). Overrides --langs/--directions.")
    p.add_argument("--langs", nargs="*", default=list(MAFAND_LANGS),
                   help="non-eng languages to include (default: hau lug yor zul)")
    p.add_argument("--directions", choices=["from-eng", "to-eng", "both"],
                   default="both", help="when --pairs is omitted")
    p.add_argument("--data-dir",
                   default=str(Path(__file__).resolve().parents[1] / "data"),
                   help="root data dir (must contain processed_files/{mafand,dcs_mafand,nllb}/...)")
    p.add_argument("--out-dir",
                   default=str(Path(__file__).resolve().parents[1] / "experiments" / "seq2seq_finetune"),
                   help="output root for canonical results")
    p.add_argument("--checkpoints-dir",
                   default=str(Path(__file__).resolve().parents[1] / "checkpoints"),
                   help="where intermediate model checkpoints go")
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1,
                   help="multiply effective batch by this without raising memory")
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-set", choices=["dcs", "mafand"], default="dcs",
                   help="which held-out test set to evaluate on (default: dcs, "
                        "for comparability with existing seq2seq_finetune results)")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="don't delete the model checkpoint after evaluation")
    p.add_argument("--skip-train", action="store_true",
                   help="useful for plumbing tests; just emits a stub result")
    p.add_argument("--dry-run", action="store_true",
                   help="print the planned runs without training")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    setups = SETUPS if args.setup == "all" else (args.setup,)
    pairs = _resolve_pairs(args)
    if not pairs:
        print("no pairs to train", file=sys.stderr)
        return 1

    plan = [(setup, pair) for setup in setups for pair in pairs]
    print(f"[plan] model={args.model} runs={len(plan)} "
          f"setups={list(setups)} pairs={pairs}")
    for setup, pair in plan:
        print(f"  - {setup}: {pair}")
    if args.dry_run:
        return 0

    failures = []
    for setup, pair in plan:
        try:
            run_one(args, setup=setup, pair=pair)
        except Exception as e:
            print(f"[FAIL] {setup}/{pair}: {e}", file=sys.stderr)
            failures.append((setup, pair, str(e)))

    if failures:
        print(f"\nCompleted with {len(failures)} failures:", file=sys.stderr)
        for s, p, msg in failures:
            print(f"  - {s}/{p}: {msg}", file=sys.stderr)
        return 1
    return 0


def _resolve_pairs(args) -> list[str]:
    if args.pairs:
        return args.pairs
    pairs = []
    for lang in args.langs:
        if args.directions in ("from-eng", "both"):
            pairs.append(f"eng-{lang}")
        if args.directions in ("to-eng", "both"):
            pairs.append(f"{lang}-eng")
    return pairs


def run_one(args, *, setup: str, pair: str) -> None:
    src, tgt = pair.split("-", 1)
    model_short = short_name(args.model)
    model_type = _model_type(args.model)

    train_path, val_path, flipped = _resolve_train_val_paths(
        Path(args.data_dir), model_type=model_type, setup=setup, pair=pair,
    )
    test_pairs_loaded = _load_test_set(
        Path(args.data_dir), test_set=args.test_set,
        model_type=model_type, src=src, tgt=tgt,
    )
    print(f"\n=== {setup} / {pair} ===")
    print(f"  train: {train_path}{' (flipped from reverse)' if flipped else ''}")
    print(f"  val:   {val_path}")
    print(f"  test:  {args.test_set} (n={len(test_pairs_loaded)})")

    if not train_path.exists():
        raise FileNotFoundError(f"train data missing: {train_path}")
    if not test_pairs_loaded:
        raise FileNotFoundError(
            f"test set '{args.test_set}' empty for {pair} — check data_dir")

    out_dir = Path(args.out_dir) / model_short / setup / pair
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_train:
        print("  [stub] --skip-train, writing placeholder result")
        _write_result(out_dir, model=args.model, model_type=model_type,
                      model_short=model_short, src=src, tgt=tgt, setup=setup,
                      args=args, metrics={}, val_metrics=None, train_meta={},
                      n_train=0, n_test=0)
        return

    ckpt_dir = Path(args.checkpoints_dir) / model_short / setup / pair
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_pairs = _read_jsonl(train_path)
    val_pairs = _read_jsonl(val_path) if val_path and val_path.exists() else []
    if flipped:
        train_pairs = [(t, s) for s, t in train_pairs]
        val_pairs = [(t, s) for s, t in val_pairs]
    test_pairs = test_pairs_loaded
    print(f"  loaded train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    t0 = time.time()
    metrics, val_metrics, train_meta, predictions = _train_and_eval(
        model_name=args.model, model_type=model_type,
        src=src, tgt=tgt,
        train_pairs=train_pairs, val_pairs=val_pairs, test_pairs=test_pairs,
        ckpt_dir=ckpt_dir, args=args,
    )
    train_meta["wall_time_sec"] = round(time.time() - t0, 1)

    _write_result(out_dir, model=args.model, model_type=model_type,
                  model_short=model_short, src=src, tgt=tgt, setup=setup,
                  args=args, metrics=metrics, val_metrics=val_metrics,
                  train_meta=train_meta,
                  n_train=len(train_pairs), n_test=len(test_pairs))

    pf = PredictionFile(
        experiment_id=f"seq2seq__{model_short}__{setup}__{pair}",
        source_lang=src, target_lang=tgt,
        samples=[{"src": s, "ref": r, "hyp": h, "domain": args.test_set}
                 for (s, r), h in zip(test_pairs, predictions)],
    )
    save_predictions(pf, out_dir / "predictions.json")
    print(f"  wrote {out_dir / 'results.json'}")

    if not args.keep_checkpoints:
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    # Free GPU memory before the next pair
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _resolve_train_val_paths(data_root: Path, *, model_type: str, setup: str,
                             pair: str) -> tuple[Path, Path | None, bool]:
    """Locate train + val JSONL for the chosen setup.

    Returns (train_path, val_path_or_None, flipped). When the requested
    direction's data is missing but the *reverse* direction exists, returns
    flipped=True and the caller swaps (src, tgt) per row at load time.
    MAFAND ships eng→xxx only; this lets us train xxx→eng off the same data.
    """
    pf = data_root / "processed_files"
    base = {
        "mafand":     pf / "mafand"     / model_type,
        "dcs":        pf / model_type,
        "dcs_mafand": pf / "dcs_mafand" / model_type,
    }[setup]

    setup_dir = base / pair
    if (setup_dir / "train.json").exists():
        flipped = False
    else:
        # Try reverse direction
        src, tgt = pair.split("-", 1)
        rev_pair = f"{tgt}-{src}"
        setup_dir = base / rev_pair
        if not (setup_dir / "train.json").exists():
            return base / pair / "train.json", None, False  # let caller raise
        flipped = True

    train = setup_dir / "train.json"
    val = setup_dir / "dev.json"
    if not val.exists():
        val_alt = setup_dir / "val.json"
        val = val_alt if val_alt.exists() else None
    return train, val, flipped


def _load_test_set(data_root: Path, *, test_set: str, model_type: str,
                   src: str, tgt: str) -> list[tuple[str, str]]:
    """Return [(src_text, tgt_text), ...] for the chosen test set.

    test_set='dcs'    → align rows of processed_files/test/<lang>.tsv
    test_set='mafand' → read processed_files/mafand/<model_type>/<pair>/test.json
    """
    pf = data_root / "processed_files"
    if test_set == "mafand":
        test_path = pf / "mafand" / model_type / f"{src}-{tgt}" / "test.json"
        return _read_jsonl(test_path)
    # DCS: align two TSVs row-by-row. Each TSV has columns
    # paper, segment, category, text (header on row 1).
    return _load_dcs_test_pair(pf / "test", src=src, tgt=tgt)


def _load_dcs_test_pair(test_dir: Path, *, src: str, tgt: str) -> list[tuple[str, str]]:
    import csv as _csv
    src_path = test_dir / f"{src}.tsv"
    tgt_path = test_dir / f"{tgt}.tsv"
    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(
            f"DCS test missing for {src}-{tgt}: {src_path} or {tgt_path}")

    def _rows(p: Path) -> list[dict]:
        with open(p) as f:
            return list(_csv.DictReader(f, delimiter="\t"))

    src_rows, tgt_rows = _rows(src_path), _rows(tgt_path)
    # Align by (paper, segment) so we don't depend on row order
    by_key = {(r["paper"], r["segment"]): r["text"] for r in tgt_rows}
    out: list[tuple[str, str]] = []
    for r in src_rows:
        ref = by_key.get((r["paper"], r["segment"]))
        if ref is None:
            continue
        if not r.get("text") or not ref:
            continue
        out.append((r["text"], ref))
    return out


def _read_jsonl(path: Path) -> list[tuple[str, str]]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tr = obj.get("translation", obj)
            # Normalize keys to (src, tgt)
            keys = [k for k in tr.keys() if k != "domain"]
            if len(keys) < 2:
                continue
            out.append((tr[keys[0]], tr[keys[1]]))
    return out


def _model_type(model_name: str) -> str:
    n = model_name.lower()
    if "nllb" in n:
        return "nllb"
    if "m2m100" in n:
        return "m2m100"
    raise ValueError(f"unsupported model for this script: {model_name}")


def _train_and_eval(*, model_name, model_type, src, tgt,
                    train_pairs, val_pairs, test_pairs,
                    ckpt_dir: Path, args):
    """Fine-tune the seq2seq model and evaluate on MAFAND test.

    Implementation note: this uses transformers' HF Trainer directly (no
    AfriScienceDataset) so it can consume the on-disk JSONL files for any
    of the three setups without restructuring data.
    """
    # Lazy heavy imports so --dry-run / --help stay fast
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer as HFSeq2SeqTrainer,
        Seq2SeqTrainingArguments,
        EarlyStoppingCallback,
        set_seed,
    )

    from afriscience_mt.utils.language_utils import get_nllb_code, get_m2m100_code
    from afriscience_mt.evaluation.metrics import (
        compute_bleu, compute_chrf, MetricsCalculator,
    )

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Language codes
    if model_type == "nllb":
        src_code, tgt_code = get_nllb_code(src), get_nllb_code(tgt)
    else:
        src_code, tgt_code = get_m2m100_code(src), get_m2m100_code(tgt)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = src_code
    tokenizer.tgt_lang = tgt_code   # required by both NLLB and M2M100 for text_target

    def tokenize(batch):
        srcs = [list(p)[0] for p in batch["pair"]]
        tgts = [list(p)[1] for p in batch["pair"]]
        tokenizer.src_lang = src_code
        tokenizer.tgt_lang = tgt_code
        return tokenizer(
            srcs, text_target=tgts,
            max_length=args.max_length, truncation=True, padding=False,
        )

    # `from_list` keeps the tuple-per-row schema HF datasets expects
    train_ds = Dataset.from_list([{"pair": list(p)} for p in train_pairs]).map(
        tokenize, batched=True, remove_columns=["pair"])
    val_ds = (Dataset.from_list([{"pair": list(p)} for p in val_pairs])
              .map(tokenize, batched=True, remove_columns=["pair"])
              if val_pairs else None)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if model_type == "nllb":
        # Force decoder to start in the target language
        model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,    # trade compute for memory
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch" if val_ds else "no",
        save_strategy="epoch" if val_ds else "no",
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=1,
        predict_with_generate=False,
        fp16=(args.precision == "fp16" and device == "cuda"),
        bf16=(args.precision == "bf16" and device == "cuda"),
        seed=args.seed,
        report_to="none",
    )

    trainer = HFSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_ds else None,
    )
    train_result = trainer.train()
    train_meta = {
        "train_loss": float(train_result.training_loss),
        "train_runtime": float(train_result.metrics.get("train_runtime", 0)),
        "epochs_completed": float(train_result.metrics.get("epoch", args.num_epochs)),
        "n_train_samples": len(train_pairs),
    }

    val_metrics = None
    if val_ds is not None:
        val_eval = trainer.evaluate()
        val_metrics = {"loss": float(val_eval.get("eval_loss", 0))}

    print("  generating test translations…")
    hypotheses = _translate(
        model=trainer.model, tokenizer=tokenizer,
        sources=[s for s, _ in test_pairs],
        model_type=model_type, src_code=src_code, tgt_code=tgt_code,
        max_length=args.max_length, batch_size=args.batch_size, device=device,
    )

    refs = [r for _, r in test_pairs]
    srcs = [s for s, _ in test_pairs]
    bleu_d = compute_bleu(hypotheses, refs)
    chrf_d = compute_chrf(hypotheses, refs)
    bleu = float(bleu_d["bleu"])
    chrf = float(chrf_d["chrf"])
    ssa_comet = None
    try:
        calc = MetricsCalculator(metrics=["ssa_comet"])
        sc = calc.compute(hypotheses=hypotheses, references=refs, sources=srcs)
        ssa_comet = sc.get("ssa_comet")
        if isinstance(ssa_comet, dict):
            ssa_comet = ssa_comet.get("ssa_comet") or ssa_comet.get("score")
    except Exception as e:
        print(f"  (ssa_comet unavailable: {e})")

    print(f"  bleu={bleu:.2f}  chrf={chrf:.2f}  ssa_comet="
          + (f"{ssa_comet:.4f}" if ssa_comet is not None else "n/a"))
    metrics = {"bleu": bleu, "chrf": chrf, "ssa_comet": ssa_comet,
               "num_samples": len(test_pairs)}

    return metrics, val_metrics, train_meta, hypotheses


def _translate(*, model, tokenizer, sources, model_type, src_code, tgt_code,
               max_length, batch_size, device):
    import torch
    from tqdm import tqdm
    model.eval()
    tokenizer.src_lang = src_code
    if model_type == "nllb":
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)
    else:
        forced_bos = tokenizer.get_lang_id(tgt_code)

    out: list[str] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(sources), batch_size),
                      desc="translate", leave=False):
            batch = sources[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=max_length).to(device)
            gen = model.generate(**inputs, forced_bos_token_id=forced_bos,
                                 max_length=max_length, num_beams=5,
                                 early_stopping=True)
            out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


def _write_result(out_dir: Path, *, model, model_type, model_short,
                  src, tgt, setup, args, metrics, val_metrics, train_meta,
                  n_train, n_test) -> None:
    config = {
        "model_name": model,
        "model_type": model_type,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "precision": args.precision,
        "seed": args.seed,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
    }
    result = ExperimentResult(
        experiment_id=f"seq2seq__{model_short}__{setup}__{src}-{tgt}",
        experiment_type="seq2seq_finetune",
        timestamp=datetime.utcnow().isoformat(),
        model=ModelInfo(name=model, type="seq2seq", short_name=model_short),
        source_lang=src,
        target_lang=tgt,
        split="test",
        dataset=setup,
        config=config,
        metrics=Metrics(
            bleu=metrics.get("bleu"),
            chrf=metrics.get("chrf"),
            ssa_comet=metrics.get("ssa_comet"),
            num_samples=metrics.get("num_samples"),
        ),
        val_metrics=Metrics(extra=val_metrics) if val_metrics else None,
        train_metrics=train_meta,
        metadata={
            "source": "new",
            "comet_metric": "ssa-comet",
            "test_source": args.test_set,
            "train_source": setup,
        },
    )
    save_result(result, out_dir / "results.json")


if __name__ == "__main__":
    raise SystemExit(main())
