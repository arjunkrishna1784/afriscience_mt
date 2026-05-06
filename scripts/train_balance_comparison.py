#!/usr/bin/env python3
"""Per-domain balance ablation for NLLB-1.3B via temperature sampling.

Three settings on the same train corpus (data/processed_files/train/<lang>.tsv):

  τ = 1   (proportional)  : natural distribution — already exists at
                            experiments/seq2seq/finetune/nllb_200_1_3b/dcs/.
                            We do NOT re-train it here.
  τ = 2   (sqrt)          : weight w_i = (1/N_d)^(1/2)  per example, where
                            N_d is the count of the example's domain.
  τ = inf (uniform)       : weight w_i = 1/N_d                — every domain
                            sampled with equal probability.

Implementation: HF Seq2SeqTrainer with a custom _get_train_sampler() returning
torch.utils.data.WeightedRandomSampler. Total optimizer steps held constant
across τ (same num_samples=len(dataset) per epoch, same epoch count), so the
sampling distribution is the only thing that changes.

Per-pair evaluation on data/processed_files/test/<lang>.tsv. Per-domain
breakdown is computed at write time by joining hyps to the test set's
`category` column.

Outputs:
  experiments/seq2seq/finetune/nllb_200_1_3b/balance_t<2|inf>/<pair>/
    predictions.json   {samples: [{hyp: "..."}]}     # hyp-only (canonical)
    results.json       canonical, metadata.is_ablation = true
                       with metadata.sampling_temperature and per-domain metrics

Usage:
  python scripts/train_balance_comparison.py \\
      --temperatures 2 inf \\
      --pairs eng-amh eng-hau eng-lug eng-nso eng-yor eng-zul \\
              amh-eng hau-eng lug-eng nso-eng yor-eng zul-eng \\
      --num-epochs 5 --batch-size 8 --grad-accum 2

Single-pair smoke test:
  python scripts/train_balance_comparison.py \\
      --temperatures 2 --pairs eng-yor --num-epochs 1 --batch-size 4
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from afriscience_mt.core.schema import (  # noqa: E402
    ExperimentResult, Metrics, ModelInfo, PredictionFile,
    save_predictions, save_result,
)

DATA_TRAIN = REPO / "data" / "processed_files" / "train"
DATA_DEV = REPO / "data" / "processed_files" / "dev"
DATA_TEST = REPO / "data" / "processed_files" / "test"

DEFAULT_PAIRS = [
    "eng-amh", "eng-hau", "eng-lug", "eng-nso", "eng-yor", "eng-zul",
    "amh-eng", "hau-eng", "lug-eng", "nso-eng", "yor-eng", "zul-eng",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="facebook/nllb-200-1.3B",
                   help="HF model id (default: NLLB-1.3B)")
    p.add_argument("--model-short", default="nllb_200_1_3b",
                   help="short name used in result paths")
    p.add_argument("--temperatures", nargs="+", default=["2", "inf"],
                   help="τ values to run; '1' is the existing dcs baseline (skipped)")
    p.add_argument("--pairs", nargs="+", default=DEFAULT_PAIRS,
                   help="lang pairs (src-tgt)")
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2,
                   help="gradient accumulation steps (effective batch = batch * accum)")
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoints-dir", default="checkpoints/balance",
                   help="local scratch for HF training; pruned after each run")
    p.add_argument("--keep-checkpoints", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="print what would be done and exit")
    return p.parse_args(argv)


def parse_temperature(s: str) -> float:
    if s.lower() in ("inf", "infinity", "uniform"):
        return math.inf
    return float(s)


def temp_label(tau: float) -> str:
    return "inf" if math.isinf(tau) else str(int(tau)) if tau == int(tau) else str(tau)


# ---------- data ----------

def _read_tsv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def load_paired(split_dir: Path, src: str, tgt: str) -> list[dict]:
    """Pair rows by (paper, segment) and return [{src, tgt, category}, ...]."""
    src_rows = _read_tsv(split_dir / f"{src}.tsv")
    tgt_rows = _read_tsv(split_dir / f"{tgt}.tsv")
    by_key = {(r["paper"], r["segment"]): r for r in tgt_rows}
    out: list[dict] = []
    for sr in src_rows:
        tr = by_key.get((sr["paper"], sr["segment"]))
        if tr is None:
            continue
        if not (sr.get("text") and tr.get("text")):
            continue
        out.append({
            "src": sr["text"],
            "tgt": tr["text"],
            "category": sr["category"],
        })
    return out


def compute_weights(rows: list[dict], tau: float) -> list[float]:
    """Per-example sampling weight under temperature τ.

      P(d) ∝ N_d^(1/τ)
      weight per example in domain d = (1/N_d)^(1/τ)        # so each domain's
                                                            # total weight ∝ 1/N_d^((1/τ)-1)
                                                            # = N_d * (1/N_d)^(1/τ)
                                                            # = N_d^(1 - 1/τ)
    τ = 1 → uniform per-example (proportional sampling).
    τ → ∞ → weight = 1/N_d  (each domain gets equal total weight; uniform-over-domains).
    """
    cnt = Counter(r["category"] for r in rows)
    if tau == 1:
        return [1.0] * len(rows)
    if math.isinf(tau):
        return [1.0 / cnt[r["category"]] for r in rows]
    inv = 1.0 / tau
    return [pow(1.0 / cnt[r["category"]], inv) for r in rows]


# ---------- training ----------

def _train_and_eval(*, model_name: str, model_short: str, src: str, tgt: str,
                    tau: float, train_rows: list[dict], val_rows: list[dict],
                    test_rows: list[dict], ckpt_dir: Path, args):
    """Fine-tune NLLB with WeightedRandomSampler keyed on per-domain weights."""
    import torch
    from datasets import Dataset
    from torch.utils.data import WeightedRandomSampler
    from transformers import (
        AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
        Seq2SeqTrainer as HFSeq2SeqTrainer,
        Seq2SeqTrainingArguments,
        EarlyStoppingCallback, set_seed,
    )

    from afriscience_mt.utils.language_utils import get_nllb_code, get_m2m100_code  # noqa: F401
    from afriscience_mt.evaluation.metrics import (
        compute_bleu, compute_chrf, MetricsCalculator,
    )

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    src_code = get_nllb_code(src)
    tgt_code = get_nllb_code(tgt)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.src_lang = src_code
    tokenizer.tgt_lang = tgt_code

    def tokenize(batch):
        return tokenizer(
            batch["src"], text_target=batch["tgt"],
            max_length=args.max_length, truncation=True, padding=False,
        )

    train_ds = Dataset.from_list([{"src": r["src"], "tgt": r["tgt"]} for r in train_rows]).map(
        tokenize, batched=True, remove_columns=["src", "tgt"])
    val_ds = (Dataset.from_list([{"src": r["src"], "tgt": r["tgt"]} for r in val_rows]).map(
        tokenize, batched=True, remove_columns=["src", "tgt"]) if val_rows else None)

    weights = compute_weights(train_rows, tau)
    weights_t = torch.tensor(weights, dtype=torch.double)

    # use_safetensors=True avoids torch.load CVE-2025-32434 on older torch (<2.6)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name, use_safetensors=True,
    ).to(device)
    model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_code)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(ckpt_dir),
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
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
        dataloader_num_workers=0,
    )

    class WeightedTrainer(HFSeq2SeqTrainer):
        """Replaces the default RandomSampler with a WeightedRandomSampler.

        num_samples = len(train_dataset) so total optimizer steps equal those of
        a τ=1 (proportional) run with the same epochs and batch size.
        """

        def _get_train_sampler(self, *_, **__):
            n = len(self.train_dataset)
            return WeightedRandomSampler(
                weights=weights_t, num_samples=n, replacement=True,
            )

    trainer = WeightedTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tokenizer, data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_ds else None,
    )
    train_result = trainer.train()
    train_meta = {
        "train_loss": float(train_result.training_loss),
        "train_runtime": float(train_result.metrics.get("train_runtime", 0)),
        "epochs_completed": float(train_result.metrics.get("epoch", args.num_epochs)),
        "n_train_samples": len(train_rows),
    }

    val_metrics = None
    if val_ds is not None:
        val_eval = trainer.evaluate()
        val_metrics = {"loss": float(val_eval.get("eval_loss", 0))}

    print("  generating test translations…")
    hypotheses = _translate(
        model=trainer.model, tokenizer=tokenizer,
        sources=[r["src"] for r in test_rows],
        src_code=src_code, tgt_code=tgt_code,
        max_length=args.max_length, batch_size=args.batch_size, device=device,
    )

    refs = [r["tgt"] for r in test_rows]
    srcs = [r["src"] for r in test_rows]

    bleu = float(compute_bleu(hypotheses, refs)["bleu"])
    chrf = float(compute_chrf(hypotheses, refs)["chrf"])
    ssa_comet = None
    try:
        calc = MetricsCalculator(metrics=["ssa_comet"])
        ssa_comet = calc.compute(hypotheses=hypotheses, references=refs,
                                 sources=srcs).get("ssa_comet")
    except Exception as e:
        print(f"  ssa_comet failed: {e!r} (will need manual rescore)")

    metrics = {
        "bleu": bleu, "chrf": chrf, "ssa_comet": ssa_comet,
        "num_samples": len(refs),
    }
    return metrics, val_metrics, train_meta, hypotheses


def _translate(*, model, tokenizer, sources, src_code, tgt_code,
               max_length, batch_size, device):
    import torch
    from tqdm import tqdm
    model.eval()
    tokenizer.src_lang = src_code
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_code)

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


# ---------- per-domain breakdown ----------

def per_domain_metrics(hypotheses: list[str], test_rows: list[dict],
                       compute_comet: bool, sources: list[str]) -> dict:
    """Group by category and compute per-domain BLEU, chrF, ssa_comet."""
    from collections import defaultdict
    from afriscience_mt.evaluation.metrics import (
        compute_bleu, compute_chrf, MetricsCalculator,
    )
    by_dom = defaultdict(list)
    for hyp, row, src in zip(hypotheses, test_rows, sources):
        by_dom[row["category"]].append((src, row["tgt"], hyp))

    out: dict[str, dict] = {}
    calc = MetricsCalculator(metrics=["ssa_comet"]) if compute_comet else None
    for dom, items in by_dom.items():
        if not items:
            continue
        srcs_d = [it[0] for it in items]
        refs_d = [it[1] for it in items]
        hyps_d = [it[2] for it in items]
        d = {
            "n": len(items),
            "bleu": float(compute_bleu(hyps_d, refs_d)["bleu"]),
            "chrf": float(compute_chrf(hyps_d, refs_d)["chrf"]),
        }
        if compute_comet:
            try:
                d["ssa_comet"] = calc.compute(
                    hypotheses=hyps_d, references=refs_d, sources=srcs_d,
                ).get("ssa_comet")
            except Exception as e:
                d["ssa_comet"] = None
                d["ssa_comet_error"] = repr(e)
        out[dom] = d
    return out


# ---------- write canonical schema ----------

def _write_outputs(out_dir: Path, *, hypotheses: list[str], test_rows: list[dict],
                   metrics: dict, val_metrics, train_meta, per_domain: dict,
                   model_name: str, model_short: str,
                   src: str, tgt: str, tau: float, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = PredictionFile(
        experiment_id=f"balance_t{temp_label(tau)}_{model_short}__{src}-{tgt}",
        source_lang=src, target_lang=tgt,
        samples=[{"hyp": h} for h in hypotheses],
    )
    save_predictions(pf, out_dir / "predictions.json")

    config = {
        "model_name": model_name,
        "sampling_temperature": "inf" if math.isinf(tau) else tau,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "grad_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "precision": args.precision,
        "seed": args.seed,
        "n_train_samples": train_meta["n_train_samples"],
        "n_test_samples": metrics["num_samples"],
    }
    result = ExperimentResult(
        experiment_id=f"balance_t{temp_label(tau)}_{model_short}__{src}-{tgt}",
        experiment_type="seq2seq_finetune",
        timestamp=datetime.utcnow().isoformat(),
        model=ModelInfo(name=model_name, type="seq2seq", short_name=model_short),
        source_lang=src, target_lang=tgt,
        split="test", dataset="dcs",
        config=config,
        metrics=Metrics(bleu=metrics["bleu"], chrf=metrics["chrf"],
                        ssa_comet=metrics["ssa_comet"],
                        num_samples=metrics["num_samples"],
                        extra={"per_domain": per_domain}),
        val_metrics=Metrics(extra=val_metrics) if val_metrics else None,
        train_metrics=train_meta,
        metadata={
            "source": "new",
            "comet_metric": "ssa-comet",
            "is_ablation": True,
            "ablation_kind": "per_domain_balance",
            "sampling_temperature": "inf" if math.isinf(tau) else tau,
            "note": (
                "Domain-balance ablation via temperature sampling "
                "(WeightedRandomSampler on per-example weight (1/N_d)^(1/tau)). "
                "tau=1 corresponds to the existing seq2seq/finetune/<model>/dcs baseline."
            ),
        },
    )
    save_result(result, out_dir / "results.json")


# ---------- driver ----------

def main(argv=None) -> int:
    args = parse_args(argv)
    taus = [parse_temperature(t) for t in args.temperatures]
    taus = [t for t in taus if t != 1]  # τ=1 is already in dcs/
    if not taus:
        print("no taus to run (τ=1 is the existing baseline; pass 2 and/or inf).",
              file=sys.stderr)
        return 1

    pairs = args.pairs
    out_root = REPO / "experiments" / "seq2seq" / "finetune" / args.model_short

    print(f"Settings: model={args.model}  τ={[temp_label(t) for t in taus]}  pairs={len(pairs)}")
    print(f"Output:   {out_root}/balance_t<τ>/<pair>/")

    if args.dry_run:
        for tau in taus:
            for pair in pairs:
                print(f"  would train: τ={temp_label(tau)} {pair}")
        return 0

    for tau in taus:
        for pair in pairs:
            src, tgt = pair.split("-")
            out_dir = out_root / f"balance_t{temp_label(tau)}" / pair
            if (out_dir / "results.json").exists():
                print(f"[skip] τ={temp_label(tau)} {pair}: already done")
                continue

            print(f"\n=== τ={temp_label(tau)} {pair} ===")
            try:
                train_rows = load_paired(DATA_TRAIN, src, tgt)
                val_rows = load_paired(DATA_DEV, src, tgt)
                test_rows = load_paired(DATA_TEST, src, tgt)
            except FileNotFoundError as e:
                print(f"  data missing: {e!r}")
                continue
            print(f"  train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")

            ckpt_dir = REPO / args.checkpoints_dir / f"t{temp_label(tau)}__{pair}"
            t0 = time.time()
            metrics, val_metrics, train_meta, hypotheses = _train_and_eval(
                model_name=args.model, model_short=args.model_short,
                src=src, tgt=tgt, tau=tau,
                train_rows=train_rows, val_rows=val_rows, test_rows=test_rows,
                ckpt_dir=ckpt_dir, args=args,
            )

            srcs_test = [r["src"] for r in test_rows]
            per_dom = per_domain_metrics(
                hypotheses, test_rows,
                compute_comet=metrics["ssa_comet"] is not None,
                sources=srcs_test,
            )

            _write_outputs(
                out_dir, hypotheses=hypotheses, test_rows=test_rows,
                metrics=metrics, val_metrics=val_metrics,
                train_meta=train_meta, per_domain=per_dom,
                model_name=args.model, model_short=args.model_short,
                src=src, tgt=tgt, tau=tau, args=args,
            )
            print(f"  done in {(time.time()-t0)/60:.1f} min: "
                  f"bleu={metrics['bleu']:.2f} chrf={metrics['chrf']:.2f} "
                  f"comet={(metrics['ssa_comet'] or 0):.4f}")

            if not args.keep_checkpoints:
                shutil.rmtree(ckpt_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
