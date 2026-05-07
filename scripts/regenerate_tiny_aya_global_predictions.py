"""Regenerate Tiny-Aya-G LoRA r=64 predictions with vLLM to fix the
sentence-doubling bug.

The original LoRA r=64 fine-tunes for ``CohereLabs/tiny-aya-global`` on the
xxx->eng direction emit ``hyp = "<sent>.<sent>"`` (the same sentence twice),
which collapses SSA-COMET while leaving ChrF/BLEU largely intact:

    pair      COMET   exact-half-dup
    lug-eng   21.9    63.1%
    yor-eng   20.5    67.2%
    zul-eng   33.1    60.1%
    nso-eng   43.3    29.2%
    amh-eng   49.0    19.0%
    hau-eng   55.4    14.1%
    eng->xxx  53-59     0-8%

Adapters live on the Hub at
``AfriScience-MT/tiny_aya_global-lora-r64-{src}-{tgt}``.

This script reloads each LoRA adapter via vLLM, regenerates with explicit
end-of-turn stop tokens + repetition_penalty + frequency_penalty, rescore
BLEU/ChrF/SSA-COMET, and overwrites predictions.json / results.json under
``experiments/llm/open/finetune/tiny_aya_global/r64/<pair>/``.

Usage (on a GPU box, project env with vllm + comet):

    # Just the broken xxx->eng pairs (default; 6 pairs)
    python scripts/regenerate_tiny_aya_global_predictions.py

    # Specific pairs
    python scripts/regenerate_tiny_aya_global_predictions.py \
        --pairs lug-eng yor-eng zul-eng

    # All 12 pairs (also redoes the eng->xxx ones, which were already clean)
    python scripts/regenerate_tiny_aya_global_predictions.py --all

    # Dry-run: write to predictions.json.new instead of overwriting
    python scripts/regenerate_tiny_aya_global_predictions.py --dry-run

After it finishes:
    python scripts/build_paper_tables.py
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "processed_files" / "test"
EXP_DIR = ROOT / "experiments" / "llm" / "open" / "finetune" / "tiny_aya_global" / "r64"

BASE_MODEL = "CohereLabs/tiny-aya-global"
ADAPTER_FMT = "AfriScience-MT/tiny_aya_global-lora-r64-{src}-{tgt}"
LORA_RANK = 64

LANG_NAMES = {
    "eng": "English", "amh": "Amharic", "hau": "Hausa",
    "lug": "Luganda", "nso": "Northern Sotho",
    "yor": "Yoruba", "zul": "isiZulu",
}

ALL_PAIRS = [
    "amh-eng", "hau-eng", "lug-eng", "nso-eng", "yor-eng", "zul-eng",
    "eng-amh", "eng-hau", "eng-lug", "eng-nso", "eng-yor", "eng-zul",
]


# ---------- data ----------

def load_test(lang: str) -> list[dict]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def build_pair_data(src: str, tgt: str) -> list[dict]:
    """Return [{src, ref}] aligned by (paper, segment)."""
    s = {(r["paper"], r["segment"]): r["text"] for r in load_test(src)}
    t = {(r["paper"], r["segment"]): r["text"] for r in load_test(tgt)}
    return [{"src": s[k], "ref": t[k]} for k in s if k in t]


# ---------- prompt (matches training in afriscience_mt/training/lora_trainer.py) ----------

def build_prompt(tokenizer, src_text: str, src_lang: str, tgt_lang: str) -> str:
    instruction = (
        f"Translate the following {LANG_NAMES[src_lang]} scientific text to "
        f"{LANG_NAMES[tgt_lang]}."
    )
    messages = [{"role": "user", "content": f"{instruction}\n\n{src_text}"}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------- vLLM ----------

def collect_stop_token_ids(tokenizer) -> list[int]:
    """End-of-turn ids for Cohere/Aya chat templates plus standard EOS."""
    ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    for cand in ("<|END_OF_TURN_TOKEN|>", "<|end_of_turn|>", "<end_of_turn>"):
        tid = tokenizer.convert_tokens_to_ids(cand)
        if (tid is not None and tid >= 0
                and tid != getattr(tokenizer, "unk_token_id", -1)
                and tid not in ids):
            ids.append(tid)
    return ids


def _patch_vllm_lora_embedding_quant_method() -> None:
    """vLLM 0.19.x's ``VocabParallelEmbeddingWithLoRA`` doesn't expose
    ``quant_method`` (the attribute lives on its wrapped ``base_layer``),
    so engine init crashes with ``AttributeError`` for models that apply
    LoRA to the embedding layer (e.g. tied-embedding Cohere/Aya checkpoints).

    Forward the attribute to ``base_layer`` as a class property.
    """
    try:
        from vllm.lora.layers.vocal_parallel_embedding import (
            VocabParallelEmbeddingWithLoRA,
        )
    except Exception:
        return  # different vLLM layout; nothing to patch
    if "quant_method" in VocabParallelEmbeddingWithLoRA.__dict__:
        return  # already present

    def _get_quant_method(self):
        return self.base_layer.quant_method

    VocabParallelEmbeddingWithLoRA.quant_method = property(_get_quant_method)


def make_llm(gpu_mem_util: float, max_model_len: int):
    """Build a vLLM LLM with LoRA enabled."""
    _patch_vllm_lora_embedding_quant_method()
    from vllm import LLM
    return LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_loras=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_util,
        dtype="bfloat16",
        trust_remote_code=True,
    )


def regenerate_pair(llm, tokenizer, src: str, tgt: str, *,
                    max_new_tokens: int, stop_ids: list[int]) -> list[dict]:
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    samples = build_pair_data(src, tgt)
    prompts = [build_prompt(tokenizer, s["src"], src, tgt) for s in samples]

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        repetition_penalty=1.15,
        frequency_penalty=0.3,
        stop_token_ids=stop_ids,
        skip_special_tokens=True,
    )
    lora_req = LoRARequest(
        lora_name=f"{src}-{tgt}",
        lora_int_id=1,
        lora_path=ADAPTER_FMT.format(src=src, tgt=tgt),
    )

    outputs = llm.generate(prompts, sampling_params=sp, lora_request=lora_req)
    out: list[dict] = []
    for s, o in zip(samples, outputs):
        out.append({"src": s["src"], "ref": s["ref"], "hyp": o.outputs[0].text.strip()})
    return out


# ---------- scoring ----------

def score(samples: list[dict]) -> dict:
    from afriscience_mt.evaluation.metrics import compute_all_metrics
    srcs = [s["src"] for s in samples]
    hyps = [s["hyp"] for s in samples]
    refs = [s["ref"] for s in samples]
    m = compute_all_metrics(srcs, hyps, refs, metrics=["bleu", "chrf", "ssa_comet"])
    return {
        "bleu": m.get("bleu"),
        "chrf": m.get("chrf"),
        "ssa_comet": m.get("ssa_comet"),
        "num_samples": len(samples),
    }


# ---------- IO ----------

def backup(path: Path) -> None:
    if path.exists():
        bk = path.with_suffix(path.suffix + ".bak")
        if not bk.exists():
            shutil.copy2(path, bk)


def update_pair_files(src: str, tgt: str, samples: list[dict],
                      metrics: dict | None) -> None:
    pair_dir = EXP_DIR / f"{src}-{tgt}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    pred_path = pair_dir / "predictions.json"
    res_path = pair_dir / "results.json"
    backup(pred_path)
    backup(res_path)

    pred_path.write_text(json.dumps({
        "experiment_id": f"lora_r64_aya_{src}_{tgt}",
        "source_lang": src,
        "target_lang": tgt,
        "samples": [{"hyp": s["hyp"]} for s in samples],
        "schema_version": "1.0",
    }, indent=2, ensure_ascii=False))

    if res_path.exists():
        res = json.loads(res_path.read_text())
    else:
        res = {
            "schema_version": "1.0",
            "experiment_id": f"lora_r64_aya_{src}_{tgt}",
            "experiment_type": "lora_finetune",
            "source_lang": src,
            "target_lang": tgt,
            "lang_pair": f"{src}-{tgt}",
            "split": "test",
            "dataset": "dcs",
            "model": {
                "name": BASE_MODEL,
                "type": "lora",
                "short_name": "tiny_aya_global",
            },
            "config": {
                "lora_rank": 64,
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "use_dora": False,
                "quantization": "bf16",
            },
            "metadata": {"source": "regenerated"},
        }

    res["timestamp"] = datetime.utcnow().isoformat()
    if metrics is not None:
        res["metrics"] = metrics
    elif "metrics" not in res:
        res["metrics"] = {"num_samples": len(samples)}
    res.setdefault("metadata", {})
    res["metadata"]["regenerated_at"] = datetime.utcnow().isoformat()
    res["metadata"]["regenerated_reason"] = (
        "Original LoRA r=64 predictions exhibited sentence-doubling on xxx->eng "
        "pairs, collapsing SSA-COMET while ChrF/BLEU were largely unaffected. "
        "Regenerated with vLLM: temperature=0, repetition_penalty=1.15, "
        "frequency_penalty=0.3, explicit chat-template stop tokens."
    )
    res["metadata"]["regenerated_engine"] = "vllm"
    res_path.write_text(json.dumps(res, indent=2, ensure_ascii=False))


# ---------- driver ----------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="lang pairs, e.g. lug-eng yor-eng zul-eng")
    ap.add_argument("--all", action="store_true",
                    help="regenerate all 12 pairs (default: just the 6 xxx-eng)")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--dry-run", action="store_true",
                    help="generate but don't overwrite files (writes to .new instead)")
    ap.add_argument("--skip-scoring", action="store_true",
                    help="generate + write predictions only; defer metrics "
                         "(use when the inference env doesn't have COMET, "
                         "then rescore with afriscience-mt evaluate --mode rescore)")
    args = ap.parse_args()

    if args.all:
        pairs = ALL_PAIRS
    elif args.pairs:
        pairs = args.pairs
    else:
        pairs = ["lug-eng", "yor-eng", "zul-eng", "nso-eng", "amh-eng", "hau-eng"]

    invalid = [p for p in pairs if p not in ALL_PAIRS]
    if invalid:
        print(f"unknown pairs: {invalid}", file=sys.stderr)
        return 2

    print(f"regenerating {len(pairs)} pairs with vLLM: {pairs}")
    t0 = time.time()

    llm = make_llm(args.gpu_mem_util, args.max_model_len)
    tokenizer = llm.get_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    stop_ids = collect_stop_token_ids(tokenizer)
    print(f"  base model loaded ({time.time()-t0:.1f}s)  stop_ids={stop_ids}")

    for pair in pairs:
        src, tgt = pair.split("-")
        ts = time.time()
        print(f"\n=== {pair} ===")

        samples = regenerate_pair(
            llm, tokenizer, src, tgt,
            max_new_tokens=args.max_new_tokens,
            stop_ids=stop_ids,
        )
        gen_t = time.time() - ts

        n = len(samples)
        dup = sum(1 for s in samples
                  if len(s["hyp"]) > 60
                  and s["hyp"][:len(s["hyp"]) // 2].strip()
                  == s["hyp"][len(s["hyp"]) // 2:].strip())
        print(f"  generated {n} hyps in {gen_t:.1f}s; "
              f"exact-half duplication: {dup}/{n} ({100*dup/n:.1f}%)")

        if args.skip_scoring:
            metrics = None
            print("  scoring deferred (--skip-scoring)")
        else:
            try:
                metrics = score(samples)
                print(f"  BLEU={metrics['bleu']:.2f}  ChrF={metrics['chrf']:.2f}  "
                      f"SSA-COMET={100*metrics['ssa_comet']:.1f}")
            except Exception as e:
                metrics = None
                print(f"  scoring failed ({type(e).__name__}: {e}); "
                      "writing predictions without metrics — rescore later")

        if args.dry_run:
            out = EXP_DIR / pair / "predictions.json.new"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"samples": samples, "metrics": metrics},
                                       indent=2, ensure_ascii=False))
            print(f"  dry-run, wrote {out}")
        else:
            update_pair_files(src, tgt, samples, metrics)
            print(f"  updated predictions.json + results.json"
                  f"{' (no metrics)' if metrics is None else ''}")

    print(f"\ntotal: {time.time()-t0:.1f}s")
    if args.skip_scoring:
        print("predictions written without metrics. now rescore in the dcs env:")
        print("  python -m afriscience_mt evaluate --mode rescore "
              "--src experiments/llm/open/finetune/tiny_aya_global/r64 "
              "--metric bleu")
        print("  python -m afriscience_mt evaluate --mode rescore "
              "--src experiments/llm/open/finetune/tiny_aya_global/r64 "
              "--metric chrf")
        print("  python -m afriscience_mt evaluate --mode rescore "
              "--src experiments/llm/open/finetune/tiny_aya_global/r64 "
              "--metric ssa_comet")
    else:
        print("now run: python scripts/build_paper_tables.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
