"""Inference-only: run Gemma-2-9B-IT with vLLM across the llama3_8b sweep.

Prompts match DCS-MT-Old/scripts/langchain_eval.py exactly (the Llama/Gemini/
GPT pipeline): a single user message containing the fully-rendered template
from LLMS/templates/template{N}.txt (with {src_text} inlined and a trailing
"{tgt_lang}: " line). Few-shot examples and glossary use the same logic as
DCS-MT-Old/scripts/utils.py (random_state=1 seed, substring matching).

Helpers from utils.py are inlined here so we don't need langchain_core,
jsonlines, or sacrebleu in the vLLM venv. This script only does inference
and writes predictions.json + a stub results.json with null metrics. Run
the dcs-env eval pass afterward (see scripts/run_gemma2_9b_it_templates.sh).

Sweep matches llama3_8b: templates 1/2/4 are 0-shot only; templates 3/5 are
ICL with K in {5,10,20,50}. 12 pairs (eng<->{amh,hau,lug,nso,yor,zul}).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "experiments" / "llm" / "open" / "icl" / "gemma2_9b_it"
TEST_DIR = ROOT / "data" / "processed_files" / "test"
TRAIN_DIR = ROOT / "data" / "processed_files" / "train"

# templates/ = Llama/Gemini/GPT API variant (single user message, src_text inlined).
# templates_offline/ = local-pipeline variant (system+user split). We use the
# former because that's what was used for X->eng in llama3_8b and for all the
# other API-served models the user wants this to be comparable to.
TEMPLATE_DIR = Path("/home/idris/ext_data/DCS-MT-Old/LLMS/templates")
GLOSSARY_DIR = Path("/home/idris/ext_data/DCS-MT-Old/data/glossary")

MODEL_NAME = "google/gemma-2-9b-it"
MODEL_SHORT = "gemma2_9b_it"
TEMP_SETTING = "t0"
MAX_NEW_TOKENS = 128
FEW_SHOT_SEED = 1  # matches utils.py FewShotExampleLoader random_state=1

LANG_NAMES = {
    "eng": "English", "amh": "Amharic", "hau": "Hausa",
    "lug": "Luganda", "nso": "Northern Sotho", "yor": "Yoruba", "zul": "Zulu",
}
LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
PAIRS = [f"eng-{l}" for l in LANGS] + [f"{l}-eng" for l in LANGS]

# (template_id, list_of_shot_counts). 50-shot is dropped — all 970 prompts
# exceed the 4096 max_model_len for every pair (worst case ~7000 tokens
# at amh-eng template5). 20-shot fits comfortably (max ~2992 tokens).
SWEEP = [
    (1, [0]),
    (2, [0]),
    (3, [5, 10, 20]),
    (4, [0]),
    (5, [5, 10, 20]),
]


# ---------- helpers inlined from DCS-MT-Old/scripts/utils.py ----------

def load_tsv_text_col(path: Path) -> list[str]:
    """utils.DatasetLoader equivalent: read .tsv, return 'text' column as list."""
    return pd.read_csv(path, sep="\t")["text"].tolist()


def load_template(path: Path) -> tuple[str, set[str]]:
    """utils.TranslationPromptGenerator._load_template — read template, extract {var} names."""
    text = path.read_text()
    input_vars = {v[v.index("{") + 1:v.index("}")]
                  for v in text.split() if "{" in v and "}" in v}
    return text, input_vars


def format_examples(src_file: Path, tgt_file: Path,
                    src_name: str, tgt_name: str, num_shots: int) -> str:
    """utils.FewShotExampleLoader.format_examples — same random_state=1 sampling."""
    src_sents = load_tsv_text_col(src_file)
    tgt_sents = load_tsv_text_col(tgt_file)
    if len(src_sents) != len(tgt_sents):
        raise ValueError("src/tgt row count mismatch for few-shot pool")
    df = pd.DataFrame({"source": src_sents, "target": tgt_sents})
    rows = df.sample(n=num_shots, random_state=FEW_SHOT_SEED).values.tolist()

    formatted = ""
    for i, (s, t) in enumerate(rows, 1):
        formatted += f"{i}.\n{src_name}: {s}\n{tgt_name}: {t}\n\n"
    return f"\nTranslation examples:\n{formatted.strip()}\n"


def format_glossary(input_text: str, src_code: str, tgt_code: str) -> str:
    """utils.GlossaryManager.format_glossary — substring match, src-first lookup."""
    src_path = GLOSSARY_DIR / f"{src_code}.tsv"
    tgt_path = GLOSSARY_DIR / f"{tgt_code}.tsv"
    if src_path.exists():
        glossary_path, key = src_path, src_code
    elif tgt_path.exists():
        glossary_path, key = tgt_path, tgt_code
    else:
        return "\n"

    df = pd.read_csv(glossary_path, sep="\t")
    matches = []
    lower = input_text.lower()
    # The original matches against df[src_code] (the source-language column),
    # so we look up that column here (utils.py uses self.src_lang).
    col = src_code if src_code in df.columns else key
    for _, row in df.iterrows():
        term = str(row[col]).strip().lower()
        if term and term in lower:
            matches.append((row[src_code] if src_code in df.columns else row[col],
                            row[tgt_code] if tgt_code in df.columns else ""))

    if not matches:
        return "\n"
    out = ""
    for i, (s, t) in enumerate(matches, 1):
        out += f"{i}.\n{src_code}: {s}\n{tgt_code}: {t}\n\n"
    return f"\nGlossary terms:\n{out.strip()}\n"


# ---------- main pipeline ----------

def load_test_refs(lang: str) -> list[str]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


def build_user_prompts(*, src_code: str, tgt_code: str, src_name: str,
                       tgt_name: str, template: int, n_shots: int) -> list[str]:
    """One user-turn string per test sample, matching langchain_eval.py."""
    template_text, input_vars = load_template(TEMPLATE_DIR / f"template{template}.txt")
    texts = load_tsv_text_col(TEST_DIR / f"{src_code}.tsv")

    examples_str = ""
    if "examples" in input_vars and n_shots:
        examples_str = format_examples(
            TRAIN_DIR / f"{src_code}.tsv",
            TRAIN_DIR / f"{tgt_code}.tsv",
            src_name, tgt_name, n_shots,
        )

    prompts = []
    for text in texts:
        glossary_str = ""
        if "glossary" in input_vars:
            glossary_str = format_glossary(text, src_code, tgt_code)
        prompts.append(template_text.format(
            src_lang=src_name, tgt_lang=tgt_name, src_text=text,
            examples=examples_str, glossary=glossary_str,
        ))
    return prompts


def write_canonical(*, out_dir: Path, src_code: str, tgt_code: str,
                    hyps: list[str], template: int, n_shots: int,
                    truncation: dict | None = None) -> None:
    """Write predictions.json + stub results.json (null metrics, dcs fills in)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    pair = f"{src_code}-{tgt_code}"
    strategy = f"template{template}_{n_shots}shot"
    exp_id = f"llm_icl__{MODEL_SHORT}__{strategy}__{pair}"
    ts = datetime.now(timezone.utc).isoformat()

    (out_dir / "predictions.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "source_lang": src_code,
        "target_lang": tgt_code,
        "samples": [{"hyp": h} for h in hyps],
    }, indent=2, ensure_ascii=False))

    (out_dir / "results.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "experiment_type": "llm_icl",
        "timestamp": ts,
        "model": {"name": MODEL_NAME, "type": "llm", "short_name": MODEL_SHORT},
        "source_lang": src_code,
        "target_lang": tgt_code,
        "lang_pair": pair,
        "split": "test",
        "dataset": "dcs",
        "config": {
            "prompt_strategy": strategy,
            "n_shots": n_shots,
            "temp_setting": TEMP_SETTING,
        },
        "metrics": {
            "bleu": None,
            "chrf": None,
            "ssa_comet": None,
            "num_samples": len(hyps),
        },
        "metadata": {
            "source": "generated_by_evaluate_gemma2_9b_it_templates",
            "is_ablation": True,
            "prompt_strategy": strategy,
            "n_shots": n_shots,
            "temp_setting": TEMP_SETTING,
            "max_new_tokens": MAX_NEW_TOKENS,
            "few_shot_seed": FEW_SHOT_SEED,
            "inference_engine": "vllm",
            "template_dir": str(TEMPLATE_DIR),
            "needs_eval": True,
            "truncation": truncation or {"n_truncated": 0},
        },
    }, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--templates", nargs="+", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quiet-vllm", action="store_true",
                    help="hide vLLM's per-prompt progress bar (default: shown)")
    args = ap.parse_args()

    pairs = args.pairs or PAIRS
    sweep = [(t, ks) for t, ks in SWEEP if not args.templates or t in args.templates]

    configs = []
    for tmpl, shots_list in sweep:
        for k in shots_list:
            for pair in pairs:
                src, tgt = pair.split("-")
                strategy = f"template{tmpl}_{k}shot"
                out_dir = OUT_BASE / strategy / pair
                if (out_dir / "predictions.json").exists() and not args.force:
                    continue
                configs.append((tmpl, k, pair, src, tgt, out_dir))

    print(f"configs to run: {len(configs)}  (model={MODEL_NAME})")
    if not configs:
        print("nothing to do — all predictions.json already exist")
        return 0

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vllm not installed in this Python.", file=sys.stderr)
        return 1

    print(f"loading vLLM... (bf16, tp={args.tp}, gpu_mem={args.gpu_mem_frac})")
    llm = LLM(
        model=MODEL_NAME,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_frac,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    tokenizer = llm.get_tokenizer()
    # Reserve room for output + chat-template wrapper (Gemma adds ~7 tokens).
    content_budget = args.max_model_len - MAX_NEW_TOKENS - 16

    def truncate_content(content: str) -> tuple[str, int]:
        """Front-truncate the user message content to fit within budget.
        Drops leading tokens (the instruction header / earliest few-shot
        examples) so the trailing src_text + "{tgt_lang}: " marker is
        always preserved. Returns (content, original_token_len);
        original_token_len > budget means truncation happened."""
        ids = tokenizer(content, add_special_tokens=False)["input_ids"]
        original = len(ids)
        if original <= content_budget:
            return content, original
        ids = ids[-content_budget:]
        return tokenizer.decode(ids, skip_special_tokens=False), original

    pbar = tqdm(configs, desc="configs", unit="cfg")
    for tmpl, k, pair, src, tgt, out_dir in pbar:
        src_name = LANG_NAMES[src]
        tgt_name = LANG_NAMES[tgt]
        pbar.set_postfix_str(f"template{tmpl}_{k}shot {pair}")

        user_prompts = build_user_prompts(
            src_code=src, tgt_code=tgt, src_name=src_name, tgt_name=tgt_name,
            template=tmpl, n_shots=k,
        )

        # Pre-truncate any over-budget prompts; collect stats for metadata.
        truncated_idxs: list[int] = []
        original_lens: list[int] = []
        final_contents: list[str] = []
        for i, p in enumerate(user_prompts):
            content, orig_len = truncate_content(p)
            original_lens.append(orig_len)
            final_contents.append(content)
            if orig_len > content_budget:
                truncated_idxs.append(i)

        trunc_meta = {
            "n_truncated": len(truncated_idxs),
            "n_total": len(user_prompts),
            "content_budget_tokens": content_budget,
            "max_original_tokens": max(original_lens) if original_lens else 0,
        }
        if truncated_idxs:
            tqdm.write(
                f"  ⚠ truncated {len(truncated_idxs)}/{len(user_prompts)} "
                f"prompts in template{tmpl}_{k}shot/{pair} "
                f"(max original={trunc_meta['max_original_tokens']} > {content_budget})"
            )

        conversations = [[{"role": "user", "content": c}] for c in final_contents]
        outputs = llm.chat(conversations, sampling_params=sampling,
                           use_tqdm=not args.quiet_vllm)
        hyps = [o.outputs[0].text.strip() for o in outputs]

        write_canonical(
            out_dir=out_dir, src_code=src, tgt_code=tgt, hyps=hyps,
            template=tmpl, n_shots=k, truncation=trunc_meta,
        )
        tqdm.write(f"  ✓ {MODEL_SHORT}/template{tmpl}_{k}shot/{pair}  (n={len(hyps)})")

    print("\ninference done. run dcs eval to fill in metrics:")
    print("  bash scripts/run_gemma2_9b_it_templates.sh   # already runs eval after inference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
