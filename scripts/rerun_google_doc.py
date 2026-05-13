"""Rerun Google (e.g. gemini-3.1-flash-lite) at the document level
across the full 3-template x 3-shot grid for all 12 directional pairs.

Architecture mirrors ``scripts/rerun_gemini_newer.py`` (sentence-level
Google rerun): async direct mode (10-way concurrent on a shared event
loop) or Google Batch API, Brevo email notifier, per-config result
caching, resumable batch state. The only differences are at the
prompt-construction layer:

  * Prompts are built fresh by ``afriscience_mt.doc_prompts.DocPromptBuilder``
    (no legacy doc-prompts were ever saved); 3 doc-level templates x
    {0, 1, 5}-shot = 9 configs per pair x 12 pairs = 108 configs per run.
  * Each config contains 28 prompts (one per test paper).
  * Outputs land under ``experiments/llm/closed/doc/<short>/doc_{n}shot_template{N}/<pair>/``
    matching the schema used by ``scripts/convert_closed_llm_doc.py`` so
    the existing aggregate / rescore / table-builder pipeline picks them
    up without changes.

Usage:
    export GOOGLE_API_KEY=...
    python scripts/rerun_google_doc.py --model gemini-3.1-flash-lite \\
        --short-name gemini_31_flash_lite
    python scripts/rerun_google_doc.py --model gemini-3.1-flash-lite \\
        --short-name gemini_31_flash_lite --use-batch
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# Make the repo root importable so afriscience_mt and scripts/* resolve.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse Brevo notifier, cache helpers, and the async generator from the
# sentence-level Gemini rerun. They are model-agnostic.
from rerun_gemini_newer import (  # type: ignore  # noqa: E402
    Notifier,
    cache_path_for,
    load_cache,
    append_cache,
    sanitise_short,
    temp_setting,
    generate_with_cache_and_progress,
)

from afriscience_mt.doc_prompts import DocPromptBuilder  # noqa: E402

EXP_ROOT = ROOT / "experiments" / "llm" / "closed"

# ---- experiment grid -------------------------------------------------------
# Doc-level closed-API runs use templates {1, 2, 3} (the three numbered
# doc templates from prompt-templates.tex) crossed with shots {0, 1}.
# That's 3 templates x 2 shots x 12 pairs = 72 cells per model.
#
# We previously ran only template 1 to match the lightweight headline
# config from configs/experiments/llm_closed.yaml, but that left
# gpt-5.4 / gemini-3.1-flash-lite absent from
# tab:doc_template_shot_ablation_comet (which compares the full
# 3-template x 3-shot grid for the legacy gpt-4o / gemini-1.5 doc
# data). With templates 2 + 3 added, the new models populate that
# table directly. 5-shot remains dropped per project policy.
#
# Use --templates / --shot-counts to override.
LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
PAIRS = [(s, t) for s in ("eng", *LANGS) for t in ("eng", *LANGS)
         if s != t and "eng" in (s, t)]
TEMPLATES = [1, 2, 3]
SHOT_COUNTS = [0, 1]
N_PAPERS_EXPECTED = 28


# ---- output writer (doc-level schema) --------------------------------------

def write_doc_canonical(*, out_dir: Path, model: str, short_name: str,
                        src: str, tgt: str, template_n: int, n_shots: int,
                        temp: float, samples: list[dict],
                        metrics: dict | None = None) -> None:
    """Write predictions.json + results.json matching the schema
    produced by scripts/convert_closed_llm_doc.py (and rerun_openai_doc.py)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = f"{src}-{tgt}"
    strategy = f"doc_{n_shots}shot_template{template_n}"
    exp_id = f"llm_doc__{short_name}__{strategy}__{pair}"

    (out_dir / "predictions.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "source_lang": src, "target_lang": tgt,
        "samples": samples,
    }, indent=2, ensure_ascii=False))

    res_metrics = metrics or {
        "bleu": None, "chrf": None, "ssa_comet": None,
        "num_samples": len(samples),
    }

    (out_dir / "results.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "experiment_type": "llm_doc",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_lang": src, "target_lang": tgt, "lang_pair": pair,
        "split": "test", "dataset": "dcs",
        "model": {"name": model, "type": "llm", "short_name": short_name},
        "config": {
            "level": "document",
            "prompt_strategy": strategy,
            "n_shots": n_shots,
            "template": f"template_{template_n}",
            "temperature": temp,
            "temp_setting": temp_setting(temp),
            "num_documents": len(samples),
        },
        "metrics": res_metrics,
        "metadata": {
            "source": "rerun_google_doc",
            "is_ablation": True,
            "prompt_strategy": strategy,
            "n_shots": n_shots,
            "comet_metric": "ssa-comet",
            "needs_eval": metrics is None,
        },
    }, indent=2, ensure_ascii=False))


def score_doc_metrics(samples: list[dict]) -> dict | None:
    """Document-level chrF/BLEU via sacrebleu, treating each paper as one
    long string. SSA-COMET is left None here and filled in by the
    standard rescore CLI."""
    try:
        import sacrebleu
    except ImportError:
        return None
    if not samples:
        return None
    hyps = [s["hyp"] for s in samples]
    refs = [s["ref"] for s in samples]
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return {
        "bleu": bleu, "chrf": chrf, "ssa_comet": None,
        "num_samples": len(samples),
    }


# ---- main ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Gemini model name (e.g. gemini-3.1-flash-lite)")
    ap.add_argument("--short-name", default=None,
                    help="model_short for summary.csv. Defaults to sanitised --model.")
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--direction", choices=["e2x", "x2e", "both"],
                    default="both")
    ap.add_argument("--templates", type=int, nargs="+", default=TEMPLATES)
    ap.add_argument("--shot-counts", type=int, nargs="+", default=SHOT_COUNTS)
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip configs whose results.json already exists.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=4096,
                    help="Doc translations can be long; 4096 default.")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--concurrency", type=int, default=10,
                    help="Concurrent async generations in direct mode.")
    ap.add_argument("--use-batch", action="store_true",
                    help="Use Gemini Batch API (~24h SLA, cheaper).")
    ap.add_argument("--batch-poll-seconds", type=int, default=120)
    ap.add_argument("--batch-concurrency", type=int, default=3,
                    help="Doc-level prompts are large; fewer in flight.")
    ap.add_argument("--batch-state-file",
                    default="experiments/_google_doc_batch_state.json")
    # Brevo email notifications.
    ap.add_argument("--notify-email", default="afriannotate@gmail.com")
    ap.add_argument("--notify-from", default=os.environ.get("BREVO_SENDER_EMAIL"))
    ap.add_argument("--brevo-api-key", default=os.environ.get("BREVO_API_KEY"))
    ap.add_argument("--notify-every", type=int, default=10)
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("error: pass --api-key or set GOOGLE_API_KEY", file=sys.stderr)
        return 2

    bad_shots = [k for k in args.shot_counts if k > 20]
    if bad_shots:
        print(f"error: shot counts > 20 are forbidden: {bad_shots}", file=sys.stderr)
        return 2

    short_name = args.short_name or sanitise_short(args.model)

    pair_filter = set(args.pairs) if args.pairs else None
    pairs = []
    for src, tgt in PAIRS:
        pair_str = f"{src}-{tgt}"
        if pair_filter and pair_str not in pair_filter:
            continue
        if args.direction == "e2x" and src != "eng":
            continue
        if args.direction == "x2e" and tgt != "eng":
            continue
        pairs.append((src, tgt))

    configs: list[tuple[int, int, str, str]] = []
    for tmpl in args.templates:
        for k in args.shot_counts:
            for src, tgt in pairs:
                configs.append((tmpl, k, src, tgt))

    out_root = EXP_ROOT / "doc" / short_name

    if args.skip_existing:
        kept = []
        skipped = 0
        for tmpl, k, src, tgt in configs:
            res = (out_root / f"doc_{k}shot_template{tmpl}"
                   / f"{src}-{tgt}" / "results.json")
            if res.exists():
                skipped += 1
                continue
            kept.append((tmpl, k, src, tgt))
        print(f"  --skip-existing: skipping {skipped} already-completed configs.")
        configs = kept

    if not configs:
        print("no matching configs to run.", file=sys.stderr)
        return 2

    print(f"Google doc rerun: model={args.model}  short={short_name}  "
          f"T={args.temperature}  batch={args.use_batch}  "
          f"configs={len(configs)} (each = 28 prompts)")

    from afriscience_mt.api.factory import create_api_client
    api_client = create_api_client(
        provider="google",
        model_name=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
    )

    builder = DocPromptBuilder(ROOT)
    state_file = ROOT / args.batch_state_file
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    t0 = time.time()

    notifier = Notifier(
        api_key=args.brevo_api_key, sender=args.notify_from,
        recipient=args.notify_email, every_n=args.notify_every,
    )
    if not notifier.enabled:
        print("notify: disabled (need BREVO_API_KEY + --notify-from + --notify-email)")
    else:
        print(f"notify: enabled -> {args.notify_email} every {args.notify_every} configs")
        notifier.send(
            f"Started: {args.model} (rerun_google_doc)",
            f"Model: {args.model}\nShort: {short_name}\n"
            f"T={args.temperature}  direction={args.direction}\n"
            f"Configs: {len(configs)}\n"
            f"Mode: {'batch' if args.use_batch else 'direct (10-way concurrent)'}",
        )

    notify_state = {"i": 0, "total": len(configs), "last_cfg": ""}
    if notifier.enabled:
        def _excepthook(exc_type, exc_value, exc_tb):
            notifier.send(
                f"FAILED at config {notify_state['i']}/{notify_state['total']}",
                f"Last attempted: {notify_state['last_cfg']}\n"
                f"Error: {exc_type.__name__}: {exc_value}\n\n"
                + "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            )
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        sys.excepthook = _excepthook

    def build_config_inputs(tmpl: int, k: int, src: str, tgt: str) -> list[dict]:
        """For one config return [{paper_id, domain, src, ref, prompt}, ...] sorted by paper_id."""
        papers = builder.list_test_papers(tgt)
        if len(papers) != N_PAPERS_EXPECTED:
            print(f"  warn: {tgt} test split has {len(papers)} papers, "
                  f"expected {N_PAPERS_EXPECTED}")
        inputs = []
        for pid, domain in papers:
            src_doc = builder.test_doc(src, pid)
            ref_doc = builder.test_doc(tgt, pid)
            prompt = builder.build_prompt(
                src_lang=src, tgt_lang=tgt,
                template_n=tmpl, n_shots=k,
                paper_id=pid, src_document=src_doc, domain=domain,
            )
            inputs.append({
                "paper_id": pid, "domain": domain,
                "src": src_doc, "ref": ref_doc, "prompt": prompt,
            })
        inputs.sort(key=lambda r: int(r["paper_id"]))
        return inputs

    def hyps_to_samples(inputs: list[dict], hyps: list[str]) -> list[dict]:
        return [
            {"paper_id": inp["paper_id"], "domain": inp["domain"],
             "src": inp["src"], "ref": inp["ref"],
             "hyp": (h or "").strip()}
            for inp, h in zip(inputs, hyps)
        ]

    def score_and_write(*, src, tgt, samples, tmpl, k, out_dir, ts, idx, label):
        metrics = score_doc_metrics(samples)
        if metrics:
            print(f"   chrF={metrics['chrf']:.2f}  "
                  f"BLEU={metrics['bleu']:.2f}  "
                  f"({time.time()-ts:.1f}s)")
        write_doc_canonical(
            out_dir=out_dir, model=args.model, short_name=short_name,
            src=src, tgt=tgt, template_n=tmpl, n_shots=k,
            temp=args.temperature, samples=samples, metrics=metrics,
        )
        cache_path = cache_path_for(out_dir)
        if metrics is not None and cache_path.exists():
            try:
                cache_path.unlink()
            except OSError:
                pass
        notify_state["i"] = idx
        notify_state["last_cfg"] = label
        notifier.maybe_progress(i=idx, total=len(configs), last_cfg=label)

    if args.use_batch:
        print(f"batch mode: concurrency={args.batch_concurrency}, "
              f"poll every {args.batch_poll_seconds}s")
        queue = list(configs)
        in_flight: dict[str, dict] = {}
        done_count = 0

        def fill_window():
            while len(in_flight) < args.batch_concurrency and queue:
                tmpl, k, src, tgt = queue.pop(0)
                pair = f"{src}-{tgt}"
                strategy = f"doc_{k}shot_template{tmpl}"
                label = f"{strategy}/{pair}"
                job_key = f"{short_name}/doc/{pair}/{strategy}"
                out_dir = out_root / strategy / pair

                print(f"\n[submit] {label}")
                inputs = build_config_inputs(tmpl, k, src, tgt)
                user_prompts = [r["prompt"] for r in inputs]
                print(f"  prompts: {len(user_prompts)}  "
                      f"avg-chars: {sum(len(p) for p in user_prompts)//max(1,len(user_prompts))}")

                existing = state.get(job_key)
                if existing:
                    job_name = existing
                    print(f"  resuming pre-submitted job: {job_name}")
                else:
                    job_name = api_client.submit_batch_job(
                        prompts=user_prompts,
                        job_name=f"afriscience-{short_name}-doc-{src}{tgt}-{strategy}",
                        system_prompt=None,
                    )
                    state[job_key] = job_name
                    state_file.parent.mkdir(parents=True, exist_ok=True)
                    state_file.write_text(json.dumps(state, indent=2))
                    print(f"  submitted: {job_name}")

                in_flight[job_key] = {
                    "cfg": (tmpl, k, src, tgt),
                    "inputs": inputs,
                    "label": label,
                    "job_name": job_name,
                    "out_dir": out_dir,
                    "started_at": time.time(),
                }

        fill_window()
        while in_flight or queue:
            completed_keys = []
            for job_key, info in list(in_flight.items()):
                try:
                    hyps = api_client.retrieve_batch_results(info["job_name"])
                except Exception as e:
                    print(f"  x {info['label']} retrieve failed "
                          f"({type(e).__name__}: {e}); will retry next poll")
                    continue
                if hyps is None:
                    continue
                tmpl, k, src, tgt = info["cfg"]
                hyps = [(h or "").strip() for h in hyps]
                print(f"\n[done] {info['label']}  ({len(hyps)} responses)")
                state.pop(job_key, None)
                state_file.write_text(json.dumps(state, indent=2))
                done_count += 1
                samples = hyps_to_samples(info["inputs"], hyps)
                score_and_write(
                    src=src, tgt=tgt, samples=samples,
                    tmpl=tmpl, k=k, out_dir=info["out_dir"],
                    ts=info["started_at"], idx=done_count,
                    label=info["label"],
                )
                completed_keys.append(job_key)
            for jk in completed_keys:
                del in_flight[jk]
            fill_window()
            if in_flight:
                time.sleep(args.batch_poll_seconds)
    else:
        # Direct async mode. One shared event loop across all configs
        # so the SDK's aiohttp session doesn't get reborn per config.
        loop = asyncio.new_event_loop()
        try:
            for idx, (tmpl, k, src, tgt) in enumerate(configs, 1):
                pair = f"{src}-{tgt}"
                strategy = f"doc_{k}shot_template{tmpl}"
                label = f"{strategy}/{pair}"
                out_dir = out_root / strategy / pair
                cache_path = cache_path_for(out_dir)

                inputs = build_config_inputs(tmpl, k, src, tgt)
                user_prompts = [r["prompt"] for r in inputs]
                print(f"\n[{idx}/{len(configs)}] {label}  "
                      f"({len(user_prompts)} prompts)  "
                      f"-> {out_dir.relative_to(ROOT)}")

                ts = time.time()
                hyps = generate_with_cache_and_progress(
                    loop, api_client,
                    prompts=user_prompts, system_prompt=None,
                    cache_path=cache_path, desc=strategy,
                )
                samples = hyps_to_samples(inputs, hyps)
                score_and_write(
                    src=src, tgt=tgt, samples=samples,
                    tmpl=tmpl, k=k, out_dir=out_dir,
                    ts=ts, idx=idx, label=label,
                )
        finally:
            try:
                loop.close()
            except Exception:
                pass

    notifier.send(
        f"Run complete: {args.model} (doc)",
        f"All {len(configs)} doc-level configs done.\n"
        f"Output: experiments/llm/closed/doc/{short_name}/",
    )

    print(f"\nTotal: {time.time() - t0:.1f}s")
    print("Aggregate to update summary.csv:")
    print("  python -m afriscience_mt aggregate --out experiments/summary.csv")
    print("Then SSA-COMET rescore (GPU):")
    print(f"  python scripts/rescore_per_domain_ssa.py "
          f"--src experiments/llm/closed/doc/{short_name}")
    print("Then rebuild paper tables:")
    print("  python scripts/build_paper_tables.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
