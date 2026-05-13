"""Run a closed-API Google model (gemini-3.1-flash-lite by default) on
the *vanilla* (open-LLM-comparable) prompts:

    sent 0shot, sent 5shot, doc 0shot, doc 1shot

Mirrors scripts/rerun_openai_vanilla.py but uses the Gemini async direct
loop and Google Batch API via generate_with_cache_and_progress from
scripts/rerun_gemini_newer.py.

Usage:
    export GOOGLE_API_KEY=...
    python scripts/rerun_google_vanilla.py \\
        --model gemini-3.1-flash-lite --short-name gemini_31_flash_lite --use-batch
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rerun_gemini_newer import (  # type: ignore  # noqa: E402
    Notifier, cache_path_for, sanitise_short, temp_setting,
    generate_with_cache_and_progress,
)

from afriscience_mt.vanilla_prompts import (  # noqa: E402
    VanillaSentBuilder, VanillaDocBuilder,
)

# Reuse the OpenAI vanilla writers + scorers — they are pure I/O / sacrebleu
# and have no provider-specific code. Just point at the same predictions.json
# schema.
from rerun_openai_vanilla import (  # type: ignore  # noqa: E402
    _write_sent, _write_doc, _score_sent, _score_doc, _read_test_col,
    PAIRS, SENT_SHOTS, DOC_SHOTS,
)

EXP_ROOT = ROOT / "experiments" / "llm" / "closed"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--short-name", default=None)
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--direction", choices=["e2x", "x2e", "both"], default="both")
    ap.add_argument("--only", choices=["sent", "doc", "both"], default="both")
    ap.add_argument("--sent-shots", type=int, nargs="+", default=SENT_SHOTS)
    ap.add_argument("--doc-shots", type=int, nargs="+", default=DOC_SHOTS)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-output-tokens", type=int, default=2048)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--use-batch", action="store_true")
    ap.add_argument("--batch-poll-seconds", type=int, default=120)
    ap.add_argument("--batch-concurrency", type=int, default=3)
    ap.add_argument("--batch-state-file",
                    default="experiments/_google_vanilla_batch_state.json")
    ap.add_argument("--notify-email", default="afriannotate@gmail.com")
    ap.add_argument("--notify-from", default=os.environ.get("BREVO_SENDER_EMAIL"))
    ap.add_argument("--brevo-api-key", default=os.environ.get("BREVO_API_KEY"))
    ap.add_argument("--notify-every", type=int, default=10)
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("error: pass --api-key or set GOOGLE_API_KEY", file=sys.stderr)
        return 2

    short_name = args.short_name or sanitise_short(args.model)
    pair_filter = set(args.pairs) if args.pairs else None
    pairs = []
    for s, t in PAIRS:
        if pair_filter and f"{s}-{t}" not in pair_filter: continue
        if args.direction == "e2x" and s != "eng": continue
        if args.direction == "x2e" and t != "eng": continue
        pairs.append((s, t))

    configs = []
    if args.only in ("sent", "both"):
        for k in args.sent_shots:
            for s, t in pairs:
                configs.append(("sent", k, s, t))
    if args.only in ("doc", "both"):
        for k in args.doc_shots:
            for s, t in pairs:
                configs.append(("doc", k, s, t))

    if args.skip_existing:
        kept = []
        skipped = 0
        for kind, k, s, t in configs:
            if kind == "sent":
                p = (EXP_ROOT / temp_setting(args.temperature) / "icl" /
                     short_name / f"{k}shot" / f"{s}-{t}" / "results.json")
            else:
                p = (EXP_ROOT / "doc" / short_name /
                     f"doc_{k}shot" / f"{s}-{t}" / "results.json")
            if p.exists():
                skipped += 1; continue
            kept.append((kind, k, s, t))
        print(f"  --skip-existing: skipping {skipped} already-completed configs.")
        configs = kept

    if not configs:
        print("nothing to do.")
        return 0

    print(f"Google vanilla rerun: model={args.model} short={short_name} "
          f"T={args.temperature} batch={args.use_batch} configs={len(configs)}")

    from afriscience_mt.api.factory import create_api_client
    api_client = create_api_client(
        provider="google", model_name=args.model, api_key=api_key,
        temperature=args.temperature, max_tokens=args.max_output_tokens,
    )

    sent_builder = VanillaSentBuilder(ROOT)
    doc_builder = VanillaDocBuilder(ROOT)
    state_file = ROOT / args.batch_state_file
    state = json.loads(state_file.read_text()) if state_file.exists() else {}

    notifier = Notifier(
        api_key=args.brevo_api_key, sender=args.notify_from,
        recipient=args.notify_email, every_n=args.notify_every,
    )
    notify_state = {"i": 0, "total": len(configs), "last_cfg": ""}
    if notifier.enabled:
        print(f"notify: enabled -> {args.notify_email} every {args.notify_every} configs")
        notifier.send(
            f"Started: {args.model} (rerun_google_vanilla)",
            f"Model: {args.model}\nShort: {short_name}\n"
            f"T={args.temperature} dir={args.direction} only={args.only}\n"
            f"Configs: {len(configs)}\nMode: {'batch' if args.use_batch else 'direct'}",
        )

        def _excepthook(exc_type, exc_value, exc_tb):
            notifier.send(
                f"FAILED at config {notify_state['i']}/{notify_state['total']}",
                f"Last: {notify_state['last_cfg']}\n"
                f"Error: {exc_type.__name__}: {exc_value}\n\n"
                + "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            )
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        sys.excepthook = _excepthook
    else:
        print("notify: disabled")

    def _build_inputs(kind, k, src, tgt):
        if kind == "sent":
            return {"prompts": sent_builder.build_prompts(src, tgt, k),
                    "srcs": _read_test_col(ROOT, src),
                    "refs": _read_test_col(ROOT, tgt)}
        papers = doc_builder.list_test_papers(tgt)
        inputs = []
        for pid, domain in papers:
            sd = doc_builder.test_doc(src, pid)
            rd = doc_builder.test_doc(tgt, pid)
            inputs.append({"paper_id": pid, "domain": domain, "src": sd, "ref": rd,
                           "prompt": doc_builder.build_prompt(
                               src=src, tgt=tgt, paper_id=pid,
                               src_document=sd, n_shots=k,
                           )})
        inputs.sort(key=lambda r: int(r["paper_id"]))
        return {"inputs": inputs}

    def _out_dir(kind, k, src, tgt):
        if kind == "sent":
            return (EXP_ROOT / temp_setting(args.temperature) / "icl" /
                    short_name / f"{k}shot" / f"{src}-{tgt}")
        return EXP_ROOT / "doc" / short_name / f"doc_{k}shot" / f"{src}-{tgt}"

    def _save(kind, k, src, tgt, hyps, ts, idx, label, payload):
        out_dir = _out_dir(kind, k, src, tgt)
        if kind == "sent":
            srcs, refs = payload["srcs"], payload["refs"]
            metrics = _score_sent(srcs, [(h or "").strip() for h in hyps], refs)
            if metrics:
                print(f"   chrF={metrics['chrf']:.2f}  BLEU={metrics['bleu']:.2f}  "
                      f"({time.time()-ts:.1f}s)")
            _write_sent(out_dir=out_dir, model=args.model, short_name=short_name,
                        src=src, tgt=tgt, n_shots=k, temp=args.temperature,
                        hyps=hyps, srcs=srcs, refs=refs, metrics=metrics)
        else:
            inputs = payload["inputs"]
            samples = [
                {"paper_id": inp["paper_id"], "domain": inp["domain"],
                 "src": inp["src"], "ref": inp["ref"],
                 "hyp": (h or "").strip()}
                for inp, h in zip(inputs, hyps)
            ]
            metrics = _score_doc(samples)
            if metrics:
                print(f"   chrF={metrics['chrf']:.2f}  BLEU={metrics['bleu']:.2f}  "
                      f"({time.time()-ts:.1f}s)")
            _write_doc(out_dir=out_dir, model=args.model, short_name=short_name,
                       src=src, tgt=tgt, n_shots=k, temp=args.temperature,
                       samples=samples, metrics=metrics)
        cp = cache_path_for(out_dir)
        if cp.exists():
            try: cp.unlink()
            except OSError: pass
        notify_state["i"] = idx
        notify_state["last_cfg"] = label
        notifier.maybe_progress(i=idx, total=len(configs), last_cfg=label)

    t0 = time.time()
    if args.use_batch:
        print(f"batch mode: concurrency={args.batch_concurrency}")
        queue = list(configs)
        in_flight: dict[str, dict] = {}
        done_count = 0

        def fill_window():
            while len(in_flight) < args.batch_concurrency and queue:
                kind, k, src, tgt = queue.pop(0)
                pair = f"{src}-{tgt}"
                strat = f"{k}shot" if kind == "sent" else f"doc_{k}shot"
                label = f"{kind}/{strat}/{pair}"
                job_key = f"{short_name}/vanilla/{kind}/{pair}/{strat}"
                payload = _build_inputs(kind, k, src, tgt)
                user_prompts = (payload["prompts"] if kind == "sent"
                                else [r["prompt"] for r in payload["inputs"]])
                print(f"\n[submit] {label}  ({len(user_prompts)} prompts)")
                existing = state.get(job_key)
                if existing:
                    job_name = existing
                    print(f"  resuming: {job_name}")
                else:
                    job_name = api_client.submit_batch_job(
                        prompts=user_prompts,
                        job_name=f"afriscience-{short_name}-vanilla-{kind}-{src}{tgt}-{strat}",
                        system_prompt=None,
                    )
                    state[job_key] = job_name
                    state_file.parent.mkdir(parents=True, exist_ok=True)
                    state_file.write_text(json.dumps(state, indent=2))
                in_flight[job_key] = {
                    "cfg": (kind, k, src, tgt),
                    "payload": payload, "label": label, "job_name": job_name,
                    "started_at": time.time(),
                }

        fill_window()
        while in_flight or queue:
            done_keys = []
            for jk, info in list(in_flight.items()):
                try:
                    hyps = api_client.retrieve_batch_results(info["job_name"])
                except Exception as e:
                    print(f"  x {info['label']} retrieve failed: {e}")
                    continue
                if hyps is None: continue
                kind, k, src, tgt = info["cfg"]
                hyps = [(h or "").strip() for h in hyps]
                print(f"\n[done] {info['label']}  ({len(hyps)} responses)")
                state.pop(jk, None)
                state_file.write_text(json.dumps(state, indent=2))
                done_count += 1
                _save(kind, k, src, tgt, hyps, info["started_at"],
                      done_count, info["label"], info["payload"])
                done_keys.append(jk)
            for jk in done_keys: del in_flight[jk]
            fill_window()
            if in_flight: time.sleep(args.batch_poll_seconds)
    else:
        loop = asyncio.new_event_loop()
        try:
            for idx, (kind, k, src, tgt) in enumerate(configs, 1):
                pair = f"{src}-{tgt}"
                strat = f"{k}shot" if kind == "sent" else f"doc_{k}shot"
                label = f"{kind}/{strat}/{pair}"
                payload = _build_inputs(kind, k, src, tgt)
                user_prompts = (payload["prompts"] if kind == "sent"
                                else [r["prompt"] for r in payload["inputs"]])
                cp = cache_path_for(_out_dir(kind, k, src, tgt))
                print(f"\n[{idx}/{len(configs)}] {label}  ({len(user_prompts)} prompts)")
                ts = time.time()
                hyps = generate_with_cache_and_progress(
                    loop, api_client, prompts=user_prompts,
                    system_prompt=None, cache_path=cp, desc=strat,
                )
                _save(kind, k, src, tgt, hyps, ts, idx, label, payload)
        finally:
            try: loop.close()
            except Exception: pass

    notifier.send(
        f"Run complete: {args.model} (vanilla)",
        f"All {len(configs)} vanilla configs done.\n"
        f"Sent -> experiments/llm/closed/{temp_setting(args.temperature)}/icl/{short_name}/\n"
        f"Doc  -> experiments/llm/closed/doc/{short_name}/",
    )
    print(f"\nTotal: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
