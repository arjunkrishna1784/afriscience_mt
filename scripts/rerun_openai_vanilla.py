"""Run a closed-API OpenAI model (gpt-5.4 by default) on the *vanilla*
(open-LLM-comparable) prompts:

    sentence-level   0shot   ->  experiments/llm/closed/t0/icl/<short>/0shot/<pair>/
    sentence-level   5shot   ->  experiments/llm/closed/t0/icl/<short>/5shot/<pair>/
    document-level   doc_0shot -> experiments/llm/closed/doc/<short>/doc_0shot/<pair>/
    document-level   doc_1shot -> experiments/llm/closed/doc/<short>/doc_1shot/<pair>/

The strategy names match the open-LLM directory layout
(experiments/llm/open/icl/<model>/{0,5}shot/, .../doc/<model>/{0,1}shot/),
so the resulting cells are byte-comparable to the open-LLM baselines
already on disk.

Architecture mirrors scripts/rerun_openai_doc.py: direct or batch mode,
Brevo notifier, --skip-existing, batch-state resume. Sent and doc are
both run in the same invocation; pass --only sent or --only doc to
restrict.

Usage:
    export OPENAI_API_KEY=...
    python scripts/rerun_openai_vanilla.py --model gpt-5.4 --short-name gpt54 --use-batch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rerun_gpt4o_templated import (  # type: ignore  # noqa: E402
    Notifier,
    cache_path_for,
    load_cache,
    append_cache,
    sanitise_short,
    temp_setting,
)

from afriscience_mt.vanilla_prompts import (  # noqa: E402
    VanillaSentBuilder,
    VanillaDocBuilder,
)

EXP_ROOT = ROOT / "experiments" / "llm" / "closed"

LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
PAIRS = [(s, t) for s in ("eng", *LANGS) for t in ("eng", *LANGS)
         if s != t and "eng" in (s, t)]
SENT_SHOTS = [0, 5]
DOC_SHOTS = [0, 1]
N_PAPERS_EXPECTED = 28


def _write_sent(*, out_dir: Path, model: str, short_name: str,
                src: str, tgt: str, n_shots: int, temp: float,
                hyps: list[str], srcs: list[str], refs: list[str],
                metrics: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = f"{src}-{tgt}"
    strategy = f"{n_shots}shot"
    exp_id = f"llm_icl__{short_name}__{strategy}__{pair}"
    n = min(len(hyps), len(srcs), len(refs))
    samples = [
        {"src": srcs[i], "ref": refs[i], "hyp": (hyps[i] or "").strip()}
        for i in range(n)
    ]
    (out_dir / "predictions.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "source_lang": src, "target_lang": tgt,
        "samples": samples,
    }, indent=2, ensure_ascii=False))
    (out_dir / "results.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "experiment_type": "llm_icl",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_lang": src, "target_lang": tgt, "lang_pair": pair,
        "split": "test", "dataset": "dcs",
        "model": {"name": model, "type": "llm", "short_name": short_name},
        "config": {
            "prompt_strategy": strategy, "n_shots": n_shots,
            "temperature": temp, "temp_setting": temp_setting(temp),
            "template": "explicit" if n_shots == 0 else "few_shot",
            "selection_method": "random" if n_shots > 0 else None,
        },
        "metrics": metrics or {"bleu": None, "chrf": None, "ssa_comet": None,
                                "num_samples": n},
        "metadata": {
            "source": "rerun_openai_vanilla",
            "is_ablation": False,
            "prompt_strategy": strategy,
            "n_shots": n_shots,
            "comet_metric": "ssa-comet",
            "needs_eval": metrics is None,
        },
    }, indent=2, ensure_ascii=False))


def _write_doc(*, out_dir: Path, model: str, short_name: str,
               src: str, tgt: str, n_shots: int, temp: float,
               samples: list[dict], metrics: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = f"{src}-{tgt}"
    strategy = f"doc_{n_shots}shot"
    exp_id = f"llm_doc__{short_name}__{strategy}__{pair}"
    (out_dir / "predictions.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "source_lang": src, "target_lang": tgt,
        "samples": samples,
    }, indent=2, ensure_ascii=False))
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
            "template": "basic",
            "temperature": temp,
            "temp_setting": temp_setting(temp),
            "num_documents": len(samples),
        },
        "metrics": metrics or {"bleu": None, "chrf": None, "ssa_comet": None,
                                "num_samples": len(samples)},
        "metadata": {
            "source": "rerun_openai_vanilla",
            "is_ablation": False,
            "prompt_strategy": strategy,
            "n_shots": n_shots,
            "comet_metric": "ssa-comet",
            "needs_eval": metrics is None,
        },
    }, indent=2, ensure_ascii=False))


def _score_sent(srcs, hyps, refs):
    try:
        import sacrebleu
    except ImportError:
        return None
    n = min(len(hyps), len(srcs), len(refs))
    if n == 0:
        return None
    chrf = sacrebleu.corpus_chrf(hyps[:n], [refs[:n]], word_order=2).score
    bleu = sacrebleu.corpus_bleu(hyps[:n], [refs[:n]]).score
    return {"bleu": bleu, "chrf": chrf, "ssa_comet": None, "num_samples": n}


def _score_doc(samples):
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
    return {"bleu": bleu, "chrf": chrf, "ssa_comet": None,
            "num_samples": len(samples)}


def _read_test_col(repo: Path, lang: str) -> list[str]:
    import csv
    with open(repo / "data" / "processed_files" / "test" / f"{lang}.tsv") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--short-name", default=None)
    ap.add_argument("--pairs", nargs="+", default=None)
    ap.add_argument("--direction", choices=["e2x", "x2e", "both"], default="both")
    ap.add_argument("--only", choices=["sent", "doc", "both"], default="both",
                    help="Restrict to sent or doc vanilla; default both.")
    ap.add_argument("--sent-shots", type=int, nargs="+", default=SENT_SHOTS)
    ap.add_argument("--doc-shots", type=int, nargs="+", default=DOC_SHOTS)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--use-batch", action="store_true")
    ap.add_argument("--batch-poll-seconds", type=int, default=120)
    ap.add_argument("--batch-concurrency", type=int, default=3)
    ap.add_argument("--batch-state-file",
                    default="experiments/_openai_vanilla_batch_state.json")
    ap.add_argument("--notify-email", default="afriannotate@gmail.com")
    ap.add_argument("--notify-from", default=os.environ.get("BREVO_SENDER_EMAIL"))
    ap.add_argument("--brevo-api-key", default=os.environ.get("BREVO_API_KEY"))
    ap.add_argument("--notify-every", type=int, default=10)
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("error: pass --api-key or set OPENAI_API_KEY", file=sys.stderr)
        return 2

    short_name = args.short_name or sanitise_short(args.model)
    pair_filter = set(args.pairs) if args.pairs else None
    pairs = []
    for s, t in PAIRS:
        if pair_filter and f"{s}-{t}" not in pair_filter:
            continue
        if args.direction == "e2x" and s != "eng": continue
        if args.direction == "x2e" and t != "eng": continue
        pairs.append((s, t))

    # Build the config queue: (kind, n_shots, src, tgt)
    configs = []
    if args.only in ("sent", "both"):
        for k in args.sent_shots:
            for s, t in pairs:
                configs.append(("sent", k, s, t))
    if args.only in ("doc", "both"):
        for k in args.doc_shots:
            for s, t in pairs:
                configs.append(("doc", k, s, t))

    # Skip existing
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
                skipped += 1
                continue
            kept.append((kind, k, s, t))
        print(f"  --skip-existing: skipping {skipped} already-completed configs.")
        configs = kept

    if not configs:
        print("nothing to do.")
        return 0

    print(f"OpenAI vanilla rerun: model={args.model} short={short_name} "
          f"T={args.temperature} batch={args.use_batch} configs={len(configs)}")

    from afriscience_mt.api.factory import create_api_client
    api_client = create_api_client(
        provider="openai", model_name=args.model, api_key=api_key,
        temperature=args.temperature, max_tokens=args.max_tokens,
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
            f"Started: {args.model} (rerun_openai_vanilla)",
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
            prompts = sent_builder.build_prompts(src, tgt, k)
            srcs = _read_test_col(ROOT, src)
            refs = _read_test_col(ROOT, tgt)
            return {"prompts": prompts, "srcs": srcs, "refs": refs}
        # doc
        papers = doc_builder.list_test_papers(tgt)
        inputs = []
        for pid, domain in papers:
            src_doc = doc_builder.test_doc(src, pid)
            ref_doc = doc_builder.test_doc(tgt, pid)
            prompt = doc_builder.build_prompt(
                src=src, tgt=tgt, paper_id=pid,
                src_document=src_doc, n_shots=k,
            )
            inputs.append({"paper_id": pid, "domain": domain,
                           "src": src_doc, "ref": ref_doc, "prompt": prompt})
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
                job_label = f"{short_name}/vanilla/{kind}/{pair}/{strat}"
                payload = _build_inputs(kind, k, src, tgt)
                user_prompts = (payload["prompts"] if kind == "sent"
                                else [r["prompt"] for r in payload["inputs"]])
                print(f"\n[submit] {label}  ({len(user_prompts)} prompts)")
                existing = state.get(job_label)
                if existing:
                    job_id = existing
                    print(f"  resuming: {job_id}")
                else:
                    job_id = api_client.submit_batch_job(
                        prompts=user_prompts, job_name=job_label, system_prompt=None,
                    )
                    state[job_label] = job_id
                    state_file.parent.mkdir(parents=True, exist_ok=True)
                    state_file.write_text(json.dumps(state, indent=2))
                in_flight[job_label] = {
                    "cfg": (kind, k, src, tgt),
                    "payload": payload, "label": label, "job_id": job_id,
                    "started_at": time.time(),
                }

        fill_window()
        while in_flight or queue:
            done_keys = []
            for jl, info in list(in_flight.items()):
                try:
                    hyps = api_client.retrieve_batch_results(info["job_id"])
                except Exception as e:
                    print(f"  x {info['label']} retrieve failed: {e}")
                    continue
                if hyps is None:
                    continue
                kind, k, src, tgt = info["cfg"]
                hyps = [(h or "").strip() for h in hyps]
                print(f"\n[done] {info['label']}  ({len(hyps)} responses)")
                state.pop(jl, None)
                state_file.write_text(json.dumps(state, indent=2))
                done_count += 1
                _save(kind, k, src, tgt, hyps, info["started_at"],
                      done_count, info["label"], info["payload"])
                done_keys.append(jl)
            for jl in done_keys: del in_flight[jl]
            fill_window()
            if in_flight: time.sleep(args.batch_poll_seconds)
    else:
        for idx, (kind, k, src, tgt) in enumerate(configs, 1):
            pair = f"{src}-{tgt}"
            strat = f"{k}shot" if kind == "sent" else f"doc_{k}shot"
            label = f"{kind}/{strat}/{pair}"
            payload = _build_inputs(kind, k, src, tgt)
            user_prompts = (payload["prompts"] if kind == "sent"
                            else [r["prompt"] for r in payload["inputs"]])
            cp = cache_path_for(_out_dir(kind, k, src, tgt))
            cached = load_cache(cp)
            print(f"\n[{idx}/{len(configs)}] {label}  "
                  f"({len(user_prompts)} prompts, {len(cached)} cached)")
            ts = time.time()
            hyps = [None] * len(user_prompts)
            for i, p in enumerate(tqdm(user_prompts, desc=strat, leave=False)):
                if i in cached:
                    hyps[i] = cached[i]; continue
                try:
                    text = (api_client._call_api(
                        [{"role": "user", "content": p}]
                    ) or "").strip()
                except Exception as e:
                    print(f"  ! row {i} failed: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    text = ""
                hyps[i] = text
                append_cache(cp, i, text)
            _save(kind, k, src, tgt, hyps, ts, idx, label, payload)

    notifier.send(
        f"Run complete: {args.model} (vanilla)",
        f"All {len(configs)} vanilla configs done.\n"
        f"Sent -> experiments/llm/closed/{temp_setting(args.temperature)}/icl/{short_name}/\n"
        f"Doc  -> experiments/llm/closed/doc/{short_name}/",
    )
    print(f"\nTotal: {time.time() - t0:.1f}s")
    print("Aggregate + rescore + rebuild as usual.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
