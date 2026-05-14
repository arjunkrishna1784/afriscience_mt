"""Batched version of scripts/run_doc_llm_judge.py.

Submits every (cell x paper x rubric) call as one big OpenAI/Gemini Batch
job per judge model (50% cheaper than direct API; up to 24h SLA but
usually much faster). Resumable across restarts: persists the batch job
ID + manifest in a state file so a re-run skips the submit step and
just polls.

Reuses the rubric prompts, JSON parsers, and aggregation from the
non-batched twin so output schema is byte-identical.

Usage:
    # GPT-5.5 judge over the new closed-API doc cells:
    python scripts/run_doc_llm_judge_batch.py \\
        --src experiments/llm/closed/doc/gpt54 \\
        --src experiments/llm/closed/doc/gemini_31_flash_lite \\
        --judge-model gpt-5.5 --out-name judge

    # Gemini-3.1-flash-lite judge (writes judge_gemini.json):
    python scripts/run_doc_llm_judge_batch.py \\
        --src experiments/llm/closed/doc/gpt54 \\
        --src experiments/llm/closed/doc/gemini_31_flash_lite \\
        --judge-model gemini-3.1-flash-lite --out-name judge_gemini

    # Run both judges in parallel via the launcher:
    bash scripts/run_all_doc_judge_batch.sh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from afriscience_mt.doc_prompts import LANG_DISPLAY  # noqa: E402
from afriscience_mt.utils.notifier import (  # noqa: E402
    add_notify_args, notifier_from_args,
)

# Reuse rubric templates, parsers, aggregator, and IO helpers from the
# non-batched script so output schema stays in sync.
from run_doc_llm_judge import (  # type: ignore  # noqa: E402
    _FLUENCY_TMPL, _ACCURACY_TMPL, _COHESION_TMPL,
    _parse_fluency, _parse_accuracy, _parse_cohesion,
    _aggregate, _iter_cells, _judge_path, _load_existing, _save,
)


RUBRICS = ("fluency", "accuracy", "cohesion")
_TMPLS = {
    "fluency": _FLUENCY_TMPL,
    "accuracy": _ACCURACY_TMPL,
    "cohesion": _COHESION_TMPL,
}
_PARSERS = {
    "fluency": _parse_fluency,
    "accuracy": _parse_accuracy,
    "cohesion": _parse_cohesion,
}


def _build_prompt(rubric: str, sample: dict, tgt_lang_display: str) -> str:
    tmpl = _TMPLS[rubric]
    if rubric == "fluency":
        return tmpl.format(tgt_lang=tgt_lang_display, hyp=sample.get("hyp", ""))
    return tmpl.format(
        tgt_lang=tgt_lang_display,
        ref=sample.get("ref", ""),
        hyp=sample.get("hyp", ""),
    )


def _cell_complete(cell: Path, out_name: str) -> bool:
    pred_path = cell / "predictions.json"
    judge = _load_existing(cell, out_name)
    if not judge:
        return False
    try:
        n_expected = len(json.loads(pred_path.read_text())["samples"])
    except Exception:
        return False
    return judge.get("aggregate", {}).get("n_papers_scored") == n_expected


def _build_manifest(
    cells: list[Path], out_name: str, force: bool,
) -> tuple[list[str], list[dict]]:
    """For every cell + paper not already judged, build (prompt, manifest_entry).

    Returns (prompts, manifest) parallel lists.
    """
    prompts: list[str] = []
    manifest: list[dict] = []
    for cell in cells:
        pred = json.loads((cell / "predictions.json").read_text())
        existing = _load_existing(cell, out_name) or {}
        existing_pids = (
            {p["paper_id"] for p in existing.get("papers", [])}
            if not force else set()
        )
        tgt_short = pred.get("target_lang", "")
        tgt_display = LANG_DISPLAY.get(tgt_short, tgt_short)
        for s in pred.get("samples", []):
            pid = s.get("paper_id", "")
            if pid in existing_pids:
                continue
            for rubric in RUBRICS:
                prompts.append(_build_prompt(rubric, s, tgt_display))
                manifest.append({
                    "cell": str(cell.resolve()),
                    "pid": pid,
                    "domain": s.get("domain", ""),
                    "src": pred.get("source_lang", ""),
                    "tgt": tgt_short,
                    "tgt_display": tgt_display,
                    "rubric": rubric,
                    "experiment_id": pred.get("experiment_id", ""),
                })
    return prompts, manifest


def _fan_out_and_save(
    *, manifest: list[dict], results: list[str],
    judge_model: str, out_name: str,
) -> int:
    """Group results by cell, merge with any existing per-paper judgments,
    and write a complete <out_name>.json per cell. Returns cells written."""
    if len(manifest) != len(results):
        raise RuntimeError(
            f"manifest/results length mismatch: "
            f"{len(manifest)} vs {len(results)}"
        )

    # cell_path -> {pid: {rubric: raw}}
    by_cell: dict[str, dict[str, dict[str, str]]] = defaultdict(
        lambda: defaultdict(dict))
    cell_meta: dict[str, dict] = {}
    for entry, raw in zip(manifest, results):
        cp = entry["cell"]
        by_cell[cp][entry["pid"]][entry["rubric"]] = raw or ""
        cell_meta.setdefault(cp, entry)

    n_written = 0
    for cell_str, by_pid in by_cell.items():
        cell = Path(cell_str)
        try:
            pred = json.loads((cell / "predictions.json").read_text())
        except Exception as e:
            print(f"  ! skip {cell}: cannot read predictions ({e})",
                  file=sys.stderr)
            continue
        existing = _load_existing(cell, out_name) or {}
        existing_by_pid = {p["paper_id"]: p for p in existing.get("papers", [])}

        papers_out: list[dict] = []
        for s in pred.get("samples", []):
            pid = s.get("paper_id", "")
            if pid in by_pid:
                rubrics_raw = by_pid[pid]
                papers_out.append({
                    "paper_id": pid,
                    "domain": s.get("domain", ""),
                    "fluency": _parse_fluency(rubrics_raw.get("fluency", "")),
                    "accuracy": _parse_accuracy(rubrics_raw.get("accuracy", "")),
                    "cohesion": _parse_cohesion(rubrics_raw.get("cohesion", "")),
                })
            elif pid in existing_by_pid:
                papers_out.append(existing_by_pid[pid])

        agg = _aggregate(papers_out)
        meta = cell_meta[cell_str]
        payload = {
            "schema_version": "1.0",
            "judge_model": judge_model,
            "source_lang": meta["src"],
            "target_lang": meta["tgt"],
            "prompt_strategy": (
                meta["experiment_id"].split("__")[-2]
                if "__" in meta["experiment_id"] else ""
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "papers": papers_out,
            "aggregate": agg,
        }
        _save(cell, payload, out_name)
        n_written += 1
    return n_written


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batched LLM-judge for doc-level closed-API runs.")
    ap.add_argument("--src", action="append", required=True,
                    help="Directory to walk for predictions.json. Repeatable.")
    ap.add_argument("--judge-model", default="gpt-5.5",
                    help="Judge model id. Names starting with 'gemini-' "
                         "route to Google batch; everything else to OpenAI "
                         "batch. Default gpt-5.5.")
    ap.add_argument("--out-name", default="judge",
                    help="Filename stem for the saved judgments "
                         "(default 'judge' -> 'judge.json').")
    ap.add_argument("--api-key", default=None,
                    help="API key (OpenAI or Google depending on "
                         "--judge-model). Defaults to the corresponding "
                         "env var.")
    ap.add_argument("--force", action="store_true",
                    help="Re-judge cells even if <out_name>.json is "
                         "already complete.")
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-state-file", default=None,
                    help="Where to persist the in-flight batch ID + "
                         "manifest. Defaults to "
                         "experiments/_judge_<short>_batch_state.json")
    ap.add_argument("--batch-poll-seconds", type=float, default=60.0,
                    help="Seconds between batch status polls.")
    ap.add_argument("--no-submit", action="store_true",
                    help="Skip the submit step; only poll an existing "
                         "batch in the state file.")
    add_notify_args(ap)
    args = ap.parse_args()

    # Provider routing.
    if args.judge_model.lower().startswith("gemini-"):
        provider = "google"
        api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("error: pass --api-key or set GOOGLE_API_KEY",
                  file=sys.stderr)
            return 2
    else:
        provider = "openai"
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("error: pass --api-key or set OPENAI_API_KEY",
                  file=sys.stderr)
            return 2

    src_roots = [Path(s).expanduser().resolve() for s in args.src]
    cells = _iter_cells(src_roots)
    if not cells:
        print("no predictions.json files found under any --src",
              file=sys.stderr)
        return 2

    # Drop fully judged cells unless --force.
    if not args.force:
        before = len(cells)
        cells = [c for c in cells if not _cell_complete(c, args.out_name)]
        print(f"  skipping {before - len(cells)} cells already judged "
              f"(use --force to redo). Remaining: {len(cells)}")

    state_default = (
        ROOT / "experiments"
        / f"_judge_{args.judge_model.replace('/', '_').replace('.', '')}_batch_state.json"
    )
    state_file = Path(args.batch_state_file) if args.batch_state_file else state_default
    state = json.loads(state_file.read_text()) if state_file.exists() else {}

    # Build API client lazily — we only need it for submit/retrieve.
    from afriscience_mt.api.factory import create_api_client  # noqa: E402
    judge = create_api_client(
        provider=provider,
        model_name=args.judge_model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
    )

    notifier = notifier_from_args(args)
    if notifier.enabled:
        print(f"notify: enabled -> {args.notify_email}")
    else:
        print("notify: disabled")

    # ---- Submit phase --------------------------------------------------
    if "batch" not in state and not args.no_submit:
        if not cells:
            print("nothing to do.")
            return 0
        print(f"building manifest for {len(cells)} cells...")
        prompts, manifest = _build_manifest(cells, args.out_name, args.force)
        print(f"  -> {len(prompts)} prompts "
              f"({len(manifest) // 3} papers x 3 rubrics)")
        if not prompts:
            print("nothing to do (all papers already judged).")
            return 0

        if len(prompts) > 50_000:
            print(f"error: {len(prompts)} prompts exceeds OpenAI batch "
                  f"limit (50,000). Narrow --src or run in chunks.",
                  file=sys.stderr)
            return 2

        job_name = f"judge_{args.judge_model}_{args.out_name}"
        print(f"submitting batch ({provider})...")
        try:
            job_id = judge.submit_batch_job(
                prompts=prompts, job_name=job_name, system_prompt=None,
            )
        except Exception as e:
            notifier.send(
                f"Judge batch SUBMIT FAILED ({args.judge_model})",
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            )
            raise
        print(f"  submitted: {job_id}")

        state = {
            "judge_model": args.judge_model,
            "out_name": args.out_name,
            "provider": provider,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "batch": {
                "job_id": job_id,
                "n_prompts": len(prompts),
                "manifest": manifest,
            },
        }
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        notifier.send(
            f"Judge batch submitted ({args.judge_model})",
            f"job_id: {job_id}\nprompts: {len(prompts)}\n"
            f"papers: {len(manifest)//3}\ncells: {len(cells)}",
        )
    elif "batch" not in state and args.no_submit:
        print("--no-submit set and no in-flight batch in state. Nothing to do.")
        return 0
    else:
        b = state["batch"]
        print(f"resuming: job_id={b['job_id']}  n_prompts={b['n_prompts']}")

    # ---- Poll phase ----------------------------------------------------
    job_id = state["batch"]["job_id"]
    manifest = state["batch"]["manifest"]
    print(f"polling every {args.batch_poll_seconds}s...")
    t0 = time.time()
    last_progress_t = t0
    while True:
        try:
            results = judge.retrieve_batch_results(job_id)
        except Exception as e:
            notifier.send(
                f"Judge batch RETRIEVE FAILED ({args.judge_model})",
                f"job_id: {job_id}\n{type(e).__name__}: {e}\n\n"
                f"{traceback.format_exc()}",
            )
            raise
        if results is not None:
            break
        # Periodic heartbeat (every hour) so the user knows we're alive.
        if time.time() - last_progress_t > 3600:
            notifier.send(
                f"Judge batch still running ({args.judge_model})",
                f"job_id: {job_id}\nelapsed: {time.time()-t0:.0f}s",
            )
            last_progress_t = time.time()
        time.sleep(args.batch_poll_seconds)

    print(f"batch complete: {len(results)} responses in "
          f"{time.time()-t0:.0f}s of polling")

    # ---- Fan out + save ------------------------------------------------
    n_cells = _fan_out_and_save(
        manifest=manifest, results=results,
        judge_model=args.judge_model, out_name=args.out_name,
    )
    print(f"wrote {args.out_name}.json for {n_cells} cells")

    # Clear state so re-running the script is a no-op.
    state.pop("batch", None)
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False))

    notifier.send(
        f"Judge batch done ({args.judge_model})",
        f"job_id: {job_id}\ncells written: {n_cells}\n"
        f"papers judged: {len(manifest)//3}",
    )
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
