"""Rescore SSA-COMET *per-domain* on selected experiment dirs and write the
breakdown into ``results.json.metrics.per_domain``. Parallel across N GPUs.

For each prediction set the script:
  1. Looks up the test split's ``category`` per row from
     data/processed_files/test/<src>.tsv.
  2. Groups (src, ref, hyp) triples by category.
  3. Runs the SSA-COMET MetricsCalculator once per category.
  4. Writes the breakdown into results.json under metrics.per_domain.<cat>.

Idempotent: skips dirs that already have ``per_domain.<cat>.ssa_comet``
unless ``--force`` is passed.

Parallel execution.  ``--gpus`` selects the GPU IDs to use; one worker
per GPU loads the COMET model once and consumes tasks from a shared
queue. A tqdm progress bar tracks completion across workers.

Usage (from repo root, env with torch + comet):

    # Auto: one worker per visible GPU, list with `nvidia-smi -L | wc -l`
    python scripts/rescore_per_domain_ssa.py \\
        --src experiments/seq2seq/finetune/nllb_200_1_3b/dcs \\
        --src experiments/llm/open/finetune/gemma2_9b_it/r64 \\
        --src experiments/llm/open/finetune/tiny_aya_earth/r64 \\
        --src experiments/llm/closed/t06/icl/gpt4o/template3_5shot \\
        --src experiments/llm/closed/t0/icl/gpt4o/template3_5shot

    # Avoid GPU 0 (busy with another job) — use only GPU 1:
    python scripts/rescore_per_domain_ssa.py --gpus 1  --src ...

    # 2 workers per GPU (if memory allows; default = 1):
    python scripts/rescore_per_domain_ssa.py --gpus 0 1 --workers-per-gpu 2 --src ...
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "processed_files" / "test"


# ---------------- shared helpers ----------------

def load_test(lang: str) -> list[dict]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


_REQUIRED_PER_DOMAIN_KEYS = ("ssa_comet", "bleu", "chrf")


def has_per_domain_ssa(results_path: Path) -> bool:
    """True iff every per_domain.<cat> entry has ssa_comet, bleu, AND chrf.

    A partial result (some cats/metrics missing) returns False so the dir
    is re-queued and the missing values get filled in.
    """
    try:
        d = json.loads(results_path.read_text())
    except Exception:
        return False
    pd = (d.get("metrics") or {}).get("per_domain") or {}
    if not pd:
        return False
    return all(
        all((stats or {}).get(k) is not None for k in _REQUIRED_PER_DOMAIN_KEYS)
        for stats in pd.values()
    )


def collect_targets(srcs: list[str], force: bool) -> tuple[list[Path], int, int]:
    """Returns (todo, n_already_done, n_no_predictions)."""
    seen: set[Path] = set()
    todo: list[Path] = []
    n_already, n_no_pred = 0, 0
    for src in srcs:
        for r in Path(src).rglob("results.json"):
            pred = r.parent / "predictions.json"
            if not pred.exists():
                n_no_pred += 1
                continue
            if pred in seen:
                continue
            seen.add(pred)
            if not force and has_per_domain_ssa(r):
                n_already += 1
                continue
            todo.append(pred)
    return todo, n_already, n_no_pred


# ---------------- per-task work (runs in a worker process) ----------------

def rescore_one(pred_path: Path, calc, force: bool = False) -> dict[str, dict] | None:
    """Compute per-domain metrics, only filling in what's missing.

    Reads the existing ``results.json.metrics.per_domain`` block and skips
    metrics that are already populated per category. Pass ``force=True``
    to always recompute every metric.
    """
    # Lazy import so this module is import-safe before the worker pins a GPU.
    from afriscience_mt.evaluation.metrics import compute_bleu, compute_chrf

    payload = json.loads(pred_path.read_text())
    src_lang = payload["source_lang"]
    tgt_lang = payload["target_lang"]
    samples = payload.get("samples") or []

    src_tsv = load_test(src_lang)
    tgt_tsv = load_test(tgt_lang)
    if len(samples) != len(src_tsv):
        return None

    # Existing per-domain state, so we know what's already done.
    res_path = pred_path.parent / "results.json"
    try:
        existing = (json.loads(res_path.read_text()).get("metrics") or {}).get("per_domain") or {}
    except Exception:
        existing = {}

    by_cat = defaultdict(lambda: {"src": [], "ref": [], "hyp": []})
    for i, s in enumerate(samples):
        hyp = s.get("hyp") or ""
        if not hyp:
            continue
        cat = src_tsv[i]["category"]
        by_cat[cat]["src"].append(src_tsv[i]["text"])
        by_cat[cat]["ref"].append(tgt_tsv[i]["text"])
        by_cat[cat]["hyp"].append(hyp)

    out: dict[str, dict] = {}
    for cat, bag in by_cat.items():
        prev = existing.get(cat) or {}
        entry: dict = {"n": len(bag["hyp"])}

        bleu_v = prev.get("bleu")
        if force or bleu_v is None:
            bleu_v = compute_bleu(bag["hyp"], bag["ref"])["bleu"]
        entry["bleu"] = bleu_v

        chrf_v = prev.get("chrf")
        if force or chrf_v is None:
            chrf_v = compute_chrf(bag["hyp"], bag["ref"])["chrf"]
        entry["chrf"] = chrf_v

        ssa_v = prev.get("ssa_comet")
        if force or ssa_v is None:
            ssa_v = calc.compute(hypotheses=bag["hyp"],
                                 references=bag["ref"],
                                 sources=bag["src"]).get("ssa_comet")
        entry["ssa_comet"] = ssa_v

        out[cat] = entry
    return out


def write_per_domain(results_path: Path, per_dom: dict[str, dict]) -> None:
    d = json.loads(results_path.read_text())
    d.setdefault("metrics", {}).setdefault("per_domain", {})
    for cat, stats in per_dom.items():
        d["metrics"]["per_domain"].setdefault(cat, {})
        d["metrics"]["per_domain"][cat].update(stats)
    d.setdefault("metadata", {}).setdefault("rescored", []).append({
        "metric": "per_domain_bleu_chrf_ssa_comet",
        "n_domains": len(per_dom),
        "timestamp": datetime.utcnow().isoformat(),
    })
    results_path.write_text(json.dumps(d, indent=2, ensure_ascii=False))


def worker_loop(gpu_id: int, worker_id: int,
                task_q: mp.Queue, done_q: mp.Queue,
                force: bool = False) -> None:
    """Pinned-to-one-GPU worker. Loads COMET once and drains the queue."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Import LATE so CUDA_VISIBLE_DEVICES is honoured.
    from afriscience_mt.evaluation.metrics import MetricsCalculator

    try:
        calc = MetricsCalculator(metrics=["ssa_comet"])
    except Exception as e:
        done_q.put({"type": "init_failed", "worker": worker_id,
                    "gpu": gpu_id, "err": f"{e}\n{traceback.format_exc()}"})
        return
    done_q.put({"type": "ready", "worker": worker_id, "gpu": gpu_id})

    while True:
        item = task_q.get()
        if item is None:                # sentinel: drain & exit
            break
        pred_path = Path(item)
        try:
            per_dom = rescore_one(pred_path, calc, force=force)
        except Exception as e:
            done_q.put({"type": "error", "worker": worker_id,
                        "path": str(pred_path), "err": str(e)})
            continue
        if not per_dom:
            done_q.put({"type": "skip", "worker": worker_id,
                        "path": str(pred_path),
                        "reason": "row count mismatch"})
            continue
        try:
            write_per_domain(pred_path.parent / "results.json", per_dom)
        except Exception as e:
            done_q.put({"type": "write_error", "worker": worker_id,
                        "path": str(pred_path), "err": str(e)})
            continue
        done_q.put({"type": "done", "worker": worker_id,
                    "path": str(pred_path), "n_domains": len(per_dom)})


# ---------------- main ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="dir to walk for predictions.json (repeatable)")
    ap.add_argument("--gpus", type=int, nargs="+", default=None,
                    help="GPU IDs to use (default: detect via "
                         "CUDA_VISIBLE_DEVICES or fall back to 0)")
    ap.add_argument("--workers-per-gpu", type=int, default=1,
                    help="workers per GPU (default 1; bump if VRAM allows)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    targets, n_already, n_no_pred = collect_targets(args.src, args.force)
    print(f"scan: {len(targets)} to rescore, "
          f"{n_already} already complete (skipped), "
          f"{n_no_pred} without predictions.json")
    if not targets:
        print("nothing to do; all dirs already have per_domain.ssa_comet "
              "for every category. (Use --force to recompute.)")
        return 0

    if args.gpus is None:
        env = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpus = [int(x) for x in env.split(",")] if env else [0]
    else:
        gpus = list(args.gpus)
    workers_per_gpu = max(1, args.workers_per_gpu)
    worker_specs = [g for g in gpus for _ in range(workers_per_gpu)]
    n_workers = len(worker_specs)
    print(f"using {n_workers} worker(s) across GPUs {gpus} "
          f"({workers_per_gpu}/GPU)")

    ctx = mp.get_context("spawn")  # safer than fork when CUDA is involved
    task_q: mp.Queue = ctx.Queue()
    done_q: mp.Queue = ctx.Queue()

    procs = []
    for wi, gpu in enumerate(worker_specs):
        p = ctx.Process(target=worker_loop,
                        args=(gpu, wi, task_q, done_q, args.force),
                        daemon=False)
        p.start()
        procs.append(p)

    # Wait for all workers to signal "ready" before queueing tasks (so the
    # progress bar starts only after model load).
    ready = 0
    while ready < n_workers:
        msg = done_q.get()
        if msg["type"] == "ready":
            ready += 1
            print(f"  ready: worker {msg['worker']} on GPU {msg['gpu']} "
                  f"({ready}/{n_workers})")
        elif msg["type"] == "init_failed":
            print(f"  worker {msg['worker']} (GPU {msg['gpu']}) FAILED to "
                  f"init COMET:\n{msg['err']}")
            for _ in range(n_workers):
                task_q.put(None)
            for p in procs:
                p.join()
            return 1

    # Enqueue tasks + sentinels
    for t in targets:
        task_q.put(str(t))
    for _ in range(n_workers):
        task_q.put(None)

    # Drain done events with a tqdm progress bar.
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    n_done = n_skip = n_err = 0
    pbar = tqdm(total=len(targets), unit="dir") if tqdm else None
    while n_done + n_skip + n_err < len(targets):
        msg = done_q.get()
        if msg["type"] == "done":
            n_done += 1
        elif msg["type"] == "skip":
            n_skip += 1
            (tqdm.write if tqdm else print)(
                f"  skip {msg['path']}: {msg['reason']}")
        elif msg["type"] in {"error", "write_error"}:
            n_err += 1
            (tqdm.write if tqdm else print)(
                f"  err  {msg['path']}: {msg['err']}")
        if pbar:
            pbar.update(1)
            pbar.set_postfix(done=n_done, skip=n_skip, err=n_err)
    if pbar:
        pbar.close()

    for p in procs:
        p.join()

    print(f"\ndone: {n_done} rescored, {n_skip} skipped, {n_err} errored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
