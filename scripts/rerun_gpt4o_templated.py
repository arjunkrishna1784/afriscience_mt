"""Rerun GPT-4o (or any newer OpenAI model) using the EXACT prompts that
were sent to the legacy gpt-4o-temp-0.0 runs, read from the legacy TSV
dump's `prompt` column. No prompt rebuilding — just resend.

This guarantees byte-for-byte prompt parity with the legacy gpt-4o
predictions, so the new outputs support a clean "model version"
ablation analogous to the Gemini rerun.

The closed-API audit (2026-05-08) found one structural gap: GPT-4o
``eng->xxx`` at T=0 only has the semantic-similarity strategies; the
templated runs (templates 1..5) live exclusively at T=0.6. Re-running
with the *same prompts* against a newer model fills that gap (and
also gives us a prompt-identical comparison to the original GPT-4o).

Source of prompts (preferred -> fallback):
    /ext_data/idris/DCS-MT-Updated/old_experiments/LLMS/gpt-4o_temp_<T>/
    /home/idris/ext_data/DCS-MT-Old/LLMS/translations/gpt-4o_temp_<T>/
    /Users/MacbookPro/Documents/GitHub/DCS-MT/LLMS/translations/gpt-4o_temp_<T>/

TSV format:  src_text \\t prompt \\t translation
The `prompt` column is in LangChain debug format
("System: <sys>\\nHuman: <user>"); we parse that and send proper
system + user messages.

Sweep: every TSV in the legacy dump for the chosen temperature.
50-shot files are skipped per project policy (docs/PAPER_TODOS.md).

Outputs land at:
    experiments/llm/closed/<temp>/icl/<short_name>/template{N}_{K}shot/<pair>/
        predictions.json + results.json

Usage:
    export OPENAI_API_KEY=...
    python scripts/rerun_gpt4o_templated.py --model gpt-4o
    python scripts/rerun_gpt4o_templated.py --model gpt-5 --temperature 0.6
    python scripts/rerun_gpt4o_templated.py --model gpt-4o --use-batch
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Make ``afriscience_mt`` importable when the script is launched from a
# wrapper that doesn't set PYTHONPATH (e.g. ``bash scripts/run_rerun_gpt54.sh``).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


# Brevo email notifier — re-exported here so existing
# `from rerun_gpt4o_templated import Notifier` callers keep working.
from afriscience_mt.utils.notifier import Notifier  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "data" / "processed_files" / "test"
EXP_ROOT = ROOT / "experiments" / "llm" / "closed"

# Preferred source first; fall back to local checkout / DCS-MT-Old.
LEGACY_DUMP_CANDIDATES = [
    Path("/ext_data/idris/DCS-MT-Updated/old_experiments/LLMS"),
    Path("/home/idris/ext_data/DCS-MT-Old/LLMS/translations"),
    Path("/Users/MacbookPro/Documents/GitHub/DCS-MT/LLMS/translations"),
]

LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
PAIRS = [f"eng-{l}" for l in LANGS] + [f"{l}-eng" for l in LANGS]
EXPECTED_N = 970


def find_legacy_dump(temp: float, override: Path | None = None) -> Path:
    sub = f"gpt-4o_temp_{temp:.1f}"
    candidates = [override] if override else LEGACY_DUMP_CANDIDATES
    tried = []
    for root in candidates:
        if root is None:
            continue
        # Accept either <root>/gpt-4o_temp_<T> or a direct path to the dir.
        for d in (root / sub, root):
            if d.is_dir() and any(d.glob("*-template*.tsv")):
                return d
            tried.append(str(d))
    raise SystemExit(f"no legacy dump found. Tried: {tried}")


def list_legacy_configs(legacy_dir: Path):
    """Yield (template, shots, src, tgt, tsv_path) from the dump.
    Handles four filename conventions seen across legacy GPT-4o dumps:
      <src>-to-<tgt>-template<N>-shots<K>.tsv   (T=0 xxx->eng layout)
      <src>-to-<tgt>-template<N>-shot-<K>.tsv   (T=0.6 eng->xxx layout)
      <src>-to-<tgt>-template<N>.tsv            (T=0.6 0-shot variants)
      <src>-to-<tgt>-semantic-shots<K>.tsv      (semantic-similarity exemplars)

    For semantic files we yield template=0 as a sentinel; ``write_canonical``
    interprets template=0 as the ``semantic_<K>shot`` strategy.
    Skips 50-shot and any *-old.tsv files per project policy.
    """
    rx_tmpl = re.compile(
        r"^(?P<src>[a-z]+)-to-(?P<tgt>[a-z]+)-template(?P<n>\d+)"
        r"(?:-shots?-?(?P<k>\d+))?(?P<old>(-old)?)\.tsv$"
    )
    rx_sem = re.compile(
        r"^(?P<src>[a-z]+)-to-(?P<tgt>[a-z]+)-semantic"
        r"(?:-shots?-?(?P<k>\d+))?(?P<old>(-old)?)\.tsv$"
    )
    out = []
    for tsv in sorted(legacy_dir.glob("*.tsv")):
        m = rx_tmpl.match(tsv.name)
        if m:
            if m.group("old"):
                continue
            k = int(m.group("k")) if m.group("k") else 0
            if k > 20:
                continue
            # Skip bare `template5.tsv` (no shot suffix): template 5 is the
            # few-shot template and never legitimately 0-shot. The legacy
            # gpt-4o T=0.6 dump has this file with 1 example embedded —
            # parses as 0-shot but content is 1-shot, and there is no
            # 1-shot sentence-level condition in the paper grid.
            if int(m.group("n")) == 5 and not m.group("k"):
                continue
            out.append((int(m.group("n")), k,
                        m.group("src"), m.group("tgt"), tsv))
            continue
        m = rx_sem.match(tsv.name)
        if m:
            if m.group("old"):
                continue
            k = int(m.group("k")) if m.group("k") else 0
            if k > 20:
                continue
            out.append((0, k, m.group("src"), m.group("tgt"), tsv))

    # The legacy semantic TSVs do not live in the *_temp_<T>/ dir; they sit
    # in a sibling LLMS/semantic-prompts/<model>/ folder with the naming
    # <src>-<tgt>_<K>-shots.tsv (no "-to-", underscore separator). Scan it
    # for T=0 dumps so semantic_<K>shot configs are picked up.
    if "_temp_0.0" in legacy_dir.name:
        model = legacy_dir.name.split("_temp_")[0]
        sem_dir = legacy_dir.parent / "semantic-prompts" / model
        rx_sem_actual = re.compile(
            r"^(?P<src>[a-z]+)-(?P<tgt>[a-z]+)_(?P<k>\d+)-shots\.tsv$"
        )
        if sem_dir.is_dir():
            for tsv in sorted(sem_dir.glob("*.tsv")):
                m = rx_sem_actual.match(tsv.name)
                if not m:
                    continue
                k = int(m.group("k"))
                if k > 20:
                    continue
                out.append((0, k, m.group("src"), m.group("tgt"), tsv))
    return out


def sanitise_short(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").lower()


def temp_setting(temp: float) -> str:
    return "t0" if abs(temp) < 1e-9 else f"t{int(round(temp * 10))}".replace("0", "")


def parse_prompt(saved: str) -> tuple[str, str]:
    """LangChain debug format: 'System: <sys>\\nHuman: <user>' -> (sys, user).
    If no Human: marker, return ('', saved) to send as a single user message."""
    if "\nHuman: " not in saved:
        return "", saved
    sys_part, user_part = saved.split("\nHuman: ", 1)
    if sys_part.startswith("System: "):
        sys_part = sys_part[len("System: "):]
    return sys_part, user_part


def read_prompts(tsv: Path) -> list[tuple[str, str]]:
    """Return [(system, user), ...] for one config TSV.

    The legacy DCS-MT semantic dumps store the prompt template WITHOUT the
    source text — the source lives in a separate ``text`` column and was
    injected at API call time as a LangChain Human message. The templated
    dumps embed the source inline (with a literal ``\\nHuman: <src>`` tail).
    For semantic rows, the in-context examples alone were enough for
    GPT-4o to hallucinate plausible translations, but GPT-5.4's reasoning
    mode refuses (~60% empty hyps) when no source is present.

    Fix: when a row's prompt has no ``Human:`` tail and a ``text`` /
    ``src_text`` column is non-empty, append ``\\nHuman: <text>`` so
    parse_prompt routes the source as the user message.
    """
    out = []
    with open(tsv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            p = r.get("prompt") or ""
            if "\nHuman: " not in p:
                src = (r.get("text") or r.get("src_text") or "").strip()
                if src:
                    p = p.rstrip() + "\nHuman: " + src
            out.append(parse_prompt(p))
    return out


def load_test_refs(lang: str) -> list[str]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


def load_test_sources(lang: str) -> list[str]:
    return load_test_refs(lang)


def cache_path_for(out_dir: Path) -> Path:
    return out_dir / "_responses.jsonl"


def load_cache(cache_path: Path) -> dict[int, str]:
    if not cache_path.exists():
        return {}
    done = {}
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                done[int(d["i"])] = d.get("hyp") or ""
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def append_cache(cache_path: Path, idx: int, hyp: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"i": idx, "hyp": hyp}, ensure_ascii=False) + "\n")


def system_is_constant(prompts: list[tuple[str, str]]) -> bool:
    """True iff every row's system part is identical (templates 1/2/3),
    False if it varies per row (templates 4/5 with row-specific glossary)."""
    if not prompts:
        return True
    first = prompts[0][0]
    return all(p[0] == first for p in prompts)


def call_one(api_client, system: str, user: str) -> str:
    """One synchronous API call routed through OpenAIClient._call_api."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return (api_client._call_api(messages) or "").strip()


def split_for_batch(prompts: list[tuple[str, str]]) -> tuple[list[str], str | None]:
    """If every row's system part is identical, return (user_prompts, system).
    If it varies per row (templates 4/5 with row-specific glossary), return
    flattened "System: ...\\nHuman: ..." strings as user_prompts and system=None.
    Either way the prompt content is byte-identical to what legacy gpt-4o saw."""
    if not prompts:
        return [], None
    sys_parts = {sp for sp, _ in prompts}
    if len(sys_parts) == 1 and next(iter(sys_parts)) != "":
        sys_text = next(iter(sys_parts))
        return [u for _, u in prompts], sys_text
    flat = [
        f"System: {sp}\nHuman: {up}" if sp else up
        for sp, up in prompts
    ]
    return flat, None


def write_canonical(*, out_dir: Path, model: str, short_name: str,
                    src: str, tgt: str, template: int, n_shots: int,
                    temp: float, hyps: list[str], system_constant: bool,
                    metrics: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = f"{src}-{tgt}"
    # template == 0 is the sentinel for semantic-similarity exemplars
    # (see list_legacy_configs). All other template numbers come from the
    # numbered prompt-templates.tex inventory.
    if template == 0:
        strategy = f"semantic_{n_shots}shot"
    else:
        strategy = f"template{template}_{n_shots}shot"
    exp_id = f"llm_icl__{short_name}__{strategy}__{pair}"

    refs = load_test_refs(tgt)
    srcs = load_test_sources(src)
    n = min(len(hyps), EXPECTED_N)
    samples = [
        {
            "src": srcs[i] if i < len(srcs) else "",
            "ref": refs[i] if i < len(refs) else "",
            "hyp": hyps[i],
        }
        for i in range(n)
    ]

    (out_dir / "predictions.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "source_lang": src, "target_lang": tgt,
        "samples": samples,
    }, indent=2, ensure_ascii=False))

    res_metrics = metrics or {
        "bleu": None, "chrf": None, "ssa_comet": None, "num_samples": n,
    }

    (out_dir / "results.json").write_text(json.dumps({
        "schema_version": "1.0",
        "experiment_id": exp_id,
        "experiment_type": "llm_icl",
        "timestamp": datetime.utcnow().isoformat(),
        "source_lang": src, "target_lang": tgt, "lang_pair": pair,
        "split": "test", "dataset": "dcs",
        "model": {"name": model, "type": "llm", "short_name": short_name},
        "config": {
            "prompt_strategy": strategy, "n_shots": n_shots,
            "temperature": temp, "temp_setting": temp_setting(temp),
        },
        "metrics": res_metrics,
        "metadata": {
            "source": "rerun_gpt4o_templated",
            "is_ablation": True,
            "prompt_strategy": strategy,
            "n_shots": n_shots,
            "temp_setting": temp_setting(temp),
            "prompt_provenance": "imported_from_legacy_gpt4o_dump",
            "system_constant_per_tsv": system_constant,
            "needs_eval": metrics is None,
        },
    }, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o",
                    help="OpenAI model name (e.g. gpt-4o, gpt-5).")
    ap.add_argument("--short-name", default=None,
                    help="model_short for summary.csv. Defaults to sanitised --model.")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Lang pairs to run (default: every pair found in dump).")
    ap.add_argument("--direction", choices=["e2x", "x2e", "both"], default="both",
                    help="Restrict to eng->xxx (e2x), xxx->eng (x2e), or both. "
                         "Use 'e2x' to fill the missing GPT-4o eng->xxx T=0 gap.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip configs whose results.json already exists. "
                         "Useful when only filling gaps.")
    ap.add_argument("--templates", type=int, nargs="+", default=None,
                    help="Restrict to these templates (default: all in dump).")
    ap.add_argument("--shot-counts", type=int, nargs="+", default=None,
                    help="Restrict to these shot counts (default: all in dump, "
                         "but 50-shot is always skipped).")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Inference temperature for the new model (default 0.0).")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--api-key", default=None,
                    help="OpenAI API key (default: $OPENAI_API_KEY).")
    ap.add_argument("--use-batch", action="store_true",
                    help="Submit prompts via the OpenAI Batch API "
                         "(50%% cheaper, ~24h SLA, no per-minute caps).")
    ap.add_argument("--batch-poll-seconds", type=int, default=120)
    ap.add_argument("--batch-concurrency", type=int, default=5,
                    help="Max in-flight batch jobs at once (--use-batch only).")
    ap.add_argument("--batch-state-file",
                    default="experiments/_openai_batch_state.json")
    # Brevo email notifications.
    ap.add_argument("--notify-email", default="afriannotate@gmail.com",
                    help="Recipient for Brevo notifications.")
    ap.add_argument("--notify-from", default=os.environ.get("BREVO_SENDER_EMAIL"),
                    help="Verified Brevo sender (default: $BREVO_SENDER_EMAIL).")
    ap.add_argument("--brevo-api-key", default=os.environ.get("BREVO_API_KEY"),
                    help="Brevo API key (default: $BREVO_API_KEY).")
    ap.add_argument("--notify-every", type=int, default=25,
                    help="Send a progress email every N completed configs.")
    ap.add_argument("--legacy-temp", type=float, default=0.0,
                    help="Legacy dump temperature to read prompts FROM "
                         "(default 0.0). The new model is run at "
                         "--temperature; these can differ if you want to "
                         "ablate temperature with identical prompts.")
    ap.add_argument("--legacy-dir", default=None,
                    help="Override the legacy dump root. Expects to contain "
                         "a sub-directory named gpt-4o_temp_<T>/ with the "
                         "*-template{N}-shots{K}.tsv files. Defaults to "
                         "the LEGACY_DUMP_CANDIDATES list.")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("error: pass --api-key or set OPENAI_API_KEY", file=sys.stderr)
        return 2

    if args.shot_counts and any(k > 20 for k in args.shot_counts):
        print("error: shot counts > 20 are forbidden (50-shot dropped 2026-05-08).",
              file=sys.stderr)
        return 2

    short_name = args.short_name or sanitise_short(args.model)

    override = Path(args.legacy_dir).expanduser() if args.legacy_dir else None
    legacy_dir = find_legacy_dump(args.legacy_temp, override=override)
    print(f"Reading prompts from: {legacy_dir}")

    configs = list_legacy_configs(legacy_dir)
    if args.pairs:
        bad = [p for p in args.pairs if p not in PAIRS]
        if bad:
            print(f"unknown pairs: {bad}", file=sys.stderr)
            return 2
        configs = [c for c in configs if f"{c[2]}-{c[3]}" in args.pairs]
    if args.direction == "e2x":
        configs = [c for c in configs if c[2] == "eng"]
    elif args.direction == "x2e":
        configs = [c for c in configs if c[3] == "eng"]
    if args.templates:
        configs = [c for c in configs if c[0] in args.templates]
    if args.shot_counts:
        configs = [c for c in configs if c[1] in args.shot_counts]

    out_temp_dir_pre = EXP_ROOT / temp_setting(args.temperature) / "icl" / short_name
    if args.skip_existing:
        kept = []
        skipped = 0
        for tmpl, k, src, tgt, tsv in configs:
            res = (out_temp_dir_pre / f"template{tmpl}_{k}shot" / f"{src}-{tgt}"
                   / "results.json")
            if res.exists():
                skipped += 1
                continue
            kept.append((tmpl, k, src, tgt, tsv))
        print(f"  --skip-existing: skipping {skipped} already-completed configs.")
        configs = kept

    if not configs:
        print("no matching configs to run.", file=sys.stderr)
        return 2

    print(f"OpenAI rerun: model={args.model}  short={short_name}  "
          f"T={args.temperature}  legacy_T={args.legacy_temp}  "
          f"batch={args.use_batch}  configs={len(configs)}")

    from afriscience_mt.api.factory import create_api_client
    api_client = create_api_client(
        provider="openai",
        model_name=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    out_temp_dir = EXP_ROOT / temp_setting(args.temperature) / "icl" / short_name
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
        print(f"notify: enabled → {args.notify_email} every {args.notify_every} configs")
        notifier.send(
            f"Started: {args.model} (rerun_gpt4o_templated)",
            f"Model: {args.model}\nShort: {short_name}\n"
            f"T={args.temperature}  legacy_T={args.legacy_temp}  "
            f"direction={args.direction}\nConfigs: {len(configs)}\n"
            f"Mode: {'batch' if args.use_batch else 'direct'}",
        )

    notify_state = {"i": 0, "total": len(configs), "last_cfg": ""}
    if notifier.enabled:
        import traceback as _tb
        def _excepthook(exc_type, exc_value, exc_tb):
            notifier.send(
                f"FAILED at config {notify_state['i']}/{notify_state['total']}",
                f"Last attempted: {notify_state['last_cfg']}\n"
                f"Error: {exc_type.__name__}: {exc_value}\n\n"
                + "".join(_tb.format_exception(exc_type, exc_value, exc_tb)),
            )
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        sys.excepthook = _excepthook

    def score_and_write(*, src, tgt, hyps, sys_const, tmpl, k, out_dir,
                        ts, idx, label):
        metrics = None
        try:
            from afriscience_mt.evaluation.metrics import compute_all_metrics
            srcs = load_test_sources(src)
            refs = load_test_refs(tgt)
            n = min(len(hyps), len(srcs), len(refs))
            m = compute_all_metrics(
                srcs[:n], hyps[:n], refs[:n],
                metrics=["chrf", "ssa_comet"],
            )
            metrics = {
                "chrf": m.get("chrf"), "ssa_comet": m.get("ssa_comet"),
                "num_samples": n,
            }
            cf, sc = metrics["chrf"], metrics["ssa_comet"]
            print(f"   ChrF={cf:.2f}  COMET={100*sc:.1f}  "
                  f"({time.time()-ts:.1f}s)")
        except Exception as e:
            print(f"   scoring failed ({type(e).__name__}: {e})")
        write_canonical(
            out_dir=out_dir, model=args.model, short_name=short_name,
            src=src, tgt=tgt, template=tmpl, n_shots=k,
            temp=args.temperature, hyps=hyps, system_constant=sys_const,
            metrics=metrics,
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
        # Parallel sliding-window orchestrator. Up to --batch-concurrency
        # OpenAI Batch jobs in flight; poll all every --batch-poll-seconds;
        # as each returns, write predictions and submit the next from the queue.
        print(f"batch mode: concurrency={args.batch_concurrency}, "
              f"poll every {args.batch_poll_seconds}s")
        queue = list(configs)
        in_flight: dict[str, dict] = {}
        done_count = 0

        def fill_window():
            while len(in_flight) < args.batch_concurrency and queue:
                tmpl, k, src, tgt, tsv = queue.pop(0)
                pair = f"{src}-{tgt}"
                strategy = (f"semantic_{k}shot" if tmpl == 0
                            else f"template{tmpl}_{k}shot")
                label = f"{strategy}/{pair}"
                job_label = f"{short_name}/{pair}/{strategy}"
                out_dir = out_temp_dir / strategy / pair

                print(f"\n[submit] {label}  (from {tsv.name})")
                prompts = read_prompts(tsv)
                user_prompts, sys_text = split_for_batch(prompts)
                sys_const = sys_text is not None
                print(f"  prompts: {len(user_prompts)}  system_constant={sys_const}")

                existing = state.get(job_label)
                if existing:
                    job_id = existing
                    print(f"  resuming pre-submitted job: {job_id}")
                else:
                    job_id = api_client.submit_batch_job(
                        prompts=user_prompts, job_name=job_label,
                        system_prompt=sys_text,
                    )
                    state[job_label] = job_id
                    state_file.parent.mkdir(parents=True, exist_ok=True)
                    state_file.write_text(json.dumps(state, indent=2))
                    print(f"  submitted: {job_id}")

                in_flight[job_label] = {
                    "cfg": (tmpl, k, src, tgt, tsv),
                    "label": label,
                    "job_id": job_id,
                    "sys_const": sys_const,
                    "out_dir": out_dir,
                    "started_at": time.time(),
                }

        fill_window()
        while in_flight or queue:
            completed_keys = []
            for job_label, info in list(in_flight.items()):
                try:
                    hyps = api_client.retrieve_batch_results(info["job_id"])
                except Exception as e:
                    print(f"  ✗ {info['label']} retrieve failed "
                          f"({type(e).__name__}: {e}); will retry next poll")
                    continue
                if hyps is None:
                    continue
                tmpl, k, src, tgt, tsv = info["cfg"]
                hyps = [(h or "").strip() for h in hyps]
                print(f"\n[done] {info['label']}  ({len(hyps)} responses)")
                state.pop(job_label, None)
                state_file.write_text(json.dumps(state, indent=2))
                done_count += 1
                score_and_write(
                    src=src, tgt=tgt, hyps=hyps, sys_const=info["sys_const"],
                    tmpl=tmpl, k=k, out_dir=info["out_dir"],
                    ts=info["started_at"], idx=done_count, label=info["label"],
                )
                completed_keys.append(job_label)
            for jk in completed_keys:
                del in_flight[jk]
            fill_window()
            if in_flight:
                time.sleep(args.batch_poll_seconds)
    else:
        # Direct (non-batch) sequential loop with per-row cache.
        for idx, (tmpl, k, src, tgt, tsv) in enumerate(configs, 1):
            pair = f"{src}-{tgt}"
            strategy = (f"semantic_{k}shot" if tmpl == 0
                        else f"template{tmpl}_{k}shot")
            label = f"{strategy}/{pair}"
            out_dir = out_temp_dir / strategy / pair

            prompts = read_prompts(tsv)
            sys_const = system_is_constant(prompts)
            cache_path = cache_path_for(out_dir)
            cached = load_cache(cache_path)

            print(f"\n[{idx}/{len(configs)}] {label}  "
                  f"({len(prompts)} prompts, {len(cached)} cached)  "
                  f"-> {out_dir.relative_to(ROOT)}")

            ts = time.time()
            hyps = [None] * len(prompts)
            for i, (s, u) in enumerate(tqdm(prompts, desc=strategy, leave=False)):
                if i in cached:
                    hyps[i] = cached[i]
                    continue
                try:
                    text = call_one(api_client, s, u)
                except Exception as e:
                    print(f"  ! row {i} failed ({type(e).__name__}: {e})", file=sys.stderr)
                    text = ""
                hyps[i] = text
                append_cache(cache_path, i, text)

            score_and_write(
                src=src, tgt=tgt, hyps=hyps, sys_const=sys_const,
                tmpl=tmpl, k=k, out_dir=out_dir,
                ts=ts, idx=idx, label=label,
            )

    notifier.send(
        f"Run complete: {args.model}",
        f"All {len(configs)} configs done.\n"
        f"Output dir: experiments/llm/closed/{temp_setting(args.temperature)}/icl/{short_name}/",
    )

    print(f"\nTotal: {time.time() - t0:.1f}s")
    print("Aggregate to update summary.csv:")
    print("  python -m afriscience_mt aggregate --out experiments/summary.csv")
    print("Then rebuild paper tables:")
    print("  python scripts/build_paper_tables.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
