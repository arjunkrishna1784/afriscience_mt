"""Rerun Gemini on a newer model (e.g. gemini-3-flash-preview) using the
EXACT prompts that were sent to gemini-1.5-flash, read from the legacy TSV
dump's `prompt` column. No prompt rebuilding — just resend.

This guarantees byte-for-byte prompt parity with gemini-1.5-flash, so the
new predictions support a clean "model version" ablation:

    legacy gemini-1.5-flash  ──┐
                                ├── identical prompts ──── compare outputs
    rerun  gemini-3-flash    ──┘

Source of prompts (preferred → fallback):
    /ext_data/idris/DCS-MT-Updated/old_experiments/LLMS/
    /home/idris/ext_data/DCS-MT-Old/LLMS/translations/

Each TSV (e.g. eng-to-amh-template3-shots5.tsv) has columns:
    src_text    prompt    translation
We read `prompt` and send it to the new model. The original prompts are in
LangChain's role-flattened format ("System: <sys>\\nHuman: <user>"); we
parse that and send proper system + user messages where the system part
is constant across the TSV (templates 1/2/3), and fall back to a single
user message where it varies per row (templates 4/5 — glossary differs
per source). Either way, the prompt content is byte-identical to what
gemini-1.5 received.

Sweep: whatever exists in the legacy dump for the chosen temperature.
50-shot files are skipped per project policy (docs/PAPER_TODOS.md).

Outputs:
    experiments/llm/closed/<temp>/icl/<short_name>/template{N}_{K}shot/<pair>/
                                                                     {predictions,results}.json

Usage:
    export GOOGLE_API_KEY=...
    python scripts/rerun_gemini_newer.py --model gemini-3-flash-preview
    python scripts/rerun_gemini_newer.py --model gemini-3-flash-preview --temperature 0.6
    python scripts/rerun_gemini_newer.py --model gemini-3-flash-preview --use-batch
"""
from __future__ import annotations

import argparse
import asyncio
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

from tqdm import tqdm

# --- silence noisy library warnings -----------------------------------------
# 1) FutureWarning about Python 3.9 EOL (from google-genai or its deps).
warnings.filterwarnings("ignore", category=FutureWarning)
# 2) The google-genai SDK uses print() (not warnings.warn) for the
#    "non-text parts" / "thought_signature" notice — Gemini-3 thinking models
#    return a thought trace alongside the answer; the .text accessor
#    correctly returns only the answer, so the notice is harmless. Wrap
#    stderr to drop those lines.
class _FilteredStderr(io.TextIOBase):
    _DROP_PATTERNS = (
        "non-text parts in the response",
        "thought_signature",
        "AFC is enabled with max remote calls",
    )
    def __init__(self, real): self.real = real; self._buf = ""
    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if not any(p in line for p in self._DROP_PATTERNS):
                self.real.write(line + "\n")
        return len(s)
    def flush(self):
        if self._buf:
            if not any(p in self._buf for p in self._DROP_PATTERNS):
                self.real.write(self._buf)
            self._buf = ""
        self.real.flush()

sys.stderr = _FilteredStderr(sys.stderr)

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR  = ROOT / "data" / "processed_files" / "test"
EXP_ROOT  = ROOT / "experiments" / "llm" / "closed"

# Preferred source first; fall back to DCS-MT-Old if unavailable.
LEGACY_DUMP_CANDIDATES = [
    Path("/ext_data/idris/DCS-MT-Updated/old_experiments/LLMS"),
    Path("/home/idris/ext_data/DCS-MT-Old/LLMS/translations"),
]

LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
PAIRS = [f"eng-{l}" for l in LANGS] + [f"{l}-eng" for l in LANGS]

EXPECTED_N = 970


def find_legacy_dump(temp: float) -> Path:
    """Return the legacy dir for gemini-1.5-flash at this temperature."""
    sub = f"gemini-1.5-flash-002_temp_{temp:.1f}"
    for root in LEGACY_DUMP_CANDIDATES:
        d = root / sub
        if d.is_dir():
            return d
    raise SystemExit(f"no legacy dump found for {sub} under {LEGACY_DUMP_CANDIDATES}")


def list_legacy_configs(legacy_dir: Path) -> list[tuple[int, int, str, str, Path]]:
    """Yield (template, shots, src, tgt, tsv_path) for every TSV in the dump,
    skipping 50-shot files (project policy). Recognises two filename forms:
        <src>-to-<tgt>-template<N>-shots<K>.tsv     -> (N, K, ...)
        <src>-to-<tgt>-semantic-shots<K>.tsv         -> (0, K, ...)
    template == 0 is the sentinel for semantic-similarity exemplars; the
    writer turns it into the ``semantic_<K>shot`` strategy."""
    rx_tmpl = re.compile(
        r"^(?P<src>[a-z]+)-to-(?P<tgt>[a-z]+)-template(?P<n>\d+)"
        r"-shots(?P<k>\d+)(?P<old>(-old)?)\.tsv$"
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
            k = int(m.group("k"))
            if k > 20:
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
    return out


def sanitise_short(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").lower()


def temp_setting(temp: float) -> str:
    return "t0" if abs(temp) < 1e-9 else f"t{int(round(temp * 10))}".replace("0", "")


def parse_prompt(saved: str) -> tuple[str, str]:
    """LangChain debug format: 'System: <sys>\\nHuman: <user>' → (sys, user).
    If no Human: marker, return ('', saved) to send as a single user message."""
    if "\nHuman: " not in saved:
        return "", saved
    sys_part, user_part = saved.split("\nHuman: ", 1)
    if sys_part.startswith("System: "):
        sys_part = sys_part[len("System: "):]
    return sys_part, user_part


def read_prompts(tsv: Path) -> list[tuple[str, str]]:
    """Return [(system, user), ...] in row order, parsed from the TSV's
    `prompt` column. Pads with empties if a row is missing prompt."""
    out = []
    with open(tsv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            p = r.get("prompt") or ""
            out.append(parse_prompt(p))
    return out


def load_test_refs(lang: str) -> list[str]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


def load_test_sources(lang: str) -> list[str]:
    with open(TEST_DIR / f"{lang}.tsv") as f:
        return [r["text"] for r in csv.DictReader(f, delimiter="\t")]


# --- Brevo email notifications ---------------------------------------------

def _send_brevo_email(*, api_key: str, sender: str, recipient: str,
                      subject: str, body: str) -> bool:
    """POST to Brevo's transactional email API. Best-effort: returns False
    on any failure (network, auth, etc.) but never raises."""
    import urllib.request
    payload = json.dumps({
        "sender": {"email": sender, "name": "AfriScience-MT"},
        "to": [{"email": recipient}],
        "subject": subject,
        "textContent": body,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=payload,
        headers={
            "accept": "application/json",
            "api-key": api_key,
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status < 300
    except Exception as e:
        print(f"  ✗ brevo notify failed: {e}", file=sys.stderr)
        return False


class Notifier:
    """Sends Brevo email at start, every N configs, success, and failure.
    Disabled silently if API key/sender/recipient are not configured."""

    def __init__(self, *, api_key: str | None, sender: str | None,
                 recipient: str | None, every_n: int):
        self.api_key = api_key
        self.sender = sender
        self.recipient = recipient
        self.every_n = every_n
        self.enabled = bool(api_key and sender and recipient)
        self.last_t = time.time()

    def send(self, subject: str, body: str) -> None:
        if not self.enabled:
            return
        ok = _send_brevo_email(
            api_key=self.api_key, sender=self.sender, recipient=self.recipient,
            subject=f"[AfriScience-MT] {subject}", body=body,
        )
        if ok:
            print(f"  ✉ notified {self.recipient}: {subject}")
        self.last_t = time.time()

    def maybe_progress(self, *, i: int, total: int, last_cfg: str) -> None:
        if not self.enabled:
            return
        # Send every N configs, OR once an hour even if N hasn't ticked.
        if i % self.every_n == 0 or (time.time() - self.last_t) > 3600:
            self.send(
                f"Progress: {i}/{total} configs",
                f"Last completed: {last_cfg}\n"
                f"Elapsed since last update: {time.time() - self.last_t:.0f}s",
            )


def cache_path_for(out_dir: Path) -> Path:
    """Per-prompt JSONL cache living next to predictions.json so a crash
    mid-config doesn't lose API calls. One line per completed prompt:
    {"i": <idx>, "hyp": <text>}. Deleted after predictions.json is written."""
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
                continue  # skip corrupt lines
    return done


def append_cache(cache_path: Path, idx: int, hyp: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"i": idx, "hyp": hyp}, ensure_ascii=False) + "\n")


async def _generate_one(api_client, idx: int, prompt: str,
                        cfg, sem: asyncio.Semaphore) -> tuple[int, str]:
    """One async generate call, rate-limited by the shared semaphore."""
    types = api_client._types
    async with sem:
        response = await api_client.client.aio.models.generate_content(
            model=api_client.config.model_name,
            contents=types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
            config=cfg,
        )
        return idx, (response.text or "").strip()


async def _async_generate_with_cache(
    api_client, *, prompts: list[str], system_prompt: str | None,
    cache_path: Path, desc: str, concurrency: int = 10,
) -> list[str]:
    """Single-loop async generation with progress + per-prompt cache. All
    requests share one aiohttp session (the SDK's), so this MUST be called
    on the same event loop for every config — see GeminiSession."""
    if api_client.client is None:
        api_client._initialize_client()
    types = api_client._types
    cfg_kwargs = dict(
        temperature=api_client.config.temperature,
        max_output_tokens=api_client.config.max_tokens,
        top_p=api_client.config.top_p,
    )
    if system_prompt:
        cfg_kwargs["system_instruction"] = system_prompt
    cfg = types.GenerateContentConfig(**cfg_kwargs)

    done = load_cache(cache_path)
    n = len(prompts)
    todo = [i for i in range(n) if i not in done]
    sem = asyncio.Semaphore(concurrency)
    pbar = tqdm(total=n, initial=n - len(todo), desc=desc, unit="p", leave=False)

    coros = [_generate_one(api_client, i, prompts[i], cfg, sem) for i in todo]
    for fut in asyncio.as_completed(coros):
        i, text = await fut
        done[i] = text
        append_cache(cache_path, i, text)
        pbar.update(1)
    pbar.close()
    return [done[i] for i in range(n)]


def generate_with_cache_and_progress(
    loop: asyncio.AbstractEventLoop, api_client, *,
    prompts: list[str], system_prompt: str | None,
    cache_path: Path, desc: str,
) -> list[str]:
    """Sync wrapper that runs the async generator on a SHARED loop —
    critical so the SDK's aiohttp session stays bound to one live loop
    across all configs (mixing event loops causes 'Event loop is closed').
    """
    return loop.run_until_complete(_async_generate_with_cache(
        api_client, prompts=prompts, system_prompt=system_prompt,
        cache_path=cache_path, desc=desc,
    ))


def write_canonical(*, out_dir: Path, model: str, short_name: str,
                    src: str, tgt: str, template: int, n_shots: int,
                    temp: float, hyps: list[str], system_constant: bool,
                    metrics: dict | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pair = f"{src}-{tgt}"
    # template == 0 is the sentinel for semantic-similarity exemplars.
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
        "source_lang": src,
        "target_lang": tgt,
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
            "prompt_strategy": strategy,
            "n_shots": n_shots,
            "temperature": temp,
            "temp_setting": temp_setting(temp),
        },
        "metrics": res_metrics,
        "metadata": {
            "source": "rerun_gemini_newer",
            "is_ablation": True,
            "prompt_strategy": strategy,
            "n_shots": n_shots,
            "temp_setting": temp_setting(temp),
            "prompt_provenance": "imported_from_legacy_gemini_15_flash_dump",
            "system_constant_per_tsv": system_constant,
            "needs_eval": metrics is None,
        },
    }, indent=2, ensure_ascii=False))


def prepare_config_prompts(tsv: Path) -> tuple[list[tuple[str, str]], str | None,
                                                 list[str], bool]:
    """Read a legacy TSV, parse each row into (system, user) parts, and
    decide whether the system part is constant across rows.

    Returns (parsed, system_prompt, user_prompts, system_constant).
    - system_constant=True  → batchable with shared system_prompt
    - system_constant=False → fall back to single-message verbatim re-join
    """
    parsed = read_prompts(tsv)
    sys_parts = {sp for sp, _ in parsed}
    system_constant = len(sys_parts) == 1 and next(iter(sys_parts)) != ""
    if system_constant:
        system_prompt = next(iter(sys_parts))
        user_prompts = [u for _, u in parsed]
    else:
        system_prompt = None
        user_prompts = [
            f"System: {sp}\nHuman: {up}" if sp else up
            for sp, up in parsed
        ]
    return parsed, system_prompt, user_prompts, system_constant


def score_write_and_notify(
    *, args, hyps: list[str], src: str, tgt: str, template: int, n_shots: int,
    out_dir: Path, short_name: str, system_constant: bool,
    notifier: "Notifier", notify_state: dict, total: int, done_so_far: int,
    label: str, started_at: float,
) -> None:
    """Compute metrics, write predictions+results, drop cache, send progress."""
    metrics = None
    if not args.skip_scoring:
        try:
            from afriscience_mt.evaluation.metrics import compute_all_metrics
            refs = load_test_refs(tgt)[:len(hyps)]
            srcs = load_test_sources(src)[:len(hyps)]
            m = compute_all_metrics(srcs, hyps, refs, metrics=["chrf", "ssa_comet"])
            metrics = {
                "bleu": None,
                "chrf": m.get("chrf"),
                "ssa_comet": m.get("ssa_comet"),
                "num_samples": len(hyps),
            }
            cf, sc = metrics["chrf"], metrics["ssa_comet"]
            print(f"  ChrF={cf:.2f}  SSA-COMET={100*sc:.1f}"
                  if cf is not None and sc is not None else f"  metrics={metrics}")
        except Exception as e:
            print(f"  scoring failed ({type(e).__name__}: {e}); writing predictions only")

    write_canonical(
        out_dir=out_dir, model=args.model, short_name=short_name,
        src=src, tgt=tgt, template=template, n_shots=n_shots,
        temp=args.temperature, hyps=hyps,
        system_constant=system_constant, metrics=metrics,
    )
    cache = cache_path_for(out_dir)
    if cache.exists():
        cache.unlink()
    print(f"  ✓ [{done_so_far}/{total}] wrote {out_dir.relative_to(ROOT)} "
          f"({time.time()-started_at:.1f}s)")
    notify_state["i"] = done_so_far
    notify_state["last_cfg"] = label
    notifier.maybe_progress(i=done_so_far, total=total, last_cfg=label)


def run_batch_parallel(
    *, api_client, todo: list, args, state: dict, state_file: Path,
    short_name: str, notifier: "Notifier", notify_state: dict,
) -> None:
    """Sliding-window batch orchestrator: keep up to --batch-concurrency jobs
    in flight, poll all of them every --batch-poll-seconds, and as each one
    returns, write its predictions and submit the next from the queue.

    Resumable: any job_name still in the state file from a previous run is
    treated as already-submitted, so we just poll instead of resubmitting.
    """
    queue = list(todo)
    in_flight: dict[str, dict] = {}  # job_key -> dict of fields
    total = len(todo)
    done = 0

    print(f"batch mode: concurrency={args.batch_concurrency}, "
          f"poll every {args.batch_poll_seconds}s")

    def fill_window():
        """Submit (or resume) up to args.batch_concurrency jobs."""
        while len(in_flight) < args.batch_concurrency and queue:
            cfg = queue.pop(0)
            tmpl, k, src, tgt, tsv, out_dir = cfg
            pair = f"{src}-{tgt}"
            strategy = (f"semantic_{k}shot" if tmpl == 0
                        else f"template{tmpl}_{k}shot")
            label = f"{strategy}/{pair}"
            job_key = f"{short_name}/{temp_setting(args.temperature)}/{pair}/{strategy}"

            print(f"\n[submit] {label}  (from {tsv.name})")
            _, system_prompt, user_prompts, system_constant = prepare_config_prompts(tsv)
            print(f"  prompts: {len(user_prompts)}  system_constant={system_constant}")

            existing = state.get(job_key)
            if existing:
                job_name = existing
                print(f"  resuming pre-submitted job: {job_name}")
            else:
                job_name = api_client.submit_batch_job(
                    prompts=user_prompts,
                    job_name=f"afriscience-{short_name}-{src}{tgt}-{strategy}",
                    system_prompt=system_prompt,
                )
                state[job_key] = job_name
                state_file.parent.mkdir(parents=True, exist_ok=True)
                state_file.write_text(json.dumps(state, indent=2))
                print(f"  submitted: {job_name}")

            in_flight[job_key] = {
                "cfg": cfg,
                "label": label,
                "job_name": job_name,
                "system_constant": system_constant,
                "started_at": time.time(),
            }

    fill_window()

    while in_flight or queue:
        # Poll every in-flight job once
        completed_keys = []
        for job_key, info in list(in_flight.items()):
            try:
                hyps = api_client.retrieve_batch_results(info["job_name"])
            except Exception as e:
                print(f"  ✗ {info['label']} retrieve failed ({type(e).__name__}: {e}); "
                      f"will retry on next poll")
                continue
            if hyps is None:
                continue  # still pending
            # Done
            tmpl, k, src, tgt, tsv, out_dir = info["cfg"]
            hyps = [(h or "").strip() for h in hyps]
            print(f"\n[done] {info['label']}  ({len(hyps)} responses)")
            state.pop(job_key, None)
            state_file.write_text(json.dumps(state, indent=2))
            done += 1
            score_write_and_notify(
                args=args, hyps=hyps, src=src, tgt=tgt,
                template=tmpl, n_shots=k, out_dir=out_dir,
                short_name=short_name, system_constant=info["system_constant"],
                notifier=notifier, notify_state=notify_state,
                total=total, done_so_far=done, label=info["label"],
                started_at=info["started_at"],
            )
            completed_keys.append(job_key)
        for jk in completed_keys:
            del in_flight[jk]

        # Top up the window (resubmit nothing — already-submitted are still in flight)
        fill_window()

        # Sleep before next poll round, unless everything's done
        if in_flight:
            time.sleep(args.batch_poll_seconds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Gemini model name (e.g. gemini-3-flash-preview)")
    ap.add_argument("--short-name", default=None,
                    help="model_short for summary.csv. Defaults to sanitised --model.")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Subset of language pairs (default: whatever's in the legacy dump).")
    ap.add_argument("--templates", nargs="+", type=int, default=None,
                    choices=[0, 1, 2, 3, 4, 5],
                    help="Subset of templates (default: all 1..5 that exist).")
    ap.add_argument("--shots", nargs="+", type=int, default=None,
                    help="Subset of shot counts (default: 0,5,10,20).")
    ap.add_argument("--shot-counts", nargs="+", type=int, default=None, dest="shots",
                    help="Alias for --shots (kept for parity with the OpenAI rerun).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="No-op alias: this script already skips cells whose "
                         "predictions.json exists by default; --force is the "
                         "opposite (re-run completed cells). Accepted for "
                         "wrapper-script symmetry with rerun_openai_doc.py "
                         "etc.")
    ap.add_argument("--direction", choices=["e2x", "x2e", "both"], default="both",
                    help="Restrict to eng->xxx (e2x), xxx->eng (x2e), or both.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Picks legacy dump dir: 0.0 → gemini-1.5-flash-002_temp_0.0/")
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--skip-scoring", action="store_true")
    ap.add_argument("--use-batch", action="store_true",
                    help="Use Gemini Batch API (50%% cheaper, ~24h SLA).")
    ap.add_argument("--batch-poll-seconds", type=int, default=120)
    ap.add_argument("--batch-concurrency", type=int, default=5,
                    help="Max in-flight batch jobs at once (--use-batch only).")
    ap.add_argument("--batch-state-file",
                    default="experiments/_gemini_batch_state.json")
    ap.add_argument("--force", action="store_true")
    # Brevo email notifications (start / every N configs / success / failure).
    ap.add_argument("--notify-email", default="afriannotate@gmail.com",
                    help="Recipient for Brevo notifications.")
    ap.add_argument("--notify-from", default=os.environ.get("BREVO_SENDER_EMAIL"),
                    help="Verified Brevo sender (default: $BREVO_SENDER_EMAIL).")
    ap.add_argument("--brevo-api-key", default=os.environ.get("BREVO_API_KEY"),
                    help="Brevo API key (default: $BREVO_API_KEY).")
    ap.add_argument("--notify-every", type=int, default=25,
                    help="Send a progress email every N completed configs.")
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("error: pass --api-key or set GOOGLE_API_KEY", file=sys.stderr)
        return 2

    short_name = args.short_name or sanitise_short(args.model)
    legacy_dir = find_legacy_dump(args.temperature)
    print(f"legacy prompts source: {legacy_dir}")

    legacy_configs = list_legacy_configs(legacy_dir)
    if args.pairs:
        bad = [p for p in args.pairs if p not in PAIRS]
        if bad:
            print(f"unknown pairs: {bad}", file=sys.stderr)
            return 2
        legacy_configs = [c for c in legacy_configs
                          if f"{c[2]}-{c[3]}" in args.pairs]
    if args.templates:
        legacy_configs = [c for c in legacy_configs if c[0] in args.templates]
    if args.shots is not None:
        legacy_configs = [c for c in legacy_configs if c[1] in args.shots]
    if args.direction == "e2x":
        legacy_configs = [c for c in legacy_configs if c[2] == "eng"]
    elif args.direction == "x2e":
        legacy_configs = [c for c in legacy_configs if c[3] == "eng"]

    out_root = EXP_ROOT / temp_setting(args.temperature) / "icl" / short_name

    todo = []
    for tmpl, k, src, tgt, tsv in legacy_configs:
        strat = f"semantic_{k}shot" if tmpl == 0 else f"template{tmpl}_{k}shot"
        out_dir = out_root / strat / f"{src}-{tgt}"
        if (out_dir / "predictions.json").exists() and not args.force:
            continue
        todo.append((tmpl, k, src, tgt, tsv, out_dir))

    print(f"model={args.model}  short={short_name}  T={args.temperature}  "
          f"available_in_dump={len(legacy_configs)}  to_run={len(todo)}")
    if not todo:
        print("nothing to do (use --force to redo)")
        return 0

    from afriscience_mt.api.factory import create_api_client
    api_client = create_api_client(
        provider="google",
        model_name=args.model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
    )

    state_file = ROOT / args.batch_state_file
    state = json.loads(state_file.read_text()) if state_file.exists() else {}

    # Single shared event loop for all configs — keeps the SDK's aiohttp
    # session alive across calls. Without this we'd hit "Event loop is closed".
    loop = asyncio.new_event_loop() if not args.use_batch else None
    if loop is not None:
        asyncio.set_event_loop(loop)

    notifier = Notifier(
        api_key=args.brevo_api_key,
        sender=args.notify_from,
        recipient=args.notify_email,
        every_n=args.notify_every,
    )
    if not notifier.enabled:
        print("notify: disabled (need BREVO_API_KEY + --notify-from + --notify-email)")
    else:
        print(f"notify: enabled → {args.notify_email} every {args.notify_every} configs")
        notifier.send(
            f"Started: {args.model}",
            f"Model: {args.model}\nShort: {short_name}\nT={args.temperature}\n"
            f"Configs to run: {len(todo)} (of {len(legacy_configs)} available)\n"
            f"Mode: {'batch' if args.use_batch else 'direct (10-way concurrent)'}",
        )

    # Track progress for the failure-notification excepthook (no try/except
    # needed in the main loop — excepthook fires for any uncaught exception
    # including KeyboardInterrupt).
    notify_state = {"i": 0, "total": len(todo), "last_cfg": ""}
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

    if args.use_batch:
        run_batch_parallel(
            api_client=api_client, todo=todo, args=args,
            state=state, state_file=state_file, short_name=short_name,
            notifier=notifier, notify_state=notify_state,
        )
    else:
        for i, (tmpl, k, src, tgt, tsv, out_dir) in enumerate(todo, 1):
            pair = f"{src}-{tgt}"
            strategy = (f"semantic_{k}shot" if tmpl == 0
                        else f"template{tmpl}_{k}shot")
            label = f"{strategy}/{pair}"
            notify_state["i"] = i
            notify_state["last_cfg"] = label
            print(f"\n[{i}/{len(todo)}] {label}  (from {tsv.name})")
            ts = time.time()

            parsed, system_prompt, user_prompts, system_constant = (
                prepare_config_prompts(tsv)
            )
            if len(parsed) < EXPECTED_N:
                print(f"  ⚠ legacy TSV has {len(parsed)} rows, expected {EXPECTED_N}")
            if system_constant:
                print(f"  system constant ({len(system_prompt)} chars); "
                      f"sending {len(user_prompts)} user messages with shared system_prompt")
            else:
                print(f"  system varies per row → sending full-prompt user messages "
                      f"({len(user_prompts)} total)")

            cache = cache_path_for(out_dir)
            already = len(load_cache(cache))
            if already:
                print(f"  resuming: {already}/{len(user_prompts)} prompts cached")
            else:
                print(f"  generating ({len(user_prompts)} prompts, ~10-way concurrent)")
            hyps = generate_with_cache_and_progress(
                loop, api_client,
                prompts=user_prompts,
                system_prompt=system_prompt,
                cache_path=cache,
                desc=label,
            )
            score_write_and_notify(
                args=args, hyps=hyps, src=src, tgt=tgt,
                template=tmpl, n_shots=k, out_dir=out_dir,
                short_name=short_name, system_constant=system_constant,
                notifier=notifier, notify_state=notify_state,
                total=len(todo), done_so_far=i, label=label,
                started_at=ts,
            )

    if loop is not None:
        try:
            pending = asyncio.all_tasks(loop)
            for t in pending:
                t.cancel()
            loop.close()
        except Exception:
            pass

    notifier.send(
        f"Run complete: {args.model}",
        f"All {len(todo)} configs done.\n"
        f"Output dir: experiments/llm/closed/{temp_setting(args.temperature)}/icl/{short_name}/\n"
        f"Next: python -m afriscience_mt aggregate --out experiments/summary.csv",
    )

    print("\nAll configs done. Aggregate to update summary.csv:")
    print("  python -m afriscience_mt aggregate --out experiments/summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
