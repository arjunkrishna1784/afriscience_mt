"""Read-only check: report the state of every Gemini batch in
experiments/_gemini_batch_state.json. Does NOT cancel anything. Use this
to decide whether the in-flight jobs are healthy (RUNNING/SUCCEEDED) or
broken (FAILED) before deciding to cancel."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = ROOT / "experiments" / "_gemini_batch_state.json"


def main() -> int:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("error: GOOGLE_API_KEY not set", file=sys.stderr)
        return 2
    if not STATE_FILE.exists():
        print("no state file — no batches recorded")
        return 0

    state = json.loads(STATE_FILE.read_text())
    if not state:
        print("state file empty — no batches recorded")
        return 0

    from google import genai
    client = genai.Client(api_key=api_key)

    print(f"checking {len(state)} batch job(s)...\n")
    healthy_pending = 0
    succeeded = 0
    failed_or_terminal = 0
    for job_key, batch_name in state.items():
        try:
            job = client.batches.get(name=batch_name)
            s = job.state.name if hasattr(job.state, "name") else str(job.state)
            verdict = (
                "✓ keep (still progressing)" if any(x in s for x in ("PENDING", "RUNNING"))
                else "★ keep (succeeded — has results)" if "SUCCEEDED" in s
                else "✗ broken (cancel/clear safely)" if any(x in s for x in ("FAILED", "EXPIRED", "CANCELLED"))
                else "? unknown"
            )
            print(f"  {job_key}")
            print(f"    {batch_name}")
            print(f"    state: {s}    →  {verdict}")
            if "SUCCEEDED" in s:
                succeeded += 1
            elif any(x in s for x in ("PENDING", "RUNNING")):
                healthy_pending += 1
            else:
                failed_or_terminal += 1
        except Exception as e:
            print(f"  {job_key}")
            print(f"    {batch_name}")
            print(f"    error: {type(e).__name__}: {e}")
            failed_or_terminal += 1
        print()

    print(f"summary: {healthy_pending} pending/running, {succeeded} succeeded, "
          f"{failed_or_terminal} broken/terminal")
    if failed_or_terminal and not (healthy_pending or succeeded):
        print("→ All broken; safe to run scripts/_cancel_gemini_batches.py to clear state.")
    elif healthy_pending or succeeded:
        print("→ Some jobs are still useful; do NOT run the cancel script if you "
              "want to keep them. Either let them finish (rerun script will pick "
              "them up via state file) or remove only the broken entries by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
