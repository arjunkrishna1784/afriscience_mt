"""Cancel any Gemini batch jobs listed in experiments/_gemini_batch_state.json
and clear the state file. Safe to run more than once — already-finished or
already-cancelled jobs are ignored."""
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
        print("no state file — nothing to cancel")
        return 0

    state = json.loads(STATE_FILE.read_text())
    if not state:
        print("state file empty — nothing to cancel")
        return 0

    print(f"cancelling {len(state)} batch jobs...")
    from google import genai
    client = genai.Client(api_key=api_key)

    cancelled = 0
    for job_key, batch_name in list(state.items()):
        try:
            job = client.batches.get(name=batch_name)
            state_str = job.state.name if hasattr(job.state, "name") else str(job.state)
            if any(s in state_str for s in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED")):
                print(f"  - {batch_name} already terminal ({state_str}); skipping")
            else:
                client.batches.cancel(name=batch_name)
                print(f"  ✓ cancelled {batch_name} ({job_key})")
                cancelled += 1
        except Exception as e:
            print(f"  ! {batch_name}: {type(e).__name__}: {e}")

    STATE_FILE.write_text(json.dumps({}, indent=2))
    print(f"\nstate file cleared. Cancelled {cancelled} job(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
