"""One-shot Brevo notification check. Sends a single test email to confirm
$BREVO_API_KEY + $BREVO_SENDER_EMAIL are wired correctly. Prints HTTP status
and the response body. Expect 201 + a JSON payload with messageId."""
import json
import os
import sys
import urllib.request


def main() -> int:
    api_key = os.environ.get("BREVO_API_KEY")
    sender = os.environ.get("BREVO_SENDER_EMAIL")
    recipient = os.environ.get("BREVO_RECIPIENT", "afriannotate@gmail.com")

    if not api_key:
        print("error: BREVO_API_KEY not set", file=sys.stderr)
        return 2
    if not sender:
        print("error: BREVO_SENDER_EMAIL not set", file=sys.stderr)
        return 2

    payload = json.dumps({
        "sender": {"email": sender, "name": "AfriScience-MT"},
        "to": [{"email": recipient}],
        "subject": "[AfriScience-MT] Brevo wiring test",
        "textContent": (
            "If you're reading this, BREVO_API_KEY + BREVO_SENDER_EMAIL "
            "are correctly configured. Safe to start the rerun_gemini_newer "
            "run with notifications enabled."
        ),
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=payload,
        headers={
            "api-key": api_key,
            "accept": "application/json",
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read().decode("utf-8", errors="replace")
            print(f"HTTP {r.status}")
            print(body)
            return 0 if r.status < 300 else 1
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        print(f"HTTP {e.code} {e.reason}", file=sys.stderr)
        print(body, file=sys.stderr)
        return 1
    except Exception as e:
        print(f"network error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
