"""Brevo email notifier for long-running scripts.

Add `--notify-email` / `--notify-from` / `--brevo-api-key` / `--notify-every`
/ `--no-notify` flags to any argparse-based script with
`add_notify_args(parser)`, then build a Notifier with
`notifier_from_args(args)`. The notifier is a no-op when any of
api_key / sender / recipient is missing, so callers can invoke .send()
and .maybe_progress() unconditionally without guards.

Example:
    parser = argparse.ArgumentParser()
    add_notify_args(parser)
    args = parser.parse_args()
    notifier = notifier_from_args(args)
    notifier.send("Started", "...")
    for i, item in enumerate(items, 1):
        ...
        notifier.maybe_progress(i=i, total=len(items), last_cfg=label)
    notifier.send("Done", "...")
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request


def _send_brevo_email(*, api_key: str, sender: str, recipient: str,
                      subject: str, body: str) -> bool:
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
        print(f"  brevo notify failed: {e}", file=sys.stderr)
        return False


class Notifier:
    def __init__(self, *, api_key, sender, recipient, every_n):
        self.api_key = api_key
        self.sender = sender
        self.recipient = recipient
        self.every_n = every_n
        self.enabled = bool(api_key and sender and recipient)
        self.last_t = time.time()

    def send(self, subject, body):
        if not self.enabled:
            return
        ok = _send_brevo_email(
            api_key=self.api_key, sender=self.sender, recipient=self.recipient,
            subject=f"[AfriScience-MT] {subject}", body=body,
        )
        if ok:
            print(f"  notified {self.recipient}: {subject}")
        self.last_t = time.time()

    def maybe_progress(self, *, i, total, last_cfg):
        if not self.enabled:
            return
        if i % self.every_n == 0 or (time.time() - self.last_t) > 3600:
            self.send(
                f"Progress: {i}/{total} configs",
                f"Last completed: {last_cfg}\n"
                f"Elapsed since last update: {time.time()-self.last_t:.0f}s",
            )


def add_notify_args(parser, *, default_email: str = "afriannotate@gmail.com",
                    default_every: int = 20) -> None:
    """Attach Brevo notifier flags to an argparse parser.

    Sender + API key default to BREVO_SENDER_EMAIL / BREVO_API_KEY env vars
    so .env-driven scripts pick them up automatically.
    """
    g = parser.add_argument_group("notifications (Brevo email)")
    g.add_argument("--notify-email", default=default_email,
                   help=f"Recipient address (default: {default_email}).")
    g.add_argument("--notify-from", default=os.environ.get("BREVO_SENDER_EMAIL"),
                   help="Verified Brevo sender (default: $BREVO_SENDER_EMAIL).")
    g.add_argument("--brevo-api-key", default=os.environ.get("BREVO_API_KEY"),
                   help="Brevo API key (default: $BREVO_API_KEY).")
    g.add_argument("--notify-every", type=int, default=default_every,
                   help=f"Send progress email every N units "
                        f"(default {default_every}).")
    g.add_argument("--no-notify", action="store_true",
                   help="Disable email notifications even if API key + "
                        "sender + recipient are all set.")


def notifier_from_args(args) -> Notifier:
    """Build a Notifier from argparse args parsed via add_notify_args()."""
    if getattr(args, "no_notify", False):
        return Notifier(api_key=None, sender=None, recipient=None,
                        every_n=10**9)
    return Notifier(
        api_key=getattr(args, "brevo_api_key", None),
        sender=getattr(args, "notify_from", None),
        recipient=getattr(args, "notify_email", None),
        every_n=getattr(args, "notify_every", 20),
    )
