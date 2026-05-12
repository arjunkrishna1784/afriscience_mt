"""Run GPT-4o as a document-level translation judge over the
predictions.json files produced by ``scripts/rerun_openai_doc.py`` or
``scripts/rerun_google_doc.py``.

The three judge prompts are the ones documented in
``_ACL2026__.../eval_prompts.tex`` (\\label{prompt:fluency_eval},
\\label{prompt:accuracy_eval}, \\label{prompt:cohesion_eval}); they
already return structured JSON, so we keep them verbatim and parse
their JSON output here. The judge model is configurable via
``--judge-model``; the paper default is gpt-5.5. Pass a
``gemini-*`` model id to use Google instead. To average two judges,
run the script twice with different ``--judge-model`` and a
different ``--out-name``, then aggregate the resulting judge.json
files offline.

Output schema (one file per cell):
    experiments/llm/closed/doc/<short>/doc_{n}shot_template{N}/<pair>/<out_name>.json
    (default <out_name> = "judge"; pass --out-name to disambiguate when
     running multiple judges over the same predictions for averaging)
    {
      "schema_version": "1.0",
      "judge_model": "gpt-5.5",
      "source_lang": "...",
      "target_lang": "...",
      "prompt_strategy": "doc_<n>shot_template<N>",
      "timestamp": "...",
      "papers": [
        {
          "paper_id": "...",
          "domain": "...",
          "fluency": {"score": 4, "explanation": "..."},
          "accuracy": {"mistakes": ["Wrong Translation: ...", ...]},
          "cohesion": {
            "lexical_mistakes": [...],
            "grammatical_mistakes": [...]
          }
        },
        ...
      ],
      "aggregate": {
        "fluency_mean": 4.3,
        "accuracy_mistakes_mean": 5.7,
        "lexical_mistakes_mean": 1.2,
        "grammatical_mistakes_mean": 0.9
      }
    }

Usage:
    # Default: gpt-5.5 as judge, writes <cell>/judge.json
    export OPENAI_API_KEY=...
    python scripts/run_doc_llm_judge.py \\
        --src experiments/llm/closed/doc/gpt54 \\
        --src experiments/llm/closed/doc/gemini_31_flash_lite

    # Use Gemini as a second judge for averaging; writes
    # <cell>/judge_gemini.json alongside the gpt-5.5 judge.json:
    export GOOGLE_API_KEY=...
    python scripts/run_doc_llm_judge.py \\
        --src experiments/llm/closed/doc/gpt54 \\
        --judge-model gemini-3.1-flash-lite --out-name judge_gemini

    # Single cell while debugging:
    python scripts/run_doc_llm_judge.py \\
        --src experiments/llm/closed/doc/gpt54/doc_0shot_template1/eng-hau
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse Brevo notifier from the sentence-level OpenAI rerun.
from rerun_gpt4o_templated import Notifier  # type: ignore  # noqa: E402

from afriscience_mt.doc_prompts import LANG_DISPLAY  # noqa: E402

# ---- prompt templates (verbatim from eval_prompts.tex) ----------------------
_FLUENCY_TMPL = (
    "Please evaluate the fluency of the following text in {tgt_lang}.\n"
    "------\n"
    "### **Instructions:**\n"
    "- **Task**: Evaluate the fluency of the text.\n"
    "- Scoring: Provide a score from 1 to 5, where:\n"
    "  - **5**: The text is **highly fluent**, with no grammatical errors, unnatural wording, or stiff syntax.\n"
    "  - **4**: The text is **mostly fluent**, with minor errors that do not impede understanding.\n"
    "  - **3**: The text is **moderately fluent**, with noticeable errors that may slightly affect comprehension.\n"
    "  - **2**: The text has **low fluency**, with frequent errors that hinder understanding.\n"
    "  - **1**: The text is **not fluent**, with severe errors that make it difficult to understand.\n"
    "- **Explanation**: Support your score with specific examples to justify your evaluation.\n"
    "------\n"
    "### **Output Format:**\n"
    "Provide your evaluation in the following JSON format:\n"
    "{{\n"
    "  \"Fluency\": {{\n"
    "    \"Score\": \"<the score>\",\n"
    "    \"Explanation\": \"<your explanation on how you made the decision>\"\n"
    "  }}\n"
    "}}\n"
    "------\n"
    "**Text to Evaluate:**\n"
    "{hyp}\n"
)

_ACCURACY_TMPL = (
    "Please evaluate the accuracy of the following translated text in "
    "{tgt_lang} by comparing it to the provided reference text.\n"
    "------\n"
    "### **Instructions:**\n"
    "- **Task**: Compare the text to the reference text.\n"
    "- Identify Mistakes: List all mistakes related to accuracy.\n"
    "- Mistake Types:\n"
    "  - **Wrong Translation**: Incorrect meaning or misinterpretation leading to wrong information.\n"
    "  - **Omission**: Missing words, phrases, or information present in the reference text.\n"
    "  - **Addition**: Extra words, phrases, or information not present in the reference text.\n"
    "  - **Others**: Mistakes that are hard to define or categorize.\n"
    "- **Note**: If the text expresses the same information as the reference text but uses different words or phrasing, it is **not** considered a mistake.\n"
    "- **Provide a List**: Summarize all mistakes without repeating the exact sentences. Provide an empty list if there are no mistakes.\n"
    "------\n"
    "### **Output Format:**\n"
    "Provide your evaluation in the following JSON format:\n"
    "{{\n"
    "  \"Accuracy\": {{\n"
    "    \"Mistakes\": [\n"
    "      \"<list of all mistakes in the text with format 'Mistake Type: summarize the mistake', provide an empty list if there are no mistakes>\"\n"
    "    ]\n"
    "  }}\n"
    "}}\n"
    "------\n"
    "**Reference Text:**\n"
    "{ref}\n"
    "------\n"
    "**Text to Evaluate:**\n"
    "{hyp}\n"
)

_COHESION_TMPL = (
    "Please evaluate the cohesion of the following translated text in "
    "{tgt_lang} by comparing it to the provided reference text.\n"
    "------\n"
    "### **Instructions:**\n"
    "- **Task**: Evaluate the cohesion of the text.\n"
    "- **Definition**: Cohesion refers to how different parts of a text are connected using language structures like grammar and vocabulary. It ensures that sentences flow smoothly and the text makes sense as a whole.\n"
    "- Identify Mistakes: List all mistakes related to cohesion.\n"
    "- Separate the mistakes into:\n"
    "  - **Lexical Cohesion Mistakes**: Issues with vocabulary usage, incorrect or missing synonyms, or overuse of certain words that disrupt the flow.\n"
    "  - **Grammatical Cohesion Mistakes**: Problems with pronouns, conjunctions, or grammatical structures that link sentences and clauses.\n"
    "- **Provide Lists**: Provide separate lists for lexical cohesion mistakes and grammatical cohesion mistakes. Provide empty lists if there are no mistakes.\n"
    "------\n"
    "### **Output Format:**\n"
    "Provide your evaluation in the following JSON format:\n"
    "{{\n"
    "  \"Cohesion\": {{\n"
    "    \"Lexical Cohesion Mistakes\": [\n"
    "      \"<list of all mistakes in the text one by one, provide an empty list if there are no mistakes>\"\n"
    "    ],\n"
    "    \"Grammatical Cohesion Mistakes\": [\n"
    "      \"<list of all mistakes in the text one by one, provide an empty list if there are no mistakes>\"\n"
    "    ]\n"
    "  }}\n"
    "}}\n"
    "------\n"
    "**Reference Text:**\n"
    "{ref}\n"
    "------\n"
    "**Text to Evaluate:**\n"
    "{hyp}\n"
)


# ---- robust JSON parsing ----------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """Try to extract a JSON object from a model response that may be wrapped
    in markdown fences, prose, etc. Returns None if nothing parses."""
    if not text:
        return None
    candidates = []
    # 1) ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))
    # 2) first {...} block
    block = _JSON_BLOCK_RE.search(text)
    if block:
        candidates.append(block.group(0))
    # 3) whole string
    candidates.append(text.strip())
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def _parse_fluency(raw: str) -> dict:
    obj = _extract_json(raw) or {}
    f = obj.get("Fluency") or obj.get("fluency") or {}
    score = f.get("Score") if isinstance(f, dict) else None
    expl = f.get("Explanation") if isinstance(f, dict) else None
    # Score sometimes comes back as a string "4" or "5/5"; coerce.
    score_num = None
    if isinstance(score, (int, float)):
        score_num = float(score)
    elif isinstance(score, str):
        m = re.search(r"(\d+(?:\.\d+)?)", score)
        if m:
            try:
                score_num = float(m.group(1))
            except ValueError:
                score_num = None
    return {"score": score_num, "explanation": expl,
            "raw": raw if score_num is None else None}


def _parse_accuracy(raw: str) -> dict:
    obj = _extract_json(raw) or {}
    a = obj.get("Accuracy") or obj.get("accuracy") or {}
    mistakes = a.get("Mistakes") if isinstance(a, dict) else []
    if not isinstance(mistakes, list):
        mistakes = []
    return {"mistakes": [str(m) for m in mistakes],
            "raw": raw if not isinstance(obj.get("Accuracy"), dict) else None}


def _parse_cohesion(raw: str) -> dict:
    obj = _extract_json(raw) or {}
    c = obj.get("Cohesion") or obj.get("cohesion") or {}
    lex = c.get("Lexical Cohesion Mistakes") if isinstance(c, dict) else []
    gram = c.get("Grammatical Cohesion Mistakes") if isinstance(c, dict) else []
    if not isinstance(lex, list): lex = []
    if not isinstance(gram, list): gram = []
    return {
        "lexical_mistakes": [str(m) for m in lex],
        "grammatical_mistakes": [str(m) for m in gram],
        "raw": raw if not isinstance(obj.get("Cohesion"), dict) else None,
    }


# ---- aggregation -----------------------------------------------------------

def _aggregate(papers: list[dict]) -> dict:
    """Compute per-cell means matching the appendix doc_evaluation columns."""
    if not papers:
        return {
            "fluency_mean": None, "accuracy_mistakes_mean": None,
            "lexical_mistakes_mean": None, "grammatical_mistakes_mean": None,
            "n_papers_scored": 0,
        }
    fl = [p["fluency"]["score"] for p in papers
          if p.get("fluency") and p["fluency"]["score"] is not None]
    am = [len(p["accuracy"]["mistakes"]) for p in papers if p.get("accuracy")]
    lm = [len(p["cohesion"]["lexical_mistakes"]) for p in papers if p.get("cohesion")]
    gm = [len(p["cohesion"]["grammatical_mistakes"]) for p in papers if p.get("cohesion")]
    return {
        "fluency_mean": sum(fl) / len(fl) if fl else None,
        "accuracy_mistakes_mean": sum(am) / len(am) if am else None,
        "lexical_mistakes_mean": sum(lm) / len(lm) if lm else None,
        "grammatical_mistakes_mean": sum(gm) / len(gm) if gm else None,
        "n_papers_scored": len(papers),
    }


# ---- I/O helpers -----------------------------------------------------------

def _iter_cells(src_roots: list[Path]) -> list[Path]:
    """Find every cell directory under the given roots that has a
    predictions.json. A "cell" is .../doc_<n>shot_template<N>/<pair>/."""
    cells = []
    for root in src_roots:
        root = root.resolve()
        if not root.exists():
            print(f"warn: src not found: {root}", file=sys.stderr)
            continue
        # The caller may have pointed at a model root or at a single cell;
        # in either case predictions.json identifies a cell.
        for p in root.rglob("predictions.json"):
            cells.append(p.parent)
    return sorted(set(cells))


def _judge_path(cell_dir: Path, out_name: str = "judge") -> Path:
    return cell_dir / f"{out_name}.json"


def _load_existing(cell_dir: Path, out_name: str = "judge") -> dict | None:
    p = _judge_path(cell_dir, out_name)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _save(cell_dir: Path, payload: dict, out_name: str = "judge") -> None:
    _judge_path(cell_dir, out_name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False))


# ---- main ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True,
                    help="Directory to walk for predictions.json. Repeatable.")
    ap.add_argument("--judge-model", default="gpt-5.5",
                    help="Judge model id. Names starting with 'gemini-' "
                         "route to the Google backend; everything else to "
                         "OpenAI. Default gpt-5.5.")
    ap.add_argument("--out-name", default="judge",
                    help="Filename stem for the saved judgments (default "
                         "'judge' -> 'judge.json'). Pass a different stem "
                         "when running multiple judges over the same "
                         "predictions so they don't overwrite each other "
                         "(e.g. --out-name judge_gemini).")
    ap.add_argument("--api-key", default=None,
                    help="API key (OpenAI or Google depending on --judge-model). "
                         "Defaults to the corresponding env var.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing <out_name>.json files.")
    ap.add_argument("--max-output-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    # Brevo notifier (reused from rerun_gpt4o_templated).
    ap.add_argument("--notify-email", default="afriannotate@gmail.com")
    ap.add_argument("--notify-from", default=os.environ.get("BREVO_SENDER_EMAIL"))
    ap.add_argument("--brevo-api-key", default=os.environ.get("BREVO_API_KEY"))
    ap.add_argument("--notify-every", type=int, default=20)
    args = ap.parse_args()

    # Route by model name: gemini-* -> Google, else -> OpenAI.
    if args.judge_model.lower().startswith("gemini-"):
        provider = "google"
        api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("error: pass --api-key or set GOOGLE_API_KEY", file=sys.stderr)
            return 2
    else:
        provider = "openai"
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("error: pass --api-key or set OPENAI_API_KEY", file=sys.stderr)
            return 2

    src_roots = [Path(s).expanduser().resolve() for s in args.src]
    cells = _iter_cells(src_roots)
    if not cells:
        print("no predictions.json files found under any --src", file=sys.stderr)
        return 2

    # Filter completed cells.
    if not args.force:
        kept = []
        skipped = 0
        for d in cells:
            existing = _load_existing(d, args.out_name)
            if existing and existing.get("aggregate", {}).get("n_papers_scored") == \
                    len(json.loads((d / "predictions.json").read_text())["samples"]):
                skipped += 1
                continue
            kept.append(d)
        print(f"  skipping {skipped} cells already judged "
              f"(use --force to redo)")
        cells = kept

    if not cells:
        print("nothing to do.")
        return 0

    print(f"LLM judge: judge={args.judge_model} (provider={provider})  "
          f"cells={len(cells)}  out={args.out_name}.json")

    from afriscience_mt.api.factory import create_api_client
    judge = create_api_client(
        provider=provider,
        model_name=args.judge_model,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_output_tokens,
    )

    notifier = Notifier(
        api_key=args.brevo_api_key, sender=args.notify_from,
        recipient=args.notify_email, every_n=args.notify_every,
    )
    if not notifier.enabled:
        print("notify: disabled")
    else:
        print(f"notify: enabled -> {args.notify_email} every {args.notify_every} cells")
        notifier.send(
            f"Started: LLM judge ({args.judge_model})",
            f"Judge: {args.judge_model}\nCells: {len(cells)}",
        )

    notify_state = {"i": 0, "total": len(cells), "last_cfg": ""}
    if notifier.enabled:
        def _excepthook(exc_type, exc_value, exc_tb):
            notifier.send(
                f"Judge FAILED at cell {notify_state['i']}/{notify_state['total']}",
                f"Last attempted: {notify_state['last_cfg']}\n"
                f"Error: {exc_type.__name__}: {exc_value}\n\n"
                + "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            )
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        sys.excepthook = _excepthook

    def _call(prompt: str) -> str:
        """Single user message; the judge prompt itself carries all context.
        The OpenAI client takes a messages list; the Google client offers
        the same `_call_api` shape via the factory wrapper."""
        return (judge._call_api([{"role": "user", "content": prompt}]) or "").strip()

    t0 = time.time()
    for idx, cell in enumerate(cells, 1):
        pred_path = cell / "predictions.json"
        try:
            pred = json.loads(pred_path.read_text())
        except json.JSONDecodeError as e:
            print(f"  [{idx}/{len(cells)}] x {cell.relative_to(ROOT)}: "
                  f"bad predictions.json ({e})")
            continue

        src_lang = pred.get("source_lang", "")
        tgt_lang_short = pred.get("target_lang", "")
        tgt_lang = LANG_DISPLAY.get(tgt_lang_short, tgt_lang_short)
        samples = pred.get("samples", [])
        label = (
            f"{cell.parent.parent.name}/{cell.parent.name}/{cell.name}"
            if cell.parent.parent != ROOT else cell.relative_to(ROOT).as_posix()
        )
        print(f"\n[{idx}/{len(cells)}] {label}  ({len(samples)} papers, "
              f"tgt={tgt_lang})")

        # Resume support: re-use already-judged papers if <out_name>.json
        # exists but is incomplete.
        existing = _load_existing(cell, args.out_name) or {}
        existing_by_pid = {
            p["paper_id"]: p for p in existing.get("papers", [])
        } if not args.force else {}

        papers_out = []
        ts = time.time()
        for s in tqdm(samples, leave=False, desc="judge"):
            pid = s.get("paper_id", "")
            if pid in existing_by_pid:
                papers_out.append(existing_by_pid[pid])
                continue
            hyp = s.get("hyp", "")
            ref = s.get("ref", "")
            try:
                fluency_raw = _call(_FLUENCY_TMPL.format(
                    tgt_lang=tgt_lang, hyp=hyp))
            except Exception as e:
                print(f"  ! fluency call failed (paper {pid}): "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                fluency_raw = ""
            try:
                accuracy_raw = _call(_ACCURACY_TMPL.format(
                    tgt_lang=tgt_lang, ref=ref, hyp=hyp))
            except Exception as e:
                print(f"  ! accuracy call failed (paper {pid}): "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                accuracy_raw = ""
            try:
                cohesion_raw = _call(_COHESION_TMPL.format(
                    tgt_lang=tgt_lang, ref=ref, hyp=hyp))
            except Exception as e:
                print(f"  ! cohesion call failed (paper {pid}): "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                cohesion_raw = ""
            papers_out.append({
                "paper_id": pid,
                "domain": s.get("domain", ""),
                "fluency": _parse_fluency(fluency_raw),
                "accuracy": _parse_accuracy(accuracy_raw),
                "cohesion": _parse_cohesion(cohesion_raw),
            })

        agg = _aggregate(papers_out)
        payload = {
            "schema_version": "1.0",
            "judge_model": args.judge_model,
            "source_lang": src_lang,
            "target_lang": tgt_lang_short,
            "prompt_strategy": pred.get("experiment_id", "").split("__")[-2]
                if "__" in pred.get("experiment_id", "") else "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "papers": papers_out,
            "aggregate": agg,
        }
        _save(cell, payload)
        print(f"  -> fluency {agg['fluency_mean']!r}, "
              f"acc-mistakes {agg['accuracy_mistakes_mean']!r}, "
              f"lex {agg['lexical_mistakes_mean']!r}, "
              f"gram {agg['grammatical_mistakes_mean']!r}  "
              f"({time.time()-ts:.1f}s)")

        notify_state["i"] = idx
        notify_state["last_cfg"] = label
        notifier.maybe_progress(i=idx, total=len(cells), last_cfg=label)

    notifier.send(
        f"LLM judge done ({args.judge_model})",
        f"All {len(cells)} cells judged. "
        f"Total: {time.time()-t0:.1f}s",
    )
    print(f"\nTotal: {time.time() - t0:.1f}s")
    print("Each cell now has a judge.json sibling to its predictions.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
