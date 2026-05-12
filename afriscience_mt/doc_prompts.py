"""Document-level prompt construction for closed-API reruns.

Mirrors the three templates and the few-shot format documented in
``_ACL2026__.../prompt-templates.tex`` (\\label{prompt:doc_templates}).
Used by ``scripts/rerun_openai_doc.py`` and ``scripts/rerun_google_doc.py``
to build identical doc-level prompts across both backends.

Why we rebuild prompts here (vs. replaying legacy prompts):
    The legacy GPT-4o / Gemini-1.5-Flash doc-level dumps only saved
    translation outputs, not the prompts that were sent. The
    sentence-level rerun scripts read prompts from a saved `prompt`
    column in the legacy TSVs; no analogous column exists for the
    legacy doc-level runs. So this module rebuilds prompts
    deterministically from the same template definitions and from
    domain-matched train docs (sampled with a paper-id-derived seed).
    Few-shot exemplar choice may differ from the legacy run, but the
    prompt format is identical.

Public API:
    LANG_DISPLAY                -- {short: display_name}
    DOC_TEMPLATES               -- {1, 2, 3: template_text}
    DocPromptBuilder(repo_root) -- stateful helper that caches train docs
        .build(...)             -- returns a single user-message string
"""
from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

# ---- language display strings ----------------------------------------------
LANG_DISPLAY: dict[str, str] = {
    "eng": "English",
    "amh": "Amharic",
    "hau": "Hausa",
    "lug": "Luganda",
    "nso": "Northern Sotho",
    "yor": "Yoruba",
    "zul": "isiZulu",
}

# ---- template strings (verbatim from prompt-templates.tex) ------------------
# {src_lang}, {tgt_lang}, {domain}, {src_document}, <<shots>> get filled in.
DOC_TEMPLATES: dict[int, str] = {
    1: (
        "Instruction:\n"
        "Translate the following document from {src_lang} to {tgt_lang}. "
        "Do not send the {src_lang} document back in the response; "
        "generate only the translation and nothing more.<<shots>>\n\n"
        "{src_lang} document: \n{src_document}\n\n"
        "{tgt_lang} document:\n"
    ),
    2: (
        "Instruction:\n"
        "Translate the following {domain} document from {src_lang} to {tgt_lang}. "
        "Do not send the {src_lang} document back in the response; "
        "generate only the translation and nothing more.<<shots>>\n\n"
        "{src_lang} document: \n{src_document}\n\n"
        "{tgt_lang} document:\n"
    ),
    3: (
        "Instruction:\n"
        "Please provide the {tgt_lang} translation for the following. "
        "Provide only the translation.<<shots>>\n\n"
        "{src_lang} document: \n{src_document}\n\n"
        "{tgt_lang} document:\n"
    ),
}

# ---- shot block (only inserted when n>0) -----------------------------------
_SHOT_HEADER = "\n\nUse the example{plural} below for consistency."
_SHOT_BLOCK = (
    "\n\nExample {idx}:\n"
    "{src_lang} document:\n{example_src}\n\n"
    "{tgt_lang} document:\n{example_tgt}"
)
_SHOT_FOOTER = "\n\nNow translate the following."


def _seed_for(pair: str, template_n: int, n_shots: int, paper_id: str) -> int:
    """Deterministic seed per (pair, template, n_shots, test_paper). The
    hash is stable across runs and across Python sessions (we hash the
    string explicitly rather than relying on Python's salted hash)."""
    key = f"{pair}|t{template_n}|s{n_shots}|p{paper_id}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


def _read_paper_file(p: Path) -> str:
    """Concatenate the lines of a doc .txt file into one whitespace-joined
    string. Matches ``convert_closed_llm_doc.read_doc_translation``."""
    return " ".join(line.strip() for line in p.read_text().splitlines()
                    if line.strip())


class DocPromptBuilder:
    """Builds doc-level prompts for one (src_lang -> tgt_lang) direction.

    Caches the train-doc index so repeated builds for the same pair don't
    re-scan the filesystem. One builder per language pair is the typical
    usage; you can also share a single instance across pairs if you want
    one big index in memory.
    """

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.docs_root = self.repo_root / "data" / "processed_files" / "docs"
        self.test_tsv_root = self.repo_root / "data" / "processed_files" / "test"
        # per-lang: {paper_id: {"domain": str, "text": str}}
        self._train_cache: dict[str, dict[str, dict]] = {}
        self._test_cache: dict[str, dict[str, dict]] = {}

    # ---- index loaders -------------------------------------------------

    def _load_split_index(self, lang: str, split: str) -> dict[str, dict]:
        """Build {paper_id: {"domain": str, "text": str}} for one (lang, split)."""
        cache = self._train_cache if split == "train" else self._test_cache
        if lang in cache:
            return cache[lang]
        # paper_id -> domain via the segment-level TSV (`category` column)
        tsv = self.repo_root / "data" / "processed_files" / split / f"{lang}.tsv"
        paper_domain: dict[str, str] = {}
        if tsv.is_file():
            with open(tsv) as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    pid = str(r["paper"])
                    if pid not in paper_domain:
                        paper_domain[pid] = r["category"]
        # paper_id -> doc text via the .txt files. Filenames are
        # "<paper_id>_<domain>.txt". We match by leading paper_id.
        idx: dict[str, dict] = {}
        doc_dir = self.docs_root / split / lang
        if doc_dir.is_dir():
            for p in doc_dir.iterdir():
                if p.suffix != ".txt":
                    continue
                stem = p.stem  # "212_engineering"
                pid, _, dom_from_name = stem.partition("_")
                if not pid.isdigit():
                    continue
                domain = paper_domain.get(pid, dom_from_name)
                idx[pid] = {"domain": domain, "text": _read_paper_file(p)}
        cache[lang] = idx
        return idx

    def list_test_papers(self, lang: str) -> list[tuple[str, str]]:
        """Return [(paper_id, domain), ...] for the test split in `lang`,
        ordered by numeric paper_id."""
        idx = self._load_split_index(lang, "test")
        return sorted(((pid, meta["domain"]) for pid, meta in idx.items()),
                      key=lambda kv: int(kv[0]))

    def test_doc(self, lang: str, paper_id: str) -> str:
        return self._load_split_index(lang, "test")[paper_id]["text"]

    # ---- shot block + prompt build -------------------------------------

    def _pick_exemplars(self, src_lang: str, tgt_lang: str, domain: str,
                        n_shots: int, seed: int) -> list[tuple[str, str]]:
        """Return up to n_shots (src_doc, tgt_doc) train-set pairs whose
        domain matches `domain` if possible, falling back to any train
        paper. The same seed produces the same picks across runs."""
        src_idx = self._load_split_index(src_lang, "train")
        tgt_idx = self._load_split_index(tgt_lang, "train")
        common = set(src_idx) & set(tgt_idx)
        domain_match = [pid for pid in common if src_idx[pid]["domain"] == domain]
        if len(domain_match) < n_shots:
            # Top up with non-matching train docs.
            extras = [pid for pid in common if pid not in set(domain_match)]
            rng_extra = random.Random(seed ^ 0xA5A5)
            rng_extra.shuffle(extras)
            domain_match = domain_match + extras
        rng = random.Random(seed)
        rng.shuffle(domain_match)
        picks = domain_match[:n_shots]
        return [(src_idx[pid]["text"], tgt_idx[pid]["text"]) for pid in picks]

    def build_prompt(self, *, src_lang: str, tgt_lang: str, template_n: int,
                     n_shots: int, paper_id: str, src_document: str,
                     domain: str) -> str:
        """Return the full user-message string ready for the API."""
        if template_n not in DOC_TEMPLATES:
            raise ValueError(f"unknown template {template_n}")
        pair = f"{src_lang}-{tgt_lang}"
        tmpl = DOC_TEMPLATES[template_n]

        # Render shot block (empty when n_shots == 0).
        if n_shots == 0:
            shots = ""
        else:
            seed = _seed_for(pair, template_n, n_shots, paper_id)
            exemplars = self._pick_exemplars(src_lang, tgt_lang, domain,
                                             n_shots, seed)
            parts = [_SHOT_HEADER.format(plural="s" if n_shots > 1 else "")]
            for i, (src_ex, tgt_ex) in enumerate(exemplars, start=1):
                parts.append(_SHOT_BLOCK.format(
                    idx=i,
                    src_lang=LANG_DISPLAY[src_lang],
                    tgt_lang=LANG_DISPLAY[tgt_lang],
                    example_src=src_ex,
                    example_tgt=tgt_ex,
                ))
            parts.append(_SHOT_FOOTER)
            shots = "".join(parts)

        return tmpl.format(
            src_lang=LANG_DISPLAY[src_lang],
            tgt_lang=LANG_DISPLAY[tgt_lang],
            domain=domain,
            src_document=src_document,
        ).replace("<<shots>>", shots)


__all__ = ["LANG_DISPLAY", "DOC_TEMPLATES", "DocPromptBuilder"]
