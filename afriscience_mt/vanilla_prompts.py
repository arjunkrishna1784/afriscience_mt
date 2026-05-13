"""Vanilla (open-LLM-comparable) prompt builders for closed-API runs.

The closed-API rerun scripts (rerun_openai_doc.py, rerun_google_doc.py,
rerun_gpt4o_templated.py, rerun_gemini_newer.py) all use the *paper*
prompt templates from ``prompt-templates.tex``. The open-source LLM runs,
in contrast, use the *yaml* templates from
``configs/prompts/sentence_templates.yaml`` and
``configs/prompts/document_templates.yaml`` -- under the names
``explicit`` (sentence, 0-shot), ``few_shot`` (sentence, 5-shot),
``basic`` (document, 0-shot), and again ``basic`` (document, 1-shot).

For a direct closed-vs-open comparison we need the closed APIs to emit
predictions on the *same vanilla prompts* the open LLMs saw. This
module renders those vanilla prompts directly so the new
rerun_openai_vanilla.py / rerun_google_vanilla.py scripts can fire
them at gpt-5.4 / gemini-3.1-flash-lite without going through the
existing yaml->prompt_builder->call_open_llm pipeline.

Sentence vanilla (mirrors configs/prompts/sentence_templates.yaml):
    0shot: ``explicit`` template (id=2)
    5shot: ``few_shot`` template (id=5) + 5 random train exemplars

Document vanilla (mirrors configs/prompts/document_templates.yaml):
    doc_0shot: ``basic`` template
    doc_1shot: ``basic`` template + 1 random train doc exemplar

Few-shot exemplar selection is seeded deterministically per
(direction, n_shots, sample_index) so re-runs reproduce identical
prompts.
"""
from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

# ---- language full-name lookups (verbatim from sentence_templates.yaml) -----
LANG_FULL: dict[str, str] = {
    "eng": "English",
    "amh": "Amharic",
    "hau": "Hausa",
    "lug": "Luganda",
    "nso": "Northern Sotho (Sepedi)",
    "yor": "Yorùbá",
    "zul": "Zulu (isiZulu)",
}

# ---- sentence-level vanilla templates --------------------------------------
# Verbatim from configs/prompts/sentence_templates.yaml: templates.explicit
# and templates.few_shot. Newlines preserved so the rendered prompt is
# byte-identical to what PromptBuilder would produce.
_SENT_EXPLICIT_TMPL = (
    "Translate from {src_full} to {tgt_full}:\n\n"
    "{src_full}: {src_text}\n"
    "{tgt_full}:\n"
)
_SENT_FEW_SHOT_TMPL = (
    "Translate {src_full} scientific texts to {tgt_full}.\n\n"
    "Examples:\n"
    "{examples}\n\n"
    "Now translate:\n"
    "{src_full}: {src_text}\n"
    "{tgt_full}:\n"
)
_SENT_FEW_SHOT_EXAMPLE_FORMAT = "{src_full}: {src}\n{tgt_full}: {tgt}"

# ---- document-level vanilla templates --------------------------------------
# Verbatim from configs/prompts/document_templates.yaml: templates.basic
# (used for doc_0shot AND doc_1shot in the open-LLM pipeline -- the only
# difference is whether one example block is prepended).
_DOC_BASIC_TMPL = (
    "Translate the following {src_lang} scientific document to {tgt_lang}.\n"
    "Maintain the structure, coherence, and scientific terminology throughout.\n\n"
    "Document:\n"
    "{src_document}\n\n"
    "Translation:\n"
)
_DOC_EXEMPLAR_BLOCK = (
    "Example:\n"
    "{src_lang} document:\n"
    "{example_src}\n\n"
    "{tgt_lang} document:\n"
    "{example_tgt}\n\n"
    "Now translate:\n"
)


def _seed_for(*parts) -> int:
    """Deterministic seed derived from the SHA-256 of the joined parts."""
    key = "|".join(str(p) for p in parts)
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big")


# ---- sentence-level builders -----------------------------------------------

def _read_test_column(tsv_path: Path, col: str = "text") -> list[str]:
    with open(tsv_path) as f:
        return [r[col] for r in csv.DictReader(f, delimiter="\t")]


class VanillaSentBuilder:
    """Builds vanilla sentence-level prompts for one (src, tgt) pair.
    Loads test/train columns lazily; cache survives the lifetime of the
    instance so a single instance can serve many config calls."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.test_dir = self.repo_root / "data" / "processed_files" / "test"
        self.train_dir = self.repo_root / "data" / "processed_files" / "train"
        self._test_cache: dict[str, list[str]] = {}
        self._train_cache: dict[str, list[str]] = {}

    def _test(self, lang: str) -> list[str]:
        if lang not in self._test_cache:
            self._test_cache[lang] = _read_test_column(self.test_dir / f"{lang}.tsv")
        return self._test_cache[lang]

    def _train(self, lang: str) -> list[str]:
        if lang not in self._train_cache:
            self._train_cache[lang] = _read_test_column(self.train_dir / f"{lang}.tsv")
        return self._train_cache[lang]

    def n_test(self, src: str, tgt: str) -> int:
        return min(len(self._test(src)), len(self._test(tgt)))

    def build_prompts(self, src: str, tgt: str, n_shots: int) -> list[str]:
        """Return one prompt per test-set sentence (970 prompts)."""
        src_full = LANG_FULL[src]
        tgt_full = LANG_FULL[tgt]
        srcs = self._test(src)
        if n_shots == 0:
            return [
                _SENT_EXPLICIT_TMPL.format(
                    src_full=src_full, tgt_full=tgt_full, src_text=s.strip()
                )
                for s in srcs
            ]
        # Few-shot: deterministic per-row exemplars from train.
        train_src = self._train(src)
        train_tgt = self._train(tgt)
        n_train = min(len(train_src), len(train_tgt))
        prompts = []
        for idx, s in enumerate(srcs):
            seed = _seed_for("sent", src, tgt, n_shots, idx)
            rng = random.Random(seed)
            picks = rng.sample(range(n_train), k=min(n_shots, n_train))
            ex_block = "\n".join(
                _SENT_FEW_SHOT_EXAMPLE_FORMAT.format(
                    src_full=src_full, tgt_full=tgt_full,
                    src=train_src[i].strip(), tgt=train_tgt[i].strip(),
                )
                for i in picks
            )
            prompts.append(_SENT_FEW_SHOT_TMPL.format(
                src_full=src_full, tgt_full=tgt_full,
                examples=ex_block, src_text=s.strip(),
            ))
        return prompts


# ---- document-level builders -----------------------------------------------

def _read_paper_text(p: Path) -> str:
    return " ".join(line.strip() for line in p.read_text().splitlines()
                    if line.strip())


class VanillaDocBuilder:
    """Builds vanilla document-level prompts for one (src, tgt) pair.
    Mirrors DocPromptBuilder's index loader but yields *yaml-template*
    prompts instead of paper-template prompts."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.docs_root = self.repo_root / "data" / "processed_files" / "docs"
        self._train_cache: dict[str, dict[str, dict]] = {}
        self._test_cache: dict[str, dict[str, dict]] = {}

    def _load_index(self, lang: str, split: str) -> dict[str, dict]:
        cache = self._train_cache if split == "train" else self._test_cache
        if lang in cache:
            return cache[lang]
        # paper_id -> domain via the segment-level TSV
        tsv = self.repo_root / "data" / "processed_files" / split / f"{lang}.tsv"
        paper_domain: dict[str, str] = {}
        if tsv.is_file():
            with open(tsv) as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    pid = str(r["paper"])
                    if pid not in paper_domain:
                        paper_domain[pid] = r["category"]
        # paper_id -> doc text
        idx: dict[str, dict] = {}
        doc_dir = self.docs_root / split / lang
        if doc_dir.is_dir():
            for p in doc_dir.iterdir():
                if p.suffix != ".txt":
                    continue
                pid, _, dom_from_name = p.stem.partition("_")
                if not pid.isdigit():
                    continue
                domain = paper_domain.get(pid, dom_from_name)
                idx[pid] = {"domain": domain, "text": _read_paper_text(p)}
        cache[lang] = idx
        return idx

    def list_test_papers(self, lang: str) -> list[tuple[str, str]]:
        idx = self._load_index(lang, "test")
        return sorted(((pid, meta["domain"]) for pid, meta in idx.items()),
                      key=lambda kv: int(kv[0]))

    def test_doc(self, lang: str, paper_id: str) -> str:
        return self._load_index(lang, "test")[paper_id]["text"]

    def _pick_exemplar(self, src: str, tgt: str, paper_id: str,
                       n_shots: int) -> tuple[str, str] | None:
        if n_shots <= 0:
            return None
        train_src = self._load_index(src, "train")
        train_tgt = self._load_index(tgt, "train")
        common = sorted(set(train_src) & set(train_tgt), key=lambda x: int(x))
        if not common:
            return None
        seed = _seed_for("doc", src, tgt, n_shots, paper_id)
        rng = random.Random(seed)
        # 1 exemplar by default (open-LLM doc_1shot uses a single example).
        pick = rng.sample(common, k=min(n_shots, len(common)))[0]
        return train_src[pick]["text"], train_tgt[pick]["text"]

    def build_prompt(self, *, src: str, tgt: str, paper_id: str,
                     src_document: str, n_shots: int) -> str:
        src_lang = LANG_FULL[src]
        tgt_lang = LANG_FULL[tgt]
        body = _DOC_BASIC_TMPL.format(
            src_lang=src_lang, tgt_lang=tgt_lang,
            src_document=src_document.strip(),
        )
        if n_shots <= 0:
            return body
        ex = self._pick_exemplar(src, tgt, paper_id, n_shots)
        if ex is None:
            return body
        ex_src, ex_tgt = ex
        block = _DOC_EXEMPLAR_BLOCK.format(
            src_lang=src_lang, tgt_lang=tgt_lang,
            example_src=ex_src.strip(), example_tgt=ex_tgt.strip(),
        )
        return block + body


__all__ = ["LANG_FULL", "VanillaSentBuilder", "VanillaDocBuilder"]
