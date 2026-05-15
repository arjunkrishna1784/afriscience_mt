"""Build doc_errors_pie_{eng_xxx,xxx_eng}.pdf from LLM-judge accuracy data.

For each direction, draws one pie per closed API showing the proportion
of accuracy errors by category (Wrong Translation / Omission / Addition
/ Others), aggregated over all 6 common doc-level template
configurations and all 6 language pairs in that direction. Mistake
counts are summed across BOTH judges (``judge.json`` -- GPT-5.5; and
``judge_gemini.json`` -- Gemini-3.1-Flash-Lite); the n=... annotation
on each pie is the per-judge mean count. Categories are extracted from
the natural-language mistake strings; compound categories like
``Wrong Translation/Omission`` are bucketed into their first listed
category.

Output overwrites the figures referenced by the appendix:
    figures/doc_errors_pie_eng_xxx.pdf
    figures/doc_errors_pie_xxx_eng.pdf
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DOC_DIR = ROOT / "experiments" / "llm" / "closed" / "doc"
FIG_DIR = (
    ROOT / "_ACL2026__Decolonizing_Science_in_Africa_through_Text_Translation_"
    / "figures"
)

MODELS = [
    ("Gemini-1.5", "gemini_15_flash"),
    ("Gemini-3.1", "gemini_31_flash_lite"),
    ("GPT-4o",     "gpt4o"),
    ("GPT-5.4",    "gpt54"),
]
COMMON_STRATEGIES = [
    "doc_0shot_template1", "doc_1shot_template1",
    "doc_0shot_template2", "doc_1shot_template2",
    "doc_0shot_template3", "doc_1shot_template3",
]
LANGS = ["amh", "hau", "lug", "nso", "yor", "zul"]
CATEGORIES = ["Wrong Translation", "Omission", "Addition", "Others"]
COLORS = {
    "Wrong Translation": "#3a7ca5",
    "Omission":          "#f4a261",
    "Addition":          "#e76f51",
    "Others":            "#a8a8a8",
}


def categorise(mistake: str) -> str:
    """Pull the first explicit category prefix from the mistake string."""
    m = re.match(r"^([^:]+):", mistake.strip())
    if not m:
        return "Others"
    head = m.group(1).strip()
    # Compound categories like "Wrong Translation/Omission" -> first listed.
    head = head.split("/")[0].strip()
    if head in CATEGORIES:
        return head
    return "Others"


JUDGE_FILES = ("judge.json", "judge_gemini.json")


def counts_for(model_short: str, direction: str) -> tuple[Counter, int]:
    """Aggregate mistake counts across all common strategies + 6 langs in
    this direction, summing over BOTH judges. Returns (counter,
    n_judges_present) so the caller can report per-judge mean counts."""
    pair_filter = (
        (lambda p: p.startswith("eng-")) if direction == "e2x"
        else (lambda p: p.endswith("-eng"))
    )
    cnt = Counter()
    n_judge_files_seen = 0
    for strat in COMMON_STRATEGIES:
        strat_dir = DOC_DIR / model_short / strat
        if not strat_dir.is_dir():
            continue
        for pair_dir in strat_dir.iterdir():
            if not pair_filter(pair_dir.name):
                continue
            for jfile in JUDGE_FILES:
                jf = pair_dir / jfile
                if not jf.exists():
                    continue
                try:
                    d = json.loads(jf.read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                n_judge_files_seen += 1
                for paper in d.get("papers", []):
                    for mistake in paper.get("accuracy", {}).get("mistakes", []):
                        if isinstance(mistake, str):
                            cnt[categorise(mistake)] += 1
    return cnt, n_judge_files_seen


def render(direction: str, out_path: Path) -> None:
    # Wider per-pie cell + extra bottom margin reserved for the shared
    # legend so per-slice labels never need to crowd the slice edges.
    fig, axes = plt.subplots(1, len(MODELS),
                             figsize=(4.6 * len(MODELS), 4.8))
    for ax, (disp, ms) in zip(axes, MODELS):
        cnt, n_judge_files = counts_for(ms, direction)
        sizes = [cnt.get(c, 0) for c in CATEGORIES]
        total = sum(sizes)
        # n_judge_files = (judge files seen) = (cells * judges_per_cell);
        # avg per judge = total / (judges_per_cell * n_cells / n_cells) =
        # total / (n_judge_files / n_cells), but more simply: total /
        # number_of_judges_seen_per_cell. With both judges always present
        # the per-judge mean count is total / 2.
        per_judge_mean = round(total / max(1, len(JUDGE_FILES)))
        title = f"{disp}  (n={per_judge_mean})"
        if total == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
            ax.set_axis_off()
            continue
        # Show only present categories in this pie; category name lookup
        # comes from the shared legend so no inline labels overlap.
        present = [(c, s) for c, s in zip(CATEGORIES, sizes) if s > 0]
        psizes  = [s for _, s in present]
        pcolors = [COLORS[c] for c, _ in present]
        # Hide the auto-percentage on slices smaller than 4% to avoid
        # label collisions on the rim of tiny slices.
        def autopct(pct):
            return f"{pct:.1f}%" if pct >= 4 else ""
        ax.pie(psizes, colors=pcolors,
               autopct=autopct, pctdistance=0.72,
               startangle=90, counterclock=False,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5},
               textprops={"fontsize": 11, "color": "white",
                          "fontweight": "bold"})
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    # Shared legend (one per figure) below all pies; no per-slice text.
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c])
               for c in CATEGORIES]
    fig.legend(handles, CATEGORIES, loc="lower center",
               ncol=len(CATEGORIES), frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=12)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    for direction, suffix in (("e2x", "eng_xxx"), ("x2e", "xxx_eng")):
        out = FIG_DIR / f"doc_errors_pie_{suffix}.pdf"
        render(direction, out)
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
