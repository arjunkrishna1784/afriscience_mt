"""One-shot restructure of experiments/ into the new arch-first layout.

OLD                                                  NEW
zero_shot/<model>/<pair>/                          → seq2seq/zero_shot/<model>/<pair>/
seq2seq_finetune/<model>/<pair>/                   → seq2seq/finetune/<model>/dcs/<pair>/
seq2seq_finetune/<model>/mafand/<pair>/            → seq2seq/finetune/<model>/mafand/<pair>/
seq2seq_finetune/<model>/dcs_mafand/<pair>/        → seq2seq/finetune/<model>/mafand_pp/<pair>/
llm_zero_shot/<open>/<pair>/                       → llm/open/zero_shot/<model>/<pair>/
llm_zero_shot/<closed>/<pair>/                     → llm/closed/{t0|t06}/zero_shot/<canonical>/<pair>/
llm_icl/<open>/icl_<N>shot[_semantic]/<pair>/      → llm/open/icl/<model>/<N>shot[_semantic]/<pair>/
llm_icl/<closed>/icl_<N>shot[_semantic]/<pair>/    → llm/closed/{t0|t06}/icl/<canonical>/<N>shot[_semantic]/<pair>/
llm_doc/<open>/doc_<N>shot[_legacy]/<pair>/        → llm/open/doc/<model>/<N>shot/<pair>/
llm_doc/<closed>/doc_<N>shot[_legacy]/<pair>/      → llm/closed/t0/doc/<canonical>/<N>shot/<pair>/
lora_finetune/<model>/r<N>/<pair>/                 → llm/open/finetune/<model>/r<N>/<pair>/

Each results.json gets:
  - metadata.original_path  preserved (chained if already migrated once)
  - metadata.is_ablation    True iff dataset==mafand_pp OR closed-LLM with t06
  - dataset                 dcs_mafand renamed to mafand_pp
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments"

CLOSED_MODELS = {"gpt4o", "gemini_15_flash"}
OPEN_LLM_MODELS = {
    "afriquellama_8b", "afriqueqwen_8b", "gemma2_9b_it", "llama3_8b",
    "tiny_aya_earth", "tiny_aya_global", "translategemma_12b",
}
SEQ2SEQ_MODELS = {
    "m2m100_418m", "m2m100_1_2b",
    "nllb_200_distilled_600m", "nllb_200_1_3b",
}


def parse_closed_model(model_dir: str) -> tuple[str, str, bool] | None:
    """Returns (canonical, t0|t06, is_alt) or None if not a closed model."""
    for c in CLOSED_MODELS:
        if model_dir == c:
            return c, "t0", False
        if model_dir == f"{c}_t06":
            return c, "t06", False
        if model_dir == f"{c}_temp_0_6":
            return c, "t06", True
    return None


def normalize_strategy(s: str) -> str:
    """icl_5shot -> 5shot, doc_0shot_legacy -> 0shot, etc."""
    s = s.removeprefix("icl_").removeprefix("doc_")
    s = s.removesuffix("_legacy")
    return s


def map_old_to_new(old_path: Path) -> Path | None:
    """Map an experiment dir under experiments/ to its new path. Returns None if unmappable."""
    parts = old_path.relative_to(EXP).parts
    top = parts[0]

    if top == "zero_shot":
        # zero_shot/<model>/<pair>/
        _, model, pair = parts
        return EXP / "seq2seq" / "zero_shot" / model / pair

    if top == "seq2seq_finetune":
        if len(parts) == 3:
            _, model, pair = parts
            return EXP / "seq2seq" / "finetune" / model / "dcs" / pair
        if len(parts) == 4:
            _, model, dataset, pair = parts
            ds_new = "mafand_pp" if dataset == "dcs_mafand" else dataset
            return EXP / "seq2seq" / "finetune" / model / ds_new / pair
        return None

    if top == "lora_finetune":
        # lora_finetune/<model>/r<N>/<pair>/
        _, model, rank, pair = parts
        return EXP / "llm" / "open" / "finetune" / model / rank / pair

    if top == "llm_zero_shot":
        _, model, pair = parts
        cm = parse_closed_model(model)
        if cm is None:
            return EXP / "llm" / "open" / "zero_shot" / model / pair
        canonical, temp, is_alt = cm
        canonical_dir = canonical + "__alt" if is_alt else canonical
        return EXP / "llm" / "closed" / temp / "zero_shot" / canonical_dir / pair

    if top == "llm_icl":
        # llm_icl/<model>/<strategy>/<pair>/
        _, model, strategy, pair = parts
        strat_new = normalize_strategy(strategy)
        cm = parse_closed_model(model)
        if cm is None:
            return EXP / "llm" / "open" / "icl" / model / strat_new / pair
        canonical, temp, is_alt = cm
        canonical_dir = canonical + "__alt" if is_alt else canonical
        return EXP / "llm" / "closed" / temp / "icl" / canonical_dir / strat_new / pair

    if top == "llm_doc":
        _, model, strategy, pair = parts
        strat_new = normalize_strategy(strategy)
        cm = parse_closed_model(model)
        if cm is None:
            return EXP / "llm" / "open" / "doc" / model / strat_new / pair
        canonical, temp, is_alt = cm  # all closed-LLM doc runs are t0 in current data
        canonical_dir = canonical + "__alt" if is_alt else canonical
        return EXP / "llm" / "closed" / temp / "doc" / canonical_dir / strat_new / pair

    return None


def is_ablation_for(parts_under_exp: tuple, dataset: str) -> bool:
    if dataset == "mafand_pp":
        return True
    if (len(parts_under_exp) >= 3
            and parts_under_exp[0] == "llm"
            and parts_under_exp[1] == "closed"
            and parts_under_exp[2] == "t06"):
        return True
    return False


def update_results_json(path: Path, old_rel: str, new_rel: str) -> None:
    with open(path) as f:
        data = json.load(f)

    if data.get("dataset") == "dcs_mafand":
        data["dataset"] = "mafand_pp"

    md = data.setdefault("metadata", {})
    prev = md.get("original_path")
    if prev and prev != old_rel:
        md["original_path_chain"] = (md.get("original_path_chain") or []) + [prev]
    md["original_path"] = old_rel
    md["restructured_to"] = new_rel
    # parts under experiments/ — first element is "experiments" so skip it
    new_parts = Path(new_rel).parts
    if new_parts and new_parts[0] == "experiments":
        new_parts = new_parts[1:]
    md["is_ablation"] = is_ablation_for(new_parts, data.get("dataset", "dcs"))

    # closed-LLM canonicalization: strip temp suffix from model.short_name
    # path shape: llm/closed/{t0|t06}/{zero_shot|icl|doc}/<model>/...
    if len(new_parts) > 4 and new_parts[0] == "llm" and new_parts[1] == "closed":
        temp = new_parts[2]
        canonical_model = new_parts[4]
        m = data.get("model", {})
        m["short_name"] = canonical_model
        md["temp_setting"] = temp

    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def git_mv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise RuntimeError(f"destination already exists: {dst}")
    # Try git mv first (so history is tracked); fall back to shutil.move.
    r = subprocess.run(["git", "mv", str(src), str(dst)],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        # not in git index yet — plain move
        shutil.move(str(src), str(dst))


def find_experiment_dirs() -> list[Path]:
    """An experiment dir is the parent of a results.json (we ignore aggregate
    files at upper levels, which we don't have in this tree)."""
    return sorted({p.parent for p in EXP.rglob("results.json")})


def main(dry_run: bool = False) -> int:
    moves: list[tuple[Path, Path]] = []
    skipped: list[Path] = []
    collisions: list[tuple[Path, Path]] = []

    for old in find_experiment_dirs():
        # If already under the new top-level layout, skip
        rel = old.relative_to(EXP).parts
        if rel and rel[0] in {"seq2seq", "llm"}:
            continue
        new = map_old_to_new(old)
        if new is None:
            skipped.append(old)
            continue
        if new.exists():
            collisions.append((old, new))
            continue
        moves.append((old, new))

    print(f"planned moves: {len(moves)}")
    print(f"skipped (unmapped): {len(skipped)}")
    print(f"collisions: {len(collisions)}")
    for s in skipped[:10]:
        print("  skip:", s)
    for old, new in collisions[:10]:
        print("  collision:", old, "->", new)

    if dry_run:
        for old, new in moves[:20]:
            print("  ", old.relative_to(REPO), "->", new.relative_to(REPO))
        return 0

    for old, new in moves:
        git_mv(old, new)
        results = new / "results.json"
        if results.exists():
            update_results_json(
                results,
                old_rel=str(old.relative_to(REPO)),
                new_rel=str(new.relative_to(REPO)),
            )

    # Clean up now-empty old top-level dirs
    for old_top in ["zero_shot", "seq2seq_finetune", "lora_finetune",
                    "llm_zero_shot", "llm_icl", "llm_doc"]:
        d = EXP / old_top
        if d.exists():
            try:
                # remove only if empty (recursively)
                empty = not any(d.rglob("*"))
                if empty:
                    shutil.rmtree(d)
            except Exception:
                pass

    print(f"done: moved {len(moves)} experiment dirs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(dry_run="--dry-run" in sys.argv))
