"""`afriscience-mt train` — unified training entry.

Modes:
  --mode mafand    fine-tune NLLB/M2M100 on MAFAND / DCS / DCS+MAFAND data
                   (the script that fills the audit's MAFAND-comparison gap)
  --mode seq2seq   single-pair seq2seq fine-tune on the configured dataset
  --mode lora      LoRA fine-tuning of an LLM on the configured dataset
  --mode multi     N-to-N multilingual fine-tuning

For now `--mode mafand` runs the self-contained
``scripts/train_mafand_comparison.py``. Other modes wrap the existing trainers
in ``afriscience_mt.training`` (TODO: refactor those into one Trainer class).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="afriscience-mt train")
    p.add_argument("--mode", required=True,
                   choices=["mafand", "seq2seq", "lora", "multi"])
    args, rest = p.parse_known_args(argv)

    if args.mode == "mafand":
        from scripts import train_mafand_comparison
        return train_mafand_comparison.main(rest)
    if args.mode == "seq2seq":
        from afriscience_mt.training.seq2seq_runner import main as run
        return run(rest)
    if args.mode == "lora":
        from afriscience_mt.training.lora_runner import main as run
        return run(rest)
    if args.mode == "multi":
        from afriscience_mt.training.multilingual import main as run
        return run(rest)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
