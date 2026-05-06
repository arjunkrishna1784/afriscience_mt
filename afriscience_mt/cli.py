"""Single CLI entry point: ``afriscience-mt <subcommand> [options]``.

Subcommands:
    train      run MT training (modes: seq2seq, lora, multilingual)
    evaluate   run/re-run evaluation against trained or base models
    aggregate  collect results.json across experiments into one CSV/JSON
    table      generate paper-ready tables (markdown, LaTeX) from aggregations
    plot       generate paper-ready plots (heatmaps, bars, radar) from aggregations
    migrate    one-time: harmonize legacy results into canonical schema
"""

from __future__ import annotations

import argparse
import sys

SUBCOMMANDS = {
    "train":     ("afriscience_mt.commands.train",     "Train an MT model (seq2seq | lora | multilingual)"),
    "evaluate":  ("afriscience_mt.commands.evaluate",  "Evaluate predictions; supports re-scoring with new metrics"),
    "aggregate": ("afriscience_mt.commands.aggregate", "Collect canonical results into a CSV/JSON summary"),
    "table":     ("afriscience_mt.commands.table",     "Render paper-ready markdown / LaTeX tables"),
    "plot":      ("afriscience_mt.commands.plot",      "Render paper-ready figures (heatmap, bars, radar)"),
    "migrate":   ("afriscience_mt.commands.migrate",   "Migrate legacy result files to the canonical schema"),
}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="afriscience-mt",
        description="AfriScience-MT: machine-translation framework for African scientific text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_format_subcommands(),
    )
    parser.add_argument("subcommand", nargs="?", choices=list(SUBCOMMANDS),
                        help="subcommand to run")
    parser.add_argument("rest", nargs=argparse.REMAINDER,
                        help="arguments forwarded to the subcommand")

    args = parser.parse_args(argv)
    if not args.subcommand:
        parser.print_help()
        return 0

    module_path, _ = SUBCOMMANDS[args.subcommand]
    import importlib
    module = importlib.import_module(module_path)
    return module.main(args.rest)


def _format_subcommands() -> str:
    lines = ["", "Subcommands:"]
    for name, (_, desc) in SUBCOMMANDS.items():
        lines.append(f"  {name:<10} {desc}")
    lines += ["",
              "Run `afriscience-mt <subcommand> --help` for subcommand-specific options.",
              "Set DCSMT_DATA_DIR / DCSMT_EXPERIMENTS_DIR / DCSMT_HF_CACHE to override paths."]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
