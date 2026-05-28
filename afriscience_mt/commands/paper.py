"""Build paper-ready tables and figures from `experiments/summary.csv`.

Usage:
    afriscience-mt paper <target>

Targets:
    tables      LaTeX tables (sentence, doc, ablations); chrF + COMET twins
    assets      orchestrator: rebuilds every paper table and asset
    plots       per-domain spider charts and other paper plots
    doc-errors  document-level accuracy-error pie charts
    balance     domain-balance heatmap and delta-by-size figures
    figure4     standalone Figure 4
"""

from __future__ import annotations

import argparse
import importlib
import sys

TARGETS: dict[str, str] = {
    "tables":     "afriscience_mt.paper.tables",
    "assets":     "afriscience_mt.paper.assets",
    "plots":      "afriscience_mt.paper.plots",
    "doc-errors": "afriscience_mt.paper.doc_errors",
    "balance":    "afriscience_mt.paper.balance",
    "figure4":    "afriscience_mt.paper.figure4",
}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="afriscience-mt paper",
        description="Build paper-ready tables and figures from "
                    "experiments/summary.csv.",
    )
    parser.add_argument(
        "target", choices=list(TARGETS),
        help="which paper asset to build",
    )
    args = parser.parse_args(argv)

    module = importlib.import_module(TARGETS[args.target])
    result = module.main() if hasattr(module, "main") else 0
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
