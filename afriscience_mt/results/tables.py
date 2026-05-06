"""Table generation for results presentation."""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .manager import ExperimentResult
from .aggregator import ResultsAggregator
from ..utils.language_utils import get_language_name
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TableGenerator:
    """Generator for results tables in various formats."""

    def __init__(
        self,
        results: List[ExperimentResult],
        metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
    ):
        """
        Initialize table generator.

        Args:
            results: Experiment results
            metrics: Metrics to include in tables
        """
        self.results = results
        self.metrics = metrics
        self.aggregator = ResultsAggregator(results)

    def generate_model_comparison_table(
        self,
        format: str = "markdown",
        precision: int = 2,
    ) -> str:
        """
        Generate model comparison table.

        Args:
            format: Output format ("markdown", "latex", "csv")
            precision: Decimal places for metrics

        Returns:
            Formatted table string
        """
        by_model = self.aggregator.aggregate_by_model(self.metrics)

        # Build table data
        headers = ["Model"] + [m.upper() for m in self.metrics]
        rows = []

        for model, data in sorted(by_model.items()):
            row = [model]
            for metric in self.metrics:
                metric_data = data.get(metric, {})
                mean = metric_data.get("mean")
                std = metric_data.get("std", 0)
                if mean is not None:
                    row.append(f"{mean:.{precision}f} ± {std:.{precision}f}")
                else:
                    row.append("-")
            rows.append(row)

        return self._format_table(headers, rows, format)

    def generate_language_table(
        self,
        format: str = "markdown",
        precision: int = 2,
    ) -> str:
        """
        Generate results table by language.

        Args:
            format: Output format
            precision: Decimal places

        Returns:
            Formatted table string
        """
        by_lang = self.aggregator.aggregate_by_language(self.metrics)

        headers = ["Language"] + [m.upper() for m in self.metrics]
        rows = []

        for lang, data in sorted(by_lang.items()):
            lang_name = data.get("language_name", lang)
            row = [lang_name]

            for metric in self.metrics:
                metric_data = data.get(metric, {})
                mean = metric_data.get("mean")
                if mean is not None:
                    row.append(f"{mean:.{precision}f}")
                else:
                    row.append("-")
            rows.append(row)

        return self._format_table(headers, rows, format)

    def generate_full_results_table(
        self,
        format: str = "markdown",
        precision: int = 2,
        include_config: bool = False,
    ) -> str:
        """
        Generate full results table with all experiments.

        Args:
            format: Output format
            precision: Decimal places
            include_config: Include configuration details

        Returns:
            Formatted table string
        """
        headers = ["Experiment", "Model", "Lang Pair"]
        headers += [m.upper() for m in self.metrics]

        if include_config:
            headers.append("Config")

        rows = []
        for result in self.results:
            row = [
                result.experiment_name,
                result.model_name or "-",
                result.lang_pair or "-",
            ]

            for metric in self.metrics:
                value = result.metrics.get(metric)
                if value is not None:
                    row.append(f"{value:.{precision}f}")
                else:
                    row.append("-")

            if include_config:
                config_str = str(result.config.get("key_params", {}))[:50]
                row.append(config_str)

            rows.append(row)

        return self._format_table(headers, rows, format)

    def generate_domain_heatmap_data(self) -> Dict[str, Any]:
        """
        Generate data for domain performance heatmap.

        Returns:
            Dict with heatmap data
        """
        by_domain = self.aggregator.aggregate_by_domain(self.metrics)

        domains = list(by_domain.keys())
        data = {metric: [] for metric in self.metrics}

        for domain in domains:
            for metric in self.metrics:
                value = by_domain[domain].get(metric, {}).get("mean", 0)
                data[metric].append(value)

        return {
            "domains": domains,
            "metrics": self.metrics,
            "values": data,
        }

    def _format_table(
        self,
        headers: List[str],
        rows: List[List[str]],
        format: str,
    ) -> str:
        """Format table in specified format."""
        if format == "markdown":
            return self._to_markdown(headers, rows)
        elif format == "latex":
            return self._to_latex(headers, rows)
        elif format == "csv":
            return self._to_csv(headers, rows)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert to markdown table."""
        lines = []

        # Header
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        # Rows
        for row in rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)

    def _to_latex(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert to LaTeX table."""
        lines = []

        # Table setup
        col_spec = "l" + "c" * (len(headers) - 1)
        lines.append("\\begin{table}[ht]")
        lines.append("\\centering")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header
        lines.append(" & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\")
        lines.append("\\midrule")

        # Rows
        for row in rows:
            lines.append(" & ".join(str(cell) for cell in row) + " \\\\")

        # Close table
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Results}")
        lines.append("\\label{tab:results}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    def _to_csv(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert to CSV."""
        lines = [",".join(headers)]
        for row in rows:
            lines.append(",".join(str(cell) for cell in row))
        return "\n".join(lines)


def generate_results_table(
    results: List[ExperimentResult],
    table_type: str = "model_comparison",
    format: str = "markdown",
    metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Convenience function to generate a results table.

    Args:
        results: Experiment results
        table_type: Type of table
        format: Output format
        metrics: Metrics to include
        output_path: Optional path to save table

    Returns:
        Table string
    """
    generator = TableGenerator(results, metrics)

    if table_type == "model_comparison":
        table = generator.generate_model_comparison_table(format)
    elif table_type == "language":
        table = generator.generate_language_table(format)
    elif table_type == "full":
        table = generator.generate_full_results_table(format)
    else:
        raise ValueError(f"Unknown table type: {table_type}")

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(table)
        logger.info(f"Table saved to {output_path}")

    return table


def generate_latex_table(
    results: List[ExperimentResult],
    table_type: str = "model_comparison",
    metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """Convenience function to generate LaTeX table."""
    return generate_results_table(
        results, table_type, "latex", metrics, output_path
    )


def generate_comparison_table(
    results1: List[ExperimentResult],
    results2: List[ExperimentResult],
    label1: str = "System 1",
    label2: str = "System 2",
    metrics: List[str] = ["bleu", "chrf", "ssa_comet"],
    format: str = "markdown",
) -> str:
    """
    Generate comparison table between two sets of results.

    Args:
        results1: First set of results
        results2: Second set of results
        label1: Label for first system
        label2: Label for second system
        metrics: Metrics to compare
        format: Output format

    Returns:
        Comparison table
    """
    agg1 = ResultsAggregator(results1)
    agg2 = ResultsAggregator(results2)

    by_lang1 = agg1.aggregate_by_language(metrics)
    by_lang2 = agg2.aggregate_by_language(metrics)

    all_langs = set(by_lang1.keys()) | set(by_lang2.keys())

    headers = ["Language"]
    for metric in metrics:
        headers.extend([f"{label1} {metric.upper()}", f"{label2} {metric.upper()}", "Δ"])

    rows = []
    for lang in sorted(all_langs):
        row = [get_language_name(lang)]

        for metric in metrics:
            val1 = by_lang1.get(lang, {}).get(metric, {}).get("mean")
            val2 = by_lang2.get(lang, {}).get(metric, {}).get("mean")

            row.append(f"{val1:.2f}" if val1 else "-")
            row.append(f"{val2:.2f}" if val2 else "-")

            if val1 and val2:
                diff = val2 - val1
                sign = "+" if diff > 0 else ""
                row.append(f"{sign}{diff:.2f}")
            else:
                row.append("-")

        rows.append(row)

    generator = TableGenerator([], metrics)
    return generator._format_table(headers, rows, format)
