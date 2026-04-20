"""Rich console table for displaying aggregated evaluation results."""

import pandas as pd
from rich.console import Console
from rich.table import Table


def render_summary_table(df: pd.DataFrame, use_bertscore: bool = True) -> None:
    """
    Aggregate per-model metrics across all questions and display as a
    formatted table sorted by BERTScore F1 (or ROUGE-L if BERTScore skipped).
    """
    numeric_cols = ["rouge1", "rouge2", "rougeL", "bleu", "f1", "latency_seconds"]
    if use_bertscore:
        numeric_cols += ["bertscore_p", "bertscore_r", "bertscore_f1"]

    valid_df = df[df["error"].isna()].copy()
    summary = (
        valid_df.groupby("model_name")[numeric_cols]
        .mean()
        .reset_index()
    )

    sort_col = "bertscore_f1" if use_bertscore else "rougeL"
    summary = summary.sort_values(sort_col, ascending=False)

    console = Console()
    table = Table(
        title="LLM Evaluation Results",
        show_header=True,
        header_style="bold cyan",
        border_style="bright_black",
        show_lines=False,
    )

    table.add_column("Model", style="bold white", min_width=16)
    table.add_column("ROUGE-1", justify="right", style="green")
    table.add_column("ROUGE-2", justify="right", style="green")
    table.add_column("ROUGE-L", justify="right", style="green")
    table.add_column("BLEU", justify="right", style="yellow")
    table.add_column("Token-F1", justify="right", style="yellow")
    if use_bertscore:
        table.add_column("BERTScore F1", justify="right", style="magenta")
    table.add_column("Avg Latency", justify="right", style="cyan")

    for _, row in summary.iterrows():
        cols = [
            str(row["model_name"]),
            f"{row['rouge1']:.4f}",
            f"{row['rouge2']:.4f}",
            f"{row['rougeL']:.4f}",
            f"{row['bleu']:.4f}",
            f"{row['f1']:.4f}",
        ]
        if use_bertscore:
            cols.append(f"{row['bertscore_f1']:.4f}")
        cols.append(f"{row['latency_seconds']:.2f}s")
        table.add_row(*cols)

    console.print()
    console.print(table)

    # Show error counts if any calls failed
    errors = df[df["error"].notna()].groupby("model_name").size()
    if not errors.empty:
        console.print("\n[bold red]Failed calls:[/bold red]")
        for model_name, count in errors.items():
            console.print(f"  {model_name}: {count} failure(s)")
    console.print()
