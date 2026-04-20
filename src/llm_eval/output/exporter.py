"""CSV export: full per-prediction results + aggregated summary."""

from pathlib import Path

import pandas as pd


def export_csv(df: pd.DataFrame, output_path: str) -> None:
    """Export full per-prediction results (one row per model per question)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig BOM makes the file open correctly in Excel on Chinese Windows
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Full results  -> {output_path}")


def export_summary_csv(df: pd.DataFrame, output_path: str) -> None:
    """Export per-model aggregated summary (mean ± std for all metrics)."""
    numeric_cols = [
        "rouge1", "rouge2", "rougeL", "bleu", "f1",
        "bertscore_p", "bertscore_r", "bertscore_f1",
        "latency_seconds",
    ]
    # Only include columns that actually exist (BERTScore may be skipped)
    existing = [c for c in numeric_cols if c in df.columns]

    summary = (
        df[df["error"].isna()]
        .groupby("model_name")[existing]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    summary_path = output_path.replace(".csv", "_summary.csv")
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Summary       -> {summary_path}")
