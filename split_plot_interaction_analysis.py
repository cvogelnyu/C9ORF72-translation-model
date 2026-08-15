#!/usr/bin/env python3
"""
Split-plot analysis for a user-defined interaction term.

Features
--------
For a user-defined interaction between two variables:
1. Treat variable A as MAIN and variable B as MODIFIER.
2. Rank MODIFIER values and split them into low and high halves.
3. Within each half, fit a simple linear regression:
       Log2T ~ MAIN
   and report:
   - slope of MAIN
   - R^2 of that relationship
   - mean of the MODIFIER subset
   - SD of the MODIFIER subset
   - n
4. Swap roles:
   - treat variable B as MAIN
   - treat variable A as MODIFIER
   and repeat the same analysis.
5. Write CSV outputs.

Assumptions
-----------
- Outcome column must be named 'Log2T'.
- Splitting uses rank order of the MODIFIER values.
- If n is odd, the lower half gets floor(n/2) rows and the upper half gets the remaining rows.
- Ties in MODIFIER are handled by stable sorting on the observed values.
- If MAIN has zero variance within a subset, slope is set to 0 and R^2 is set to 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def safe_fit_simple_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns slope, intercept, R^2 for y ~ x.
    If x has zero variance or fewer than 2 observations, returns slope=0, intercept=mean(y), R^2=0.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2 or np.std(x, ddof=0) == 0:
        intercept = float(np.mean(y)) if len(y) > 0 else np.nan
        return 0.0, intercept, 0.0

    model = LinearRegression()
    X = x.reshape(-1, 1)
    model.fit(X, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(X, y))
    return slope, intercept, r2


def split_low_high_by_modifier(df: pd.DataFrame, modifier: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort by modifier and split into low/high halves.
    If n is odd, low gets floor(n/2), high gets ceil(n/2).
    """
    ordered = df.sort_values(by=modifier, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    cut = n // 2
    low = ordered.iloc[:cut].copy()
    high = ordered.iloc[cut:].copy()
    return low, high


def analyze_direction(df: pd.DataFrame, main: str, modifier: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run split-half analysis for one direction:
      - modifier defines low/high split
      - model is Log2T ~ main within each half

    Returns:
      summary_df: one row per half
      subset_df: original rows with assigned half labels
    """
    low_df, high_df = split_low_high_by_modifier(df, modifier)

    summaries: List[Dict] = []
    subset_frames: List[pd.DataFrame] = []

    for half_name, sub in [("low", low_df), ("high", high_df)]:
        slope, intercept, r2 = safe_fit_simple_regression(
            sub[main].to_numpy(dtype=float),
            sub["Log2T"].to_numpy(dtype=float),
        )

        modifier_values = sub[modifier].to_numpy(dtype=float)
        modifier_mean = float(np.mean(modifier_values)) if len(modifier_values) > 0 else np.nan
        modifier_sd = float(np.std(modifier_values, ddof=1)) if len(modifier_values) > 1 else 0.0

        summaries.append({
            "main": main,
            "modifier": modifier,
            "half": half_name,
            "n": int(len(sub)),
            "main_slope_vs_Log2T": slope,
            "intercept": intercept,
            "r2": r2,
            "modifier_mean": modifier_mean,
            "modifier_sd": modifier_sd,
            "modifier_min": float(np.min(modifier_values)) if len(modifier_values) > 0 else np.nan,
            "modifier_max": float(np.max(modifier_values)) if len(modifier_values) > 0 else np.nan,
        })

        tmp = sub.copy()
        tmp["analysis_main"] = main
        tmp["analysis_modifier"] = modifier
        tmp["modifier_half"] = half_name
        subset_frames.append(tmp)

    summary_df = pd.DataFrame(summaries)
    subset_df = pd.concat(subset_frames, ignore_index=True)
    return summary_df, subset_df


def validate_inputs(df: pd.DataFrame, var1: str, var2: str) -> None:
    missing = [c for c in ["Log2T", var1, var2] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def run_analysis(csv_path: Path, var1: str, var2: str, out_dir: Path) -> None:
    df = pd.read_csv(csv_path)
    validate_inputs(df, var1, var2)

    # Keep only rows with complete data for the variables used here
    analysis_df = df.loc[:, [var1, var2, "Log2T"]].dropna().reset_index(drop=True)

    # Direction 1: var1 as MAIN, var2 as MODIFIER
    summary_12, subsets_12 = analyze_direction(analysis_df, main=var1, modifier=var2)

    # Direction 2: var2 as MAIN, var1 as MODIFIER
    summary_21, subsets_21 = analyze_direction(analysis_df, main=var2, modifier=var1)

    combined_summary = pd.concat([summary_12, summary_21], ignore_index=True)
    combined_subsets = pd.concat([subsets_12, subsets_21], ignore_index=True)

    run_settings = pd.DataFrame([{
        "csv_path": str(csv_path),
        "out_dir": str(out_dir),
        "var1": var1,
        "var2": var2,
        "n_complete_cases": int(len(analysis_df)),
        "split_rule": "sort by modifier, low=floor(n/2), high=remaining",
        "model_within_half": "Log2T ~ MAIN",
    }])

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_12.to_csv(out_dir / f"split_analysis_{var1}_as_MAIN_{var2}_as_MODIFIER.csv", index=False)
    summary_21.to_csv(out_dir / f"split_analysis_{var2}_as_MAIN_{var1}_as_MODIFIER.csv", index=False)
    combined_summary.to_csv(out_dir / "split_analysis_summary_all.csv", index=False)
    subsets_12.to_csv(out_dir / f"split_subsets_{var1}_as_MAIN_{var2}_as_MODIFIER.csv", index=False)
    subsets_21.to_csv(out_dir / f"split_subsets_{var2}_as_MAIN_{var1}_as_MODIFIER.csv", index=False)
    combined_subsets.to_csv(out_dir / "split_subsets_all.csv", index=False)
    run_settings.to_csv(out_dir / "run_settings.csv", index=False)

    print(f"Wrote CSV outputs to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split-plot analysis for a user-defined interaction."
    )
    parser.add_argument("csv_path", type=Path, help="Input CSV file.")
    parser.add_argument("--var1", required=True, help="First variable in the interaction.")
    parser.add_argument("--var2", required=True, help="Second variable in the interaction.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("split_plot_analysis_results"),
        help="Output directory for CSV files.",
    )

    args = parser.parse_args()
    run_analysis(
        csv_path=args.csv_path,
        var1=args.var1,
        var2=args.var2,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
