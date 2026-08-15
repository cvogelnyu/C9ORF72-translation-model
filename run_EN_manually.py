#!/usr/bin/env python3
"""
Elastic Net on user-defined terms for Log2T

Features
--------
- Reads a CSV file
- Models outcome column 'Log2T'
- Uses user-defined main effects and interaction terms
- Z-score normalization (default on)
- Uses user-defined Elastic Net alpha and l1_ratio
- Computes:
    * LOOCV pooled R^2
    * observed, predicted, residuals for LOOCV
    * bootstrap coefficient summaries for 100 replicates by default
      (mean, SD, 2.5th percentile, 50th percentile, 97.5th percentile)
- Writes results to CSV files

Assumptions
-----------
- Interactions are built as products of the selected main-effect columns.
- If normalization is enabled, each LOOCV or bootstrap training sample is scaled
  using training-sample means and SDs only, then interactions are built from the
  scaled main effects.
- If normalization is disabled, interactions are built from raw main effects.
- If a selected predictor is constant in a training split, that column is set to 0
  in that split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet


def pooled_oof_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def parse_interactions(items: Sequence[str]) -> List[Tuple[str, str]]:
    out = []
    for item in items:
        s = item.strip()
        if not s:
            continue
        if "x" in s:
            a, b = s.split("x", 1)
        elif "*" in s:
            a, b = s.split("*", 1)
        else:
            raise ValueError(
                f"Interaction '{item}' must use 'x' or '*' between two variable names."
            )
        a = a.strip()
        b = b.strip()
        if not a or not b:
            raise ValueError(f"Invalid interaction specification: '{item}'")
        out.append((a, b))
    return out


def load_term_spec(
    mains: Sequence[str] | None,
    interactions: Sequence[str] | None,
    terms_json: Path | None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    if terms_json is not None:
        with open(terms_json, "r", encoding="utf-8") as f:
            spec = json.load(f)
        mains_list = spec.get("mains", [])
        interactions_list = spec.get("interactions", [])
    else:
        mains_list = list(mains or [])
        interactions_list = list(interactions or [])

    if not mains_list:
        raise ValueError("At least one main effect must be provided.")

    parsed_interactions = parse_interactions(interactions_list)
    return list(mains_list), parsed_interactions


def validate_columns(df: pd.DataFrame, mains: Sequence[str], interactions: Sequence[Tuple[str, str]]) -> None:
    missing = sorted(set([c for c in mains if c not in df.columns]))
    for a, b in interactions:
        if a not in df.columns:
            missing.append(a)
        if b not in df.columns:
            missing.append(b)
    if "Log2T" not in df.columns:
        missing.append("Log2T")
    if missing:
        raise ValueError(f"Missing required columns: {sorted(set(missing))}")


def transform_split(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    mains: Sequence[str],
    interactions: Sequence[Tuple[str, str]],
    normalize: bool = True,
):
    Xtr_raw = X_train_df.loc[:, mains].to_numpy(dtype=float)
    Xte_raw = X_test_df.loc[:, mains].to_numpy(dtype=float)

    if normalize:
        mu = Xtr_raw.mean(axis=0)
        sd = Xtr_raw.std(axis=0, ddof=0)
        keep = sd > 0

        Xtr = np.zeros_like(Xtr_raw, dtype=float)
        Xte = np.zeros_like(Xte_raw, dtype=float)
        if np.any(keep):
            Xtr[:, keep] = (Xtr_raw[:, keep] - mu[keep]) / sd[keep]
            Xte[:, keep] = (Xte_raw[:, keep] - mu[keep]) / sd[keep]
    else:
        Xtr = Xtr_raw.copy()
        Xte = Xte_raw.copy()

    main_pos = {name: i for i, name in enumerate(mains)}

    parts_tr = []
    parts_te = []
    names = []

    for j, name in enumerate(mains):
        parts_tr.append(Xtr[:, [j]])
        parts_te.append(Xte[:, [j]])
        names.append(name)

    for a, b in interactions:
        ia = main_pos[a]
        ib = main_pos[b]
        inter_tr = (Xtr[:, ia] * Xtr[:, ib]).reshape(-1, 1)
        inter_te = (Xte[:, ia] * Xte[:, ib]).reshape(-1, 1)
        names.append(f"{a}x{b}")
        parts_tr.append(inter_tr)
        parts_te.append(inter_te)

    Xtr_design = np.hstack(parts_tr) if parts_tr else np.zeros((len(X_train_df), 0))
    Xte_design = np.hstack(parts_te) if parts_te else np.zeros((len(X_test_df), 0))

    return Xtr_design, Xte_design, names


def fit_model(X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float) -> ElasticNet:
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=100000,
        tol=1e-6,
        selection="cyclic",
    )
    model.fit(X, y)
    return model


def run_loocv(
    df: pd.DataFrame,
    mains: Sequence[str],
    interactions: Sequence[Tuple[str, str]],
    alpha: float,
    l1_ratio: float,
    normalize: bool,
):
    y = df["Log2T"].to_numpy(dtype=float)
    n = len(df)

    preds = np.zeros(n, dtype=float)
    intercepts = np.zeros(n, dtype=float)
    coef_rows = []

    for i in range(n):
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[i] = False

        train_df = df.iloc[tr_mask].reset_index(drop=True)
        test_df = df.iloc[~tr_mask].reset_index(drop=True)

        Xtr, Xte, term_names = transform_split(
            train_df, test_df, mains, interactions, normalize=normalize
        )
        ytr = train_df["Log2T"].to_numpy(dtype=float)

        model = fit_model(Xtr, ytr, alpha=alpha, l1_ratio=l1_ratio)
        pred = float(model.predict(Xte)[0])

        preds[i] = pred
        intercepts[i] = float(model.intercept_)
        coef_rows.append(dict(zip(term_names, model.coef_)))

    residuals = y - preds
    obs_pred = pd.DataFrame({
        "row_index": np.arange(n),
        "observed": y,
        "predicted": preds,
        "residual": residuals,
        "abs_residual": np.abs(residuals),
        "squared_residual": residuals ** 2,
        "intercept": intercepts,
    })

    coef_df = pd.DataFrame(coef_rows).fillna(0.0)
    coef_summary = pd.DataFrame({
        "term": coef_df.columns,
        "loocv_coef_mean": coef_df.mean(axis=0).values,
        "loocv_coef_sd": coef_df.std(axis=0, ddof=1).values,
        "loocv_coef_q025": coef_df.quantile(0.025, axis=0).values,
        "loocv_coef_q500": coef_df.quantile(0.500, axis=0).values,
        "loocv_coef_q975": coef_df.quantile(0.975, axis=0).values,
    }).sort_values("term").reset_index(drop=True)

    r2 = pooled_oof_r2(y, preds)
    return obs_pred, coef_summary, r2


def run_bootstrap(
    df: pd.DataFrame,
    mains: Sequence[str],
    interactions: Sequence[Tuple[str, str]],
    alpha: float,
    l1_ratio: float,
    normalize: bool,
    n_boot: int,
    random_seed: int,
):
    rng = np.random.default_rng(random_seed)
    n = len(df)

    coef_rows = []
    intercepts = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_df = df.iloc[idx].reset_index(drop=True)
        Xb, _, term_names = transform_split(
            boot_df, boot_df.iloc[:0].copy(), mains, interactions, normalize=normalize
        )
        yb = boot_df["Log2T"].to_numpy(dtype=float)

        model = fit_model(Xb, yb, alpha=alpha, l1_ratio=l1_ratio)
        coef_rows.append(dict(zip(term_names, model.coef_)))
        intercepts.append(float(model.intercept_))

    coef_df = pd.DataFrame(coef_rows).fillna(0.0)
    coef_summary = pd.DataFrame({
        "term": coef_df.columns,
        "boot_coef_mean": coef_df.mean(axis=0).values,
        "boot_coef_sd": coef_df.std(axis=0, ddof=1).values,
        "boot_coef_q025": coef_df.quantile(0.025, axis=0).values,
        "boot_coef_q500": coef_df.quantile(0.500, axis=0).values,
        "boot_coef_q975": coef_df.quantile(0.975, axis=0).values,
    }).sort_values("term").reset_index(drop=True)

    intercept_df = pd.DataFrame({
        "boot_intercept_mean": [float(np.mean(intercepts))],
        "boot_intercept_sd": [float(np.std(intercepts, ddof=1))],
        "boot_intercept_q025": [float(np.quantile(intercepts, 0.025))],
        "boot_intercept_q500": [float(np.quantile(intercepts, 0.500))],
        "boot_intercept_q975": [float(np.quantile(intercepts, 0.975))],
        "n_boot": [n_boot],
    })

    return coef_summary, intercept_df


def write_outputs(
    out_dir: Path,
    obs_pred: pd.DataFrame,
    loocv_coef_summary: pd.DataFrame,
    bootstrap_coef_summary: pd.DataFrame,
    bootstrap_intercept_summary: pd.DataFrame,
    summary_df: pd.DataFrame,
    settings_df: pd.DataFrame,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_pred.to_csv(out_dir / "observed_predicted_residuals.csv", index=False)
    loocv_coef_summary.to_csv(out_dir / "loocv_coefficient_summary.csv", index=False)
    bootstrap_coef_summary.to_csv(out_dir / "bootstrap_coefficient_summary.csv", index=False)
    bootstrap_intercept_summary.to_csv(out_dir / "bootstrap_intercept_summary.csv", index=False)
    summary_df.to_csv(out_dir / "model_summary.csv", index=False)
    settings_df.to_csv(out_dir / "run_settings.csv", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Elastic Net on user-defined mains and interactions for Log2T."
    )
    parser.add_argument("csv_path", type=Path, help="Input CSV file.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("enet_user_terms_results"),
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Elastic Net alpha.",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        required=True,
        help="Elastic Net l1_ratio.",
    )
    parser.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=True,
        help="Apply Z-score normalization within each training split (default).",
    )
    parser.add_argument(
        "--raw",
        dest="normalize",
        action="store_false",
        help="Use raw data without Z-score normalization.",
    )
    parser.add_argument(
        "--mains",
        nargs="*",
        default=None,
        help="List of main-effect variables, e.g. --mains F0a F2a SL1",
    )
    parser.add_argument(
        "--interactions",
        nargs="*",
        default=None,
        help="List of interactions, e.g. --interactions F0axSL1 F2axA88 or F0a*SL1",
    )
    parser.add_argument(
        "--terms-json",
        type=Path,
        default=None,
        help="Optional JSON file with keys 'mains' and 'interactions'.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=100,
        help="Number of bootstrap replicates (default 100).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=12345,
        help="Random seed for bootstrapping.",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    mains, interactions = load_term_spec(
        mains=args.mains,
        interactions=args.interactions,
        terms_json=args.terms_json,
    )
    validate_columns(df, mains, interactions)

    obs_pred, loocv_coef_summary, loocv_r2 = run_loocv(
        df=df,
        mains=mains,
        interactions=interactions,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        normalize=args.normalize,
    )

    bootstrap_coef_summary, bootstrap_intercept_summary = run_bootstrap(
        df=df,
        mains=mains,
        interactions=interactions,
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        normalize=args.normalize,
        n_boot=args.n_boot,
        random_seed=args.random_seed,
    )

    summary_df = pd.DataFrame({
        "metric": ["loocv_r2", "n_rows", "n_mains", "n_interactions"],
        "value": [loocv_r2, len(df), len(mains), len(interactions)],
    })

    settings_df = pd.DataFrame([{
        "csv_path": str(args.csv_path),
        "out_dir": str(args.out_dir),
        "alpha": args.alpha,
        "l1_ratio": args.l1_ratio,
        "normalize": args.normalize,
        "n_boot": args.n_boot,
        "random_seed": args.random_seed,
        "mains": ";".join(mains),
        "interactions": ";".join([f"{a}x{b}" for a, b in interactions]),
    }])

    write_outputs(
        out_dir=args.out_dir,
        obs_pred=obs_pred,
        loocv_coef_summary=loocv_coef_summary,
        bootstrap_coef_summary=bootstrap_coef_summary,
        bootstrap_intercept_summary=bootstrap_intercept_summary,
        summary_df=summary_df,
        settings_df=settings_df,
    )

    print(f"Wrote CSV outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
