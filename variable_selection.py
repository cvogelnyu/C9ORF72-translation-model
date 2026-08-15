#!/usr/bin/env python3
"""
Elastic Net interaction selection workflow 

Features
----------------------
- Stage 0: mains-only baseline with fixed Elastic Net hyperparameters
- Stage 1A diagnostics on the full dataset
- Stage 2A: stable-pool definition, redundancy filter, K selection by LOOCV,
  and outer LOOCV evaluation of the fixed selected interaction set
- Stage 2B: conditional bootstrap coefficient summaries for the Stage 2A
  fixed interaction set
- Stage 3A: direct Elastic Net on mains + I_fixed, with LOOCV and bootstrap

Assumptions
-----------
- Outcome column is named 'Log2T'
- Any column named 'Unnamed: 0' is treated as an ID column and excluded
- All remaining non-outcome columns are candidate main predictors
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression


# -----------------------------
# Fixed workflow settings
# -----------------------------
ALPHA_F = 0.3
L1_RATIO_F = 0.05
MAX_ITER = 100000
TOL = 1e-6
SELECTION = "cyclic"

REDUNDANCY_CUTOFF = 0.80
NONZERO_FREQ_MIN = 0.60
SIGN_STABILITY_MIN = 0.80
PARSIMONY_DELTA = 0.01
BOOTSTRAP_B = 100
RANDOM_SEED = 12345


# -----------------------------
# Utilities
# -----------------------------
def pooled_oof_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((y_true - y_pred) ** 2) / denom


def fit_enet(X: np.ndarray, y: np.ndarray) -> ElasticNet:
    model = ElasticNet(
        alpha=ALPHA_F,
        l1_ratio=L1_RATIO_F,
        fit_intercept=True,
        max_iter=MAX_ITER,
        tol=TOL,
        selection=SELECTION,
    )
    model.fit(X, y)
    return model


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def quantiles(x: np.ndarray, probs=(0.025, 0.975)) -> Tuple[float, ...]:
    return tuple(np.quantile(np.asarray(x, dtype=float), probs))


# -----------------------------
# Split-specific preprocessing
# -----------------------------
@dataclass
class SplitData:
    Z_train: np.ndarray
    Z_test: np.ndarray
    main_names_active: List[str]
    main_names_all: List[str]
    main_kept_mask: np.ndarray
    W_train: np.ndarray
    W_test: np.ndarray
    int_names_active: List[str]
    int_names_all: List[str]
    int_kept_mask: np.ndarray
    main_stats: Dict[str, List[str]]
    int_stats: Dict[str, List[str]]


def build_split_data(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
) -> SplitData:
    main_names_all = list(X_train_df.columns)
    Xtr = X_train_df.to_numpy(dtype=float)
    Xte = X_test_df.to_numpy(dtype=float)

    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0, ddof=0)

    main_kept_mask = sigma > 0
    Ztr_full = np.zeros_like(Xtr, dtype=float)
    Zte_full = np.zeros_like(Xte, dtype=float)

    kept_idx = np.where(main_kept_mask)[0]
    if kept_idx.size > 0:
        Ztr_full[:, kept_idx] = (Xtr[:, kept_idx] - mu[kept_idx]) / sigma[kept_idx]
        Zte_full[:, kept_idx] = (Xte[:, kept_idx] - mu[kept_idx]) / sigma[kept_idx]

    main_names_active = [main_names_all[i] for i in kept_idx]
    Ztr = Ztr_full[:, kept_idx] if kept_idx.size > 0 else np.zeros((len(Xtr), 0))
    Zte = Zte_full[:, kept_idx] if kept_idx.size > 0 else np.zeros((len(Xte), 0))

    constant_main_names = [main_names_all[i] for i in np.where(~main_kept_mask)[0]]

    pair_idx = list(itertools.combinations(range(len(main_names_active)), 2))
    int_names_all = [f"{main_names_active[i]}x{main_names_active[j]}" for i, j in pair_idx]

    if pair_idx:
        Wtr_raw = np.column_stack([Ztr[:, i] * Ztr[:, j] for i, j in pair_idx])
        Wte_raw = np.column_stack([Zte[:, i] * Zte[:, j] for i, j in pair_idx])

        mu_w = Wtr_raw.mean(axis=0)
        sigma_w = Wtr_raw.std(axis=0, ddof=0)
        int_kept_mask = sigma_w > 0

        Wtr = np.zeros((len(Xtr), int_kept_mask.sum()), dtype=float)
        Wte = np.zeros((len(Xte), int_kept_mask.sum()), dtype=float)

        kept_int_idx = np.where(int_kept_mask)[0]
        if kept_int_idx.size > 0:
            Wtr = (Wtr_raw[:, kept_int_idx] - mu_w[kept_int_idx]) / sigma_w[kept_int_idx]
            Wte = (Wte_raw[:, kept_int_idx] - mu_w[kept_int_idx]) / sigma_w[kept_int_idx]
        int_names_active = [int_names_all[i] for i in kept_int_idx]
        dropped_int_names = [int_names_all[i] for i in np.where(~int_kept_mask)[0]]
    else:
        Wtr = np.zeros((len(Xtr), 0), dtype=float)
        Wte = np.zeros((len(Xte), 0), dtype=float)
        int_names_active = []
        int_names_all = []
        int_kept_mask = np.zeros(0, dtype=bool)
        dropped_int_names = []

    return SplitData(
        Z_train=Ztr,
        Z_test=Zte,
        main_names_active=main_names_active,
        main_names_all=main_names_all,
        main_kept_mask=main_kept_mask,
        W_train=Wtr,
        W_test=Wte,
        int_names_active=int_names_active,
        int_names_all=int_names_all,
        int_kept_mask=int_kept_mask,
        main_stats={"constant_mains": constant_main_names},
        int_stats={"dropped_interactions": dropped_int_names},
    )


def map_selected_interactions_to_split(
    selected_names: Sequence[str],
    split_int_names_active: Sequence[str],
) -> List[int]:
    pos = {name: i for i, name in enumerate(split_int_names_active)}
    return [pos[name] for name in selected_names if name in pos]


# -----------------------------
# Workflow stages
# -----------------------------
def stage0_main_effects_loocv(X: pd.DataFrame, y: np.ndarray) -> Dict:
    n = len(X)
    preds = np.zeros(n, dtype=float)
    coef_rows = []

    for i in range(n):
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[i] = False
        split = build_split_data(X.iloc[tr_mask], X.iloc[~tr_mask])

        if split.Z_train.shape[1] == 0:
            pred = float(np.mean(y[tr_mask]))
            coef_map = {}
        else:
            model = fit_enet(split.Z_train, y[tr_mask])
            pred = float(model.predict(split.Z_test)[0])
            coef_map = dict(zip(split.main_names_active, model.coef_))

        preds[i] = pred
        coef_rows.append(coef_map)

    coef_df = pd.DataFrame(coef_rows).fillna(0.0)
    coef_summary = pd.DataFrame({
        "term": coef_df.columns,
        "mean_coef": coef_df.mean(axis=0).values,
        "sd_coef": coef_df.std(axis=0, ddof=1).values,
    }).sort_values("term").reset_index(drop=True)

    return {
        "r2_oof": pooled_oof_r2(y, preds),
        "predictions": preds,
        "coef_summary": coef_summary,
    }


def global_stage1A_diagnostics(X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    n = len(X)
    iss_records: Dict[str, List[float]] = {}
    nz_records: Dict[str, List[float]] = {}
    sign_records: Dict[str, List[float]] = {}

    for i in range(n):
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[i] = False
        split = build_split_data(X.iloc[tr_mask], X.iloc[~tr_mask])

        if split.Z_train.shape[1] == 0:
            r = y[tr_mask] - np.mean(y[tr_mask])
        else:
            h1 = LinearRegression()
            h1.fit(split.Z_train, y[tr_mask])
            r = y[tr_mask] - h1.predict(split.Z_train)

        # ISS for all active interactions in this split
        for j, name in enumerate(split.int_names_active):
            iss = abs(safe_corr(r, split.W_train[:, j]))
            iss_records.setdefault(name, []).append(iss)

        # H2 on all active interactions
        if split.W_train.shape[1] > 0:
            h2 = fit_enet(split.W_train, r)
            beta = h2.coef_
            for name, b in zip(split.int_names_active, beta):
                nz = float(b != 0.0)
                nz_records.setdefault(name, []).append(nz)
                sign_records.setdefault(name, []).append(np.sign(b) if b != 0.0 else 0.0)

        # Interactions dropped in this split should contribute zeros
        active_set = set(split.int_names_active)
        all_possible = set(split.int_names_all)
        inactive = all_possible - active_set
        for name in inactive:
            iss_records.setdefault(name, []).append(0.0)
            nz_records.setdefault(name, []).append(0.0)
            sign_records.setdefault(name, []).append(0.0)

    all_names = sorted(set(iss_records) | set(nz_records) | set(sign_records))
    rows = []
    for name in all_names:
        iss_vals = np.array(iss_records.get(name, []), dtype=float)
        nz_vals = np.array(nz_records.get(name, []), dtype=float)
        s_vals = np.array(sign_records.get(name, []), dtype=float)

        nonzero_signs = s_vals[s_vals != 0]
        if nonzero_signs.size == 0:
            sign_stability = 0.0
        else:
            sign_stability = max(
                np.mean(nonzero_signs > 0),
                np.mean(nonzero_signs < 0),
            )

        rows.append({
            "interaction": name,
            "iss_mean": float(np.mean(iss_vals)) if iss_vals.size else 0.0,
            "iss_sd": float(np.std(iss_vals, ddof=1)) if iss_vals.size > 1 else 0.0,
            "nonzero_freq": float(np.mean(nz_vals)) if nz_vals.size else 0.0,
            "sign_stability": float(sign_stability),
        })

    out = pd.DataFrame(rows).sort_values(
        ["iss_mean", "nonzero_freq", "sign_stability", "interaction"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return out


def apply_redundancy_filter(
    X: pd.DataFrame,
    stable_pool_ranked: Sequence[str],
    cutoff: float = REDUNDANCY_CUTOFF,
) -> Tuple[List[str], pd.DataFrame]:
    split = build_split_data(X, X.iloc[:0].copy())
    main_df = pd.DataFrame(split.Z_train, columns=split.main_names_active)
    int_df = pd.DataFrame(split.W_train, columns=split.int_names_active)

    kept: List[str] = []
    dropped_rows = []

    for name in stable_pool_ranked:
        if name not in int_df.columns:
            dropped_rows.append({
                "interaction": name,
                "dropped_for_redundancy": True,
                "reason": "not available on full data after split-specific preprocessing",
                "with_term": "",
                "abs_corr": np.nan,
            })
            continue

        x = int_df[name].to_numpy()
        drop = False

        # Compare to all active mains
        for m in main_df.columns:
            c = abs(safe_corr(x, main_df[m].to_numpy()))
            if c > cutoff:
                dropped_rows.append({
                    "interaction": name,
                    "dropped_for_redundancy": True,
                    "reason": "correlated_with_main",
                    "with_term": m,
                    "abs_corr": c,
                })
                drop = True
                break
        if drop:
            continue

        # Compare to higher-ranked retained interactions
        for prev in kept:
            c = abs(safe_corr(x, int_df[prev].to_numpy()))
            if c > cutoff:
                dropped_rows.append({
                    "interaction": name,
                    "dropped_for_redundancy": True,
                    "reason": "correlated_with_higher_ranked_interaction",
                    "with_term": prev,
                    "abs_corr": c,
                })
                drop = True
                break

        if not drop:
            kept.append(name)

    dropped_df = pd.DataFrame(dropped_rows)
    return kept, dropped_df


def loocv_hierarchy_with_selected_interactions(
    X: pd.DataFrame,
    y: np.ndarray,
    selected_interactions: Sequence[str],
) -> Dict:
    n = len(X)
    preds = np.zeros(n, dtype=float)
    h1_coef_rows = []
    h2_coef_rows = []

    for i in range(n):
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[i] = False
        split = build_split_data(X.iloc[tr_mask], X.iloc[~tr_mask])

        # H1: OLS on active mains
        if split.Z_train.shape[1] == 0:
            yhat_tr_h1 = np.repeat(np.mean(y[tr_mask]), tr_mask.sum())
            yhat_te_h1 = np.array([np.mean(y[tr_mask])], dtype=float)
            h1_coef_map = {}
        else:
            h1 = LinearRegression()
            h1.fit(split.Z_train, y[tr_mask])
            yhat_tr_h1 = h1.predict(split.Z_train)
            yhat_te_h1 = h1.predict(split.Z_test)
            h1_coef_map = dict(zip(split.main_names_active, h1.coef_))

        r_tr = y[tr_mask] - yhat_tr_h1

        cols = map_selected_interactions_to_split(selected_interactions, split.int_names_active)
        if len(cols) == 0:
            preds[i] = float(yhat_te_h1[0])
            h2_coef_map = {}
        else:
            X2_tr = split.W_train[:, cols]
            X2_te = split.W_test[:, cols]
            names2 = [split.int_names_active[c] for c in cols]
            h2 = fit_enet(X2_tr, r_tr)
            rhat_te = h2.predict(X2_te)
            preds[i] = float(yhat_te_h1[0] + rhat_te[0])
            h2_coef_map = dict(zip(names2, h2.coef_))

        h1_coef_rows.append(h1_coef_map)
        h2_coef_rows.append(h2_coef_map)

    h1_df = pd.DataFrame(h1_coef_rows).fillna(0.0)
    h2_df = pd.DataFrame(h2_coef_rows).fillna(0.0)

    h1_summary = pd.DataFrame({
        "term": h1_df.columns,
        "mean_coef": h1_df.mean(axis=0).values,
        "sd_coef": h1_df.std(axis=0, ddof=1).values,
        "component": "H1_main",
    }) if h1_df.shape[1] > 0 else pd.DataFrame(columns=["term", "mean_coef", "sd_coef", "component"])

    h2_summary = pd.DataFrame({
        "term": h2_df.columns,
        "mean_coef": h2_df.mean(axis=0).values,
        "sd_coef": h2_df.std(axis=0, ddof=1).values,
        "component": "H2_interaction",
    }) if h2_df.shape[1] > 0 else pd.DataFrame(columns=["term", "mean_coef", "sd_coef", "component"])

    coef_summary = pd.concat([h1_summary, h2_summary], ignore_index=True).sort_values(
        ["component", "term"]
    ).reset_index(drop=True)

    return {
        "r2_oof": pooled_oof_r2(y, preds),
        "predictions": preds,
        "coef_summary": coef_summary,
    }


def loocv_direct_enet(
    X: pd.DataFrame,
    y: np.ndarray,
    selected_interactions: Sequence[str],
) -> Dict:
    n = len(X)
    preds = np.zeros(n, dtype=float)
    coef_rows = []

    for i in range(n):
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[i] = False
        split = build_split_data(X.iloc[tr_mask], X.iloc[~tr_mask])

        Xtr_parts = []
        Xte_parts = []
        names = []

        if split.Z_train.shape[1] > 0:
            Xtr_parts.append(split.Z_train)
            Xte_parts.append(split.Z_test)
            names.extend(split.main_names_active)

        cols = map_selected_interactions_to_split(selected_interactions, split.int_names_active)
        if len(cols) > 0:
            Xtr_parts.append(split.W_train[:, cols])
            Xte_parts.append(split.W_test[:, cols])
            names.extend([split.int_names_active[c] for c in cols])

        if len(Xtr_parts) == 0:
            preds[i] = float(np.mean(y[tr_mask]))
            coef_map = {}
        else:
            Xtr = np.column_stack(Xtr_parts)
            Xte = np.column_stack(Xte_parts)
            model = fit_enet(Xtr, y[tr_mask])
            preds[i] = float(model.predict(Xte)[0])
            coef_map = dict(zip(names, model.coef_))

        coef_rows.append(coef_map)

    coef_df = pd.DataFrame(coef_rows).fillna(0.0)
    coef_summary = pd.DataFrame({
        "term": coef_df.columns,
        "mean_coef": coef_df.mean(axis=0).values,
        "sd_coef": coef_df.std(axis=0, ddof=1).values,
    }).sort_values("term").reset_index(drop=True)

    return {
        "r2_oof": pooled_oof_r2(y, preds),
        "predictions": preds,
        "coef_summary": coef_summary,
    }


def choose_k_by_loocv(
    X: pd.DataFrame,
    y: np.ndarray,
    retained_pool_ranked: Sequence[str],
) -> Tuple[int, pd.DataFrame]:
    rows = []
    best_r2 = -np.inf

    for k in range(len(retained_pool_ranked) + 1):
        selected = retained_pool_ranked[:k]
        res = loocv_hierarchy_with_selected_interactions(X, y, selected)
        r2 = res["r2_oof"]
        rows.append({"K": k, "r2_oof": r2})
        best_r2 = max(best_r2, r2)

    curve = pd.DataFrame(rows)
    eligible = curve[curve["r2_oof"] >= best_r2 - PARSIMONY_DELTA]
    k_star = int(eligible["K"].min())
    return k_star, curve


def bootstrap_stage2B(
    X: pd.DataFrame,
    y: np.ndarray,
    selected_interactions: Sequence[str],
    B: int = BOOTSTRAP_B,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(X)
    rows = []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        Xb = X.iloc[idx].reset_index(drop=True)
        yb = y[idx]

        split = build_split_data(Xb, Xb.iloc[:0].copy())

        # H1
        if split.Z_train.shape[1] == 0:
            yhat_h1 = np.repeat(np.mean(yb), n)
            h1_map = {}
        else:
            h1 = LinearRegression()
            h1.fit(split.Z_train, yb)
            yhat_h1 = h1.predict(split.Z_train)
            h1_map = dict(zip(split.main_names_active, h1.coef_))

        # H2 on fixed interactions
        r = yb - yhat_h1
        cols = map_selected_interactions_to_split(selected_interactions, split.int_names_active)
        if len(cols) == 0:
            h2_map = {}
        else:
            X2 = split.W_train[:, cols]
            names2 = [split.int_names_active[c] for c in cols]
            h2 = fit_enet(X2, r)
            h2_map = dict(zip(names2, h2.coef_))

        merged = {}
        for k, v in h1_map.items():
            merged[f"H1::{k}"] = v
        for k, v in h2_map.items():
            merged[f"H2::{k}"] = v
        rows.append(merged)

    coef_df = pd.DataFrame(rows).fillna(0.0)
    summary_rows = []
    for term in coef_df.columns:
        vals = coef_df[term].to_numpy()
        q025, q975 = quantiles(vals)
        component, clean_term = term.split("::", 1)
        summary_rows.append({
            "component": component,
            "term": clean_term,
            "boot_mean": float(np.mean(vals)),
            "boot_sd": float(np.std(vals, ddof=1)),
            "boot_q025": float(q025),
            "boot_q975": float(q975),
        })

    return pd.DataFrame(summary_rows).sort_values(["component", "term"]).reset_index(drop=True)


def bootstrap_stage3A(
    X: pd.DataFrame,
    y: np.ndarray,
    selected_interactions: Sequence[str],
    B: int = BOOTSTRAP_B,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(X)
    rows = []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        Xb = X.iloc[idx].reset_index(drop=True)
        yb = y[idx]
        split = build_split_data(Xb, Xb.iloc[:0].copy())

        X_parts = []
        names = []

        if split.Z_train.shape[1] > 0:
            X_parts.append(split.Z_train)
            names.extend(split.main_names_active)

        cols = map_selected_interactions_to_split(selected_interactions, split.int_names_active)
        if len(cols) > 0:
            X_parts.append(split.W_train[:, cols])
            names.extend([split.int_names_active[c] for c in cols])

        if len(X_parts) == 0:
            coef_map = {}
        else:
            X_design = np.column_stack(X_parts)
            model = fit_enet(X_design, yb)
            coef_map = dict(zip(names, model.coef_))

        rows.append(coef_map)

    coef_df = pd.DataFrame(rows).fillna(0.0)
    summary_rows = []
    for term in coef_df.columns:
        vals = coef_df[term].to_numpy()
        q025, q975 = quantiles(vals)
        summary_rows.append({
            "term": term,
            "boot_mean": float(np.mean(vals)),
            "boot_sd": float(np.std(vals, ddof=1)),
            "boot_q025": float(q025),
            "boot_q975": float(q975),
        })

    return pd.DataFrame(summary_rows).sort_values("term").reset_index(drop=True)


# -----------------------------
# I/O and runner
# -----------------------------
def load_data(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    drop_cols = [c for c in df.columns if c == "Unnamed: 0"]
    if "Log2T" not in df.columns:
        raise ValueError("CSV must contain an outcome column named 'Log2T'.")
    X = df.drop(columns=drop_cols + ["Log2T"]).copy()
    y = df["Log2T"].to_numpy(dtype=float)
    return X, y


def run_variable_selection_workflow(csv_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_data(csv_path)

    # Stage 0
    stage0 = stage0_main_effects_loocv(X, y)

    # Global Stage 1A
    diag = global_stage1A_diagnostics(X, y)
    stable_pool = diag[
        (diag["nonzero_freq"] >= NONZERO_FREQ_MIN) &
        (diag["sign_stability"] >= SIGN_STABILITY_MIN)
    ].copy().reset_index(drop=True)

    stable_pool_ranked = list(stable_pool.sort_values(
        ["iss_mean", "nonzero_freq", "sign_stability", "interaction"],
        ascending=[False, False, False, True],
    )["interaction"])

    retained_pool, dropped_redundancy = apply_redundancy_filter(X, stable_pool_ranked)

    # Stage 2A
    k_star, k_curve = choose_k_by_loocv(X, y, retained_pool)
    i_fixed = retained_pool[:k_star]
    stage2a = loocv_hierarchy_with_selected_interactions(X, y, i_fixed)

    # Stage 2B
    stage2b_boot = bootstrap_stage2B(X, y, i_fixed, B=BOOTSTRAP_B, seed=RANDOM_SEED)

    # Stage 3A
    stage3a = loocv_direct_enet(X, y, i_fixed)
    stage3a_boot = bootstrap_stage3A(X, y, i_fixed, B=BOOTSTRAP_B, seed=RANDOM_SEED)

    # Summary
    summary = pd.DataFrame([
        {"stage": "Stage 0 mains-only", "r2_oof": stage0["r2_oof"], "n_interactions": 0},
        {"stage": "Stage 2A selected hierarchy", "r2_oof": stage2a["r2_oof"], "n_interactions": len(i_fixed)},
        {"stage": "Stage 3A direct ENet mains + I_fixed", "r2_oof": stage3a["r2_oof"], "n_interactions": len(i_fixed)},
    ])

    selected_interactions_df = pd.DataFrame({
        "K_star": [k_star] * len(i_fixed) if len(i_fixed) else [k_star],
        "selected_interaction": i_fixed if len(i_fixed) else [""],
    })

    # Save tabular outputs as CSV
    summary.to_csv(out_dir / "summary.csv", index=False)
    stage0["coef_summary"].to_csv(out_dir / "stage0_coef_summary.csv", index=False)
    diag.to_csv(out_dir / "global_stage1A_diagnostics.csv", index=False)
    stable_pool.to_csv(out_dir / "stable_pool_before_redundancy.csv", index=False)
    pd.DataFrame({"retained_after_redundancy": retained_pool}).to_csv(
        out_dir / "stable_pool_after_redundancy.csv", index=False
    )
    dropped_redundancy.to_csv(out_dir / "redundancy_drops.csv", index=False)
    k_curve.to_csv(out_dir / "k_vs_r2_curve.csv", index=False)
    selected_interactions_df.to_csv(out_dir / "selected_interactions.csv", index=False)
    stage2a["coef_summary"].to_csv(out_dir / "stage2a_coef_summary.csv", index=False)
    stage2b_boot.to_csv(out_dir / "stage2b_bootstrap_summary.csv", index=False)
    stage3a["coef_summary"].to_csv(out_dir / "stage3a_coef_summary.csv", index=False)
    stage3a_boot.to_csv(out_dir / "stage3a_bootstrap_summary.csv", index=False)

    # Save a compact JSON summary for easy inspection
    compact = {
        "settings": {
            "alpha": ALPHA_F,
            "l1_ratio": L1_RATIO_F,
            "redundancy_cutoff": REDUNDANCY_CUTOFF,
            "nonzero_freq_min": NONZERO_FREQ_MIN,
            "sign_stability_min": SIGN_STABILITY_MIN,
            "parsimony_delta": PARSIMONY_DELTA,
            "bootstrap_B": BOOTSTRAP_B,
            "random_seed": RANDOM_SEED,
        },
        "main_predictors": list(X.columns),
        "summary": summary.to_dict(orient="records"),
        "k_star": k_star,
        "i_fixed": i_fixed,
        "n_stable_pool_before_redundancy": int(len(stable_pool_ranked)),
        "n_stable_pool_after_redundancy": int(len(retained_pool)),
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)

    print(f"Wrote outputs to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the variable-selection subset of the Elastic Net interaction workflow."
    )
    parser.add_argument("csv_path", type=Path, help="Path to input CSV with outcome column Log2T.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("variable_selection_results"),
        help="Directory to write result files.",
    )
    args = parser.parse_args()
    run_variable_selection_workflow(args.csv_path, args.out_dir)


if __name__ == "__main__":
    main()
