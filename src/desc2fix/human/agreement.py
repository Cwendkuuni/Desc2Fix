#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from desc2fix.human.schema import SCORE_FIELDS


KEYS = [
    "rq",
    "dataset",
    "representation",
    "patch_view",
    "pair_id",
]


def safe_corr(x, y, method: str) -> Dict[str, Optional[float]]:
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return {"value": np.nan, "pvalue": np.nan}

    try:
        from scipy.stats import spearmanr, kendalltau

        if method == "spearman":
            res = spearmanr(x, y)
            return {"value": float(res.statistic), "pvalue": float(res.pvalue)}

        if method == "kendall":
            res = kendalltau(x, y)
            return {"value": float(res.statistic), "pvalue": float(res.pvalue)}

    except Exception:
        return {"value": float(x.corr(y, method=method)), "pvalue": np.nan}

    raise ValueError(method)


def rmse(x, y) -> float:
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    mask = x.notna() & y.notna()

    if mask.sum() == 0:
        return np.nan

    return float(np.sqrt(np.mean((x[mask] - y[mask]) ** 2)))


def mae(x, y) -> float:
    x = pd.Series(x, dtype="float64")
    y = pd.Series(y, dtype="float64")
    mask = x.notna() & y.notna()

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(np.abs(x[mask] - y[mask])))


def mean_bias(pred, ref) -> float:
    pred = pd.Series(pred, dtype="float64")
    ref = pd.Series(ref, dtype="float64")
    mask = pred.notna() & ref.notna()

    if mask.sum() == 0:
        return np.nan

    return float(np.mean(pred[mask] - ref[mask]))


def score_col(df: pd.DataFrame, field: str, preferred: str) -> str:
    """
    Resolve score column after pandas merges.

    preferred can be:
      - "raw": field
      - "human": field_human
      - "llm": field_llm
      - "ref": field_ref
    """
    candidates = []

    if preferred == "raw":
        candidates = [field, f"{field}_human", f"{field}_llm", f"{field}_ref"]
    elif preferred == "human":
        candidates = [f"{field}_human", field, f"{field}_ref"]
    elif preferred == "llm":
        candidates = [f"{field}_llm", field]
    elif preferred == "ref":
        candidates = [f"{field}_ref", field, f"{field}_human"]
    else:
        candidates = [field]

    for c in candidates:
        if c in df.columns:
            return c

    raise KeyError(
        f"Cannot resolve column for field={field}, preferred={preferred}. "
        f"Available columns: {list(df.columns)}"
    )


def human_average(human_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate human annotations by instance.

    Input columns must include raw human scores:
      alignment_score, coverage, correctness, specificity

    Output columns:
      <metric>_mean, <metric>_std
    """
    rows = []

    if human_df.empty:
        return pd.DataFrame(rows)

    grouped = human_df.groupby(KEYS, dropna=False)

    for key, g in grouped:
        row = dict(zip(KEYS, key))
        row["n_raters"] = g["rater"].nunique()

        for field in SCORE_FIELDS:
            col = score_col(g, field, "raw")
            row[f"{field}_mean"] = g[col].mean()
            row[f"{field}_std"] = g[col].std()

        rows.append(row)

    return pd.DataFrame(rows)


def inter_human_agreement(human_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if human_df.empty:
        return pd.DataFrame(rows)

    group_cols = ["rq", "dataset", "representation", "patch_view"]

    for group_key, g in human_df.groupby(group_cols, dropna=False):
        raters = sorted(g["rater"].dropna().unique())

        if len(raters) < 2:
            continue

        for i in range(len(raters)):
            for j in range(i + 1, len(raters)):
                r1 = raters[i]
                r2 = raters[j]

                a = g[g["rater"] == r1]
                b = g[g["rater"] == r2]

                merged = a.merge(
                    b,
                    on=KEYS,
                    suffixes=(f"_{r1}", f"_{r2}"),
                )

                if merged.empty:
                    continue

                for field in SCORE_FIELDS:
                    x_col = f"{field}_{r1}"
                    y_col = f"{field}_{r2}"

                    if x_col not in merged.columns or y_col not in merged.columns:
                        continue

                    x = merged[x_col]
                    y = merged[y_col]

                    s = safe_corr(x, y, "spearman")
                    k = safe_corr(x, y, "kendall")

                    rows.append(
                        {
                            "rq": group_key[0],
                            "dataset": group_key[1],
                            "representation": group_key[2],
                            "patch_view": group_key[3],
                            "metric": field,
                            "rater_a": r1,
                            "rater_b": r2,
                            "n": int((x.notna() & y.notna()).sum()),
                            "spearman": s["value"],
                            "spear_p": s["pvalue"],
                            "kendall": k["value"],
                            "kend_p": k["pvalue"],
                            "mae": mae(x, y),
                            "rmse": rmse(x, y),
                        }
                    )

    return pd.DataFrame(rows)


def human_llm_agreement(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    *,
    include_human_avg: bool = True,
) -> pd.DataFrame:
    rows = []

    if human_df.empty or llm_df.empty:
        return pd.DataFrame(rows)

    human_refs = []

    # Individual human references.
    for rater, g in human_df.groupby("rater", dropna=False):
        h = g.copy()
        h["human_ref"] = rater

        for field in SCORE_FIELDS:
            col = score_col(h, field, "raw")
            h[f"{field}_ref"] = h[col]

        human_refs.append(h)

    # HumanAvg reference.
    if include_human_avg:
        avg = human_average(human_df)

        if not avg.empty:
            avg["human_ref"] = "HumanAvg"

            for field in SCORE_FIELDS:
                mean_col = f"{field}_mean"
                if mean_col not in avg.columns:
                    raise KeyError(
                        f"Missing expected human average column: {mean_col}. "
                        f"Available columns: {list(avg.columns)}"
                    )
                avg[f"{field}_ref"] = avg[mean_col]

            human_refs.append(avg)

    if not human_refs:
        return pd.DataFrame(rows)

    human_ref_df = pd.concat(human_refs, ignore_index=True, sort=False)

    merged = human_ref_df.merge(
        llm_df,
        on=KEYS,
        suffixes=("_human", "_llm"),
    )

    if merged.empty:
        return pd.DataFrame(rows)

    group_cols = [
        "rq",
        "dataset",
        "representation",
        "patch_view",
        "human_ref",
        "model",
    ]

    for group_key, g in merged.groupby(group_cols, dropna=False):
        for field in SCORE_FIELDS:
            x_col = score_col(g, field, "ref")
            y_col = score_col(g, field, "llm")

            x = g[x_col]
            y = g[y_col]

            s = safe_corr(x, y, "spearman")
            k = safe_corr(x, y, "kendall")

            rows.append(
                {
                    "rq": group_key[0],
                    "dataset": group_key[1],
                    "representation": group_key[2],
                    "patch_view": group_key[3],
                    "human": group_key[4],
                    "model": group_key[5],
                    "metric": field,
                    "n": int((x.notna() & y.notna()).sum()),
                    "spearman": s["value"],
                    "spear_p": s["pvalue"],
                    "kendall": k["value"],
                    "kend_p": k["pvalue"],
                    "mae": mae(y, x),
                    "rmse": rmse(y, x),
                    "mean_bias": mean_bias(y, x),
                }
            )

    return pd.DataFrame(rows)


def unmatched_debug(human_df: pd.DataFrame, llm_df: pd.DataFrame):
    if human_df.empty:
        human_keys = pd.DataFrame(columns=KEYS)
    else:
        human_keys = human_df[KEYS].drop_duplicates()

    if llm_df.empty:
        llm_keys = pd.DataFrame(columns=KEYS)
    else:
        llm_keys = llm_df[KEYS].drop_duplicates()

    h_without_l = human_keys.merge(llm_keys, on=KEYS, how="left", indicator=True)
    h_without_l = h_without_l[h_without_l["_merge"] == "left_only"].drop(columns=["_merge"])

    l_without_h = llm_keys.merge(human_keys, on=KEYS, how="left", indicator=True)
    l_without_h = l_without_h[l_without_h["_merge"] == "left_only"].drop(columns=["_merge"])

    return h_without_l, l_without_h
