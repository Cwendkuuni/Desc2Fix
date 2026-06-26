#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate Plotly HD figures for the Desc2Fix human-evaluation package.

This script intentionally uses Plotly only (no matplotlib), following the
paper-style plotting conventions used in the existing Desc2Fix scripts:
  - white background
  - large fonts for LaTeX readability
  - HTML + PNG + PDF export
  - high-resolution static export through Kaleido
  - canonical ordering of datasets, models, metrics, representations, and patch views

Inputs:
  results/tables/human_evaluation/global_human_evaluation_summary.csv
  results/tables/human_evaluation/rq1_llm_vs_human_summary.csv
  results/tables/human_evaluation/rq1_human_pairwise_summary.csv
  results/tables/human_evaluation/rq1_llm_pairwise_summary.csv
  results/tables/human_evaluation/RQ2_Report_Test_human_llm_agreement.csv
  results/tables/human_evaluation/RQ3_Report_Patch_human_llm_agreement.csv
  results/tables/human_evaluation/RQ4_Test_Patch_Internal_human_llm_agreement.csv
  results/tables/human_evaluation/*_debug_summary.csv

Outputs:
  results/figures/human_evaluation_plotly/*.html
  results/figures/human_evaluation_plotly/*.png
  results/figures/human_evaluation_plotly/*.pdf

Usage:
  python scripts/generate_plotly_figures.py

Optional:
  python scripts/generate_plotly_figures.py --formats html png pdf --scale 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------
# Canonical orders and labels
# ---------------------------------------------------------------------

DATASETS = ["Defects4J", "SWTBench"]
MODELS = ["gpt4o", "deepseek"]
MODEL_LABEL = {
    "gpt4o": "GPT-4o",
    "deepseek": "DeepSeek",
}

REPRESENTATIONS = ["RAW", "STRUCTURED"]
PATCH_VIEWS = ["full", "add", "remove"]

METRICS = ["alignment_score", "coverage", "correctness", "specificity"]
METRIC_LABEL = {
    "alignment_score": "Alignment",
    "coverage": "Coverage",
    "correctness": "Correctness",
    "specificity": "Specificity",
}

SIGNAL_FIELDS = [
    "exception",
    "api_involved",
    "input_conditions",
    "reproduction_steps",
    "expected",
    "actual",
]
SIGNAL_FIELD_LABEL = {
    "exception": "Exception",
    "api_involved": "API",
    "input_conditions": "Input",
    "reproduction_steps": "Repro",
    "expected": "Expected",
    "actual": "Actual",
}

RQ_ORDER = ["rq1", "rq2", "rq3", "rq4_internal"]
RQ_LABEL = {
    "rq1": "RQ1 Extraction",
    "rq2": "RQ2 Report-Test",
    "rq3": "RQ3 Report-Patch",
    "rq4_internal": "RQ4 Test-Patch",
}


# ---------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------

PLOTLY_TEMPLATE = "plotly_white"

COLOR_SEQUENCE = px.colors.qualitative.Set2
ALT_COLOR_SEQUENCE = px.colors.qualitative.Pastel
HEATMAP_SCALE = "Viridis"
DIVERGING_SCALE = "RdBu_r"

BASE_FONT_SIZE = 22
AXIS_FONT_SIZE = 22
TICK_FONT_SIZE = 20
LEGEND_FONT_SIZE = 20
TITLE_FONT_SIZE = 24
ANNOTATION_FONT_SIZE = 18

EXPORT_WIDTH = 1500
EXPORT_HEIGHT = 900


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Plotly HD figures for Desc2Fix human-evaluation results."
    )

    parser.add_argument(
        "--tables-dir",
        default="results/tables/human_evaluation",
        help="Directory containing consolidated human-evaluation tables.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/figures/human_evaluation_plotly",
        help="Directory where figures will be written.",
    )

    parser.add_argument(
        "--formats",
        nargs="+",
        default=["html", "png", "pdf"],
        choices=["html", "png", "pdf", "svg"],
        help="Output formats.",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=6,
        help="Kaleido export scale for static images.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=EXPORT_WIDTH,
        help="Default static export width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=EXPORT_HEIGHT,
        help="Default static export height.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        msg = f"[WARN] Missing file: {path}"
        if required:
            raise FileNotFoundError(msg)
        print(msg)
        return pd.DataFrame()
    return pd.read_csv(path)


def write_all(
    fig: go.Figure,
    base: Path,
    *,
    formats: Iterable[str],
    width: int = EXPORT_WIDTH,
    height: int = EXPORT_HEIGHT,
    scale: int = 6,
) -> None:
    """
    Export a Plotly figure to HTML and static formats.

    PNG/PDF/SVG require Kaleido. If static export fails, HTML is still kept.
    """
    ensure_dir(base.parent)

    fig.update_layout(width=width, height=height)

    for fmt in formats:
        out = base.with_suffix(f".{fmt}")

        if fmt == "html":
            fig.write_html(str(out), include_plotlyjs="cdn")
            print(f"[OK] {out}")
            continue

        try:
            fig.write_image(str(out), width=width, height=height, scale=scale)
            print(f"[OK] {out}")
        except Exception as e:
            print(f"[WARN] Static export failed for {out.name}: {e}")


def clean_string_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip().str.replace(r"[\t\n\r]", "", regex=True)
    return out


def numeric_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def safe_map_labels(series: pd.Series, mapping: dict) -> pd.Series:
    """
    Map labels safely even when the input series is a pandas Categorical.

    Pandas Categorical can raise:
      TypeError: Cannot set a Categorical with another, without identical categories
    when map(...).fillna(original_categorical) is used. Converting to string
    before mapping avoids this problem.
    """
    text_series = series.astype(str)
    return text_series.map(mapping).fillna(text_series)


def apply_base_layout(
    fig: go.Figure,
    *,
    title: str = "",
    legend_orientation: str = "h",
    showlegend: Optional[bool] = None,
    margin: Optional[dict] = None,
) -> go.Figure:
    if margin is None:
        margin = dict(t=90, b=95, l=110, r=70)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title, x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        font=dict(size=BASE_FONT_SIZE),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=margin,
        legend=dict(
            orientation=legend_orientation,
            y=1.14 if legend_orientation == "h" else 1.0,
            x=0.5 if legend_orientation == "h" else 1.02,
            xanchor="center" if legend_orientation == "h" else "left",
            yanchor="bottom" if legend_orientation == "h" else "top",
            font=dict(size=LEGEND_FONT_SIZE),
        ),
    )

    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)

    fig.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
        tickfont=dict(size=TICK_FONT_SIZE),
        title_font=dict(size=AXIS_FONT_SIZE),
        showgrid=False,
    )
    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=False,
        ticks="outside",
        tickfont=dict(size=TICK_FONT_SIZE),
        title_font=dict(size=AXIS_FONT_SIZE),
        gridcolor="rgba(0,0,0,0.12)",
        zeroline=False,
    )

    return fig


def label_models(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    out = df.copy()
    if model_col in out.columns:
        out["model_label"] = safe_map_labels(out[model_col], MODEL_LABEL)
    return out


def safe_slug(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("↔", "to")
        .replace(":", "")
    )


# ---------------------------------------------------------------------
# Figure 0: overview row counts
# ---------------------------------------------------------------------

def fig_overview_row_counts(
    global_df: pd.DataFrame,
    outdir: Path,
    formats: list[str],
    width: int,
    height: int,
    scale: int,
) -> None:
    counts = (
        global_df.groupby(["rq", "rq_label"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    counts["rq_display"] = counts["rq"].astype(str).map(RQ_LABEL).fillna(counts["rq_label"].astype(str))
    counts["rq"] = pd.Categorical(counts["rq"], categories=RQ_ORDER, ordered=True)
    counts = counts.sort_values("rq")

    fig = px.bar(
        counts,
        x="rq_display",
        y="rows",
        text=counts["rows"].astype(str),
        color="rq_display",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="outside", textfont_size=ANNOTATION_FONT_SIZE)
    fig.update_layout(showlegend=False, bargap=0.25)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Rows", rangemode="tozero")
    apply_base_layout(fig, title="")

    write_all(
        fig,
        outdir / "fig00_human_eval_row_counts",
        formats=formats,
        width=1200,
        height=720,
        scale=scale,
    )


# ---------------------------------------------------------------------
# RQ1 figures
# ---------------------------------------------------------------------

def fig_rq1_llm_vs_human_bars(
    rq1_df: pd.DataFrame,
    outdir: Path,
    formats: list[str],
    width: int,
    height: int,
    scale: int,
) -> None:
    if rq1_df.empty:
        print("[WARN] RQ1 table is empty; skipping RQ1 figures.")
        return

    df = rq1_df[
        (rq1_df["level"] == "macro")
        & (rq1_df["right_source"] == "HumanAvg")
    ].copy()

    if df.empty:
        print("[WARN] No RQ1 macro HumanAvg rows; skipping RQ1 bars.")
        return

    df = label_models(df, "left_source")
    df["model_label"] = safe_map_labels(df["left_source"], MODEL_LABEL)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASETS, ordered=True)
    df["model_label"] = pd.Categorical(
        df["model_label"],
        categories=[MODEL_LABEL[m] for m in MODELS],
        ordered=True,
    )
    df = df.sort_values(["dataset", "model_label"])

    long_df = df.melt(
        id_vars=["dataset", "model_label", "n"],
        value_vars=["similarity_mean", "presence_agreement_mean"],
        var_name="measure",
        value_name="value",
    )
    long_df["measure_label"] = long_df["measure"].map(
        {
            "similarity_mean": "Signal similarity",
            "presence_agreement_mean": "Presence agreement",
        }
    )

    fig = px.bar(
        long_df,
        x="dataset",
        y="value",
        color="model_label",
        barmode="group",
        facet_col="measure_label",
        text=long_df["value"].round(3).astype(str),
        category_orders={
            "dataset": DATASETS,
            "model_label": [MODEL_LABEL[m] for m in MODELS],
            "measure_label": ["Signal similarity", "Presence agreement"],
        },
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="outside", textfont_size=ANNOTATION_FONT_SIZE)
    fig.update_yaxes(range=[0, 1], title="Score")
    fig.update_xaxes(title="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=20)))
    apply_base_layout(fig, title="")
    fig.update_layout(bargap=0.18, bargroupgap=0.08, legend_title_text="Model")

    write_all(
        fig,
        outdir / "fig01_rq1_llm_vs_human_similarity_presence",
        formats=formats,
        width=1500,
        height=780,
        scale=scale,
    )


def fig_rq1_human_and_llm_pairwise(
    human_pairwise_df: pd.DataFrame,
    llm_pairwise_df: pd.DataFrame,
    outdir: Path,
    formats: list[str],
    width: int,
    height: int,
    scale: int,
) -> None:
    parts = []

    if not human_pairwise_df.empty:
        h = human_pairwise_df[human_pairwise_df["level"] == "macro"].copy()
        h["comparison"] = "Human1 vs Human2"
        parts.append(
            h[["dataset", "comparison", "similarity_mean", "presence_agreement_mean"]]
        )

    if not llm_pairwise_df.empty:
        l = llm_pairwise_df[llm_pairwise_df["level"] == "macro"].copy()
        l["comparison"] = "DeepSeek vs GPT-4o"
        parts.append(
            l[["dataset", "comparison", "similarity_mean", "presence_agreement_mean"]]
        )

    if not parts:
        print("[WARN] No RQ1 pairwise summaries; skipping pairwise figure.")
        return

    df = pd.concat(parts, ignore_index=True)
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASETS, ordered=True)
    df = df.sort_values(["dataset", "comparison"])

    long_df = df.melt(
        id_vars=["dataset", "comparison"],
        value_vars=["similarity_mean", "presence_agreement_mean"],
        var_name="measure",
        value_name="value",
    )
    long_df["measure_label"] = long_df["measure"].map(
        {
            "similarity_mean": "Signal similarity",
            "presence_agreement_mean": "Presence agreement",
        }
    )

    fig = px.bar(
        long_df,
        x="dataset",
        y="value",
        color="comparison",
        barmode="group",
        facet_col="measure_label",
        text=long_df["value"].round(3).astype(str),
        category_orders={
            "dataset": DATASETS,
            "measure_label": ["Signal similarity", "Presence agreement"],
        },
        color_discrete_sequence=ALT_COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="outside", textfont_size=ANNOTATION_FONT_SIZE)
    fig.update_yaxes(range=[0, 1], title="Score")
    fig.update_xaxes(title="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=20)))
    apply_base_layout(fig, title="")
    fig.update_layout(bargap=0.18, bargroupgap=0.08, legend_title_text="Comparison")

    write_all(
        fig,
        outdir / "fig02_rq1_pairwise_human_and_llm",
        formats=formats,
        width=1500,
        height=780,
        scale=scale,
    )


# ---------------------------------------------------------------------
# Generic HumanAvg agreement helpers
# ---------------------------------------------------------------------

def filter_human_avg(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = clean_string_columns(
        out,
        ["dataset", "representation", "patch_view", "metric", "human", "model"],
    )
    out = numeric_columns(
        out,
        ["n", "spearman", "kendall", "mae", "rmse", "mean_bias"],
    )
    return out[out["human"] == "HumanAvg"].copy()


def aggregate_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate over HumanAvg rows while keeping dataset, representation,
    patch view, metric and model. The input is already HumanAvg-only.
    """
    group_cols = ["dataset", "representation", "patch_view", "metric", "model"]

    present = [c for c in group_cols if c in df.columns]
    return (
        df.groupby(present, dropna=False, observed=False)
        .agg(
            spearman=("spearman", "mean"),
            kendall=("kendall", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mean_bias=("mean_bias", "mean"),
            n=("n", "sum"),
        )
        .reset_index()
    )


def prepare_alignment_table(df: pd.DataFrame) -> pd.DataFrame:
    out = filter_human_avg(df)
    if out.empty:
        return out

    out["dataset"] = pd.Categorical(out["dataset"], categories=DATASETS, ordered=True)
    out["metric"] = pd.Categorical(out["metric"], categories=METRICS, ordered=True)
    out["model"] = pd.Categorical(out["model"], categories=MODELS, ordered=True)

    if "representation" in out.columns:
        out["representation"] = pd.Categorical(
            out["representation"],
            categories=REPRESENTATIONS,
            ordered=True,
        )
    if "patch_view" in out.columns:
        out["patch_view"] = pd.Categorical(
            out["patch_view"],
            categories=["none"] + PATCH_VIEWS,
            ordered=True,
        )

    out = out.sort_values(
        [c for c in ["dataset", "representation", "patch_view", "metric", "model"] if c in out.columns]
    )
    out["model_label"] = safe_map_labels(out["model"], MODEL_LABEL)
    out["metric_label"] = safe_map_labels(out["metric"], METRIC_LABEL)

    return out


# ---------------------------------------------------------------------
# RQ2 figures
# ---------------------------------------------------------------------

def fig_rq2_spearman_heatmap(
    rq2_df: pd.DataFrame,
    outdir: Path,
    formats: list[str],
    width: int,
    height: int,
    scale: int,
) -> None:
    df = prepare_alignment_table(rq2_df)
    if df.empty:
        print("[WARN] RQ2 HumanAvg table empty; skipping RQ2 heatmap.")
        return

    # Row = dataset/model, columns = representation, panels = metrics.
    df["row_label"] = (
        df["dataset"].astype(str) + " / " + safe_map_labels(df["model"], MODEL_LABEL)
    )
    row_order = [
        f"{ds} / {MODEL_LABEL[m]}"
        for ds in DATASETS
        for m in MODELS
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[METRIC_LABEL[m] for m in METRICS],
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for idx, metric in enumerate(METRICS, start=1):
        r = 1 if idx <= 2 else 2
        c = 1 if idx in [1, 3] else 2

        block = df[df["metric"].astype(str) == metric]
        mat = (
            block.pivot_table(
                index="row_label",
                columns="representation",
                values="spearman",
                aggfunc="mean",
                observed=False,
            )
            .reindex(index=row_order, columns=REPRESENTATIONS)
        )

        z = mat.values.astype(float)
        text = np.empty_like(z, dtype=object)
        for yi in range(z.shape[0]):
            for xi in range(z.shape[1]):
                val = z[yi, xi]
                text[yi, xi] = f"ρ={val:.2f}" if pd.notna(val) else ""

        hm = go.Heatmap(
            z=z,
            x=REPRESENTATIONS,
            y=row_order,
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14},
            zmin=-1,
            zmax=1,
            colorscale=HEATMAP_SCALE,
            colorbar=dict(title="ρ"),
            showscale=(idx == 2),
            hovertemplate="Row=%{y}<br>Representation=%{x}<br>ρ=%{z:.3f}<extra></extra>",
        )
        fig.add_trace(hm, row=r, col=c)

        fig.update_yaxes(showticklabels=(c == 1), row=r, col=c)
        fig.update_xaxes(showticklabels=True, row=r, col=c)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text="", x=0.5, font=dict(size=TITLE_FONT_SIZE)),
        font=dict(size=18),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=190, r=80, t=80, b=70),
    )

    write_all(
        fig,
        outdir / "fig03_rq2_spearman_heatmap_humanavg",
        formats=formats,
        width=1500,
        height=950,
        scale=scale,
    )


def fig_rq2_mae_bars(
    rq2_df: pd.DataFrame,
    outdir: Path,
    formats: list[str],
    width: int,
    height: int,
    scale: int,
) -> None:
    df = prepare_alignment_table(rq2_df)
    if df.empty:
        print("[WARN] RQ2 HumanAvg table empty; skipping RQ2 MAE bars.")
        return

    fig = px.bar(
        df,
        x="metric_label",
        y="mae",
        color="model_label",
        barmode="group",
        facet_col="representation",
        facet_row="dataset",
        text=df["mae"].round(2).astype(str),
        category_orders={
            "dataset": DATASETS,
            "representation": REPRESENTATIONS,
            "metric_label": [METRIC_LABEL[m] for m in METRICS],
            "model_label": [MODEL_LABEL[m] for m in MODELS],
        },
        color_discrete_sequence=COLOR_SEQUENCE,
        hover_data=["n", "spearman", "rmse", "mean_bias"],
    )
    fig.update_traces(textposition="outside", textfont_size=13)
    fig.update_yaxes(title="MAE")
    fig.update_xaxes(title="", tickangle=-20)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=18)))
    apply_base_layout(fig, title="", margin=dict(t=100, b=130, l=110, r=70))
    fig.update_layout(legend_title_text="Model")

    write_all(
        fig,
        outdir / "fig04_rq2_mae_grid_humanavg",
        formats=formats,
        width=1600,
        height=950,
        scale=scale,
    )


# ---------------------------------------------------------------------
# RQ3/RQ4 heatmaps and bars
# ---------------------------------------------------------------------

def figure_patch_spearman_heatmaps(
    df_in: pd.DataFrame,
    *,
    rq_name: str,
    output_stem_prefix: str,
    outdir: Path,
    formats: list[str],
    width: int,
    height: int,
    scale: int,
    representations: list[str],
    patch_views: list[str],
) -> None:
    df = prepare_alignment_table(df_in)
    if df.empty:
        print(f"[WARN] {rq_name} HumanAvg table empty; skipping heatmaps.")
        return

    for ds in DATASETS:
        sub = df[df["dataset"].astype(str) == ds].copy()
        if sub.empty:
            continue

        fig = make_subplots(
            rows=len(representations),
            cols=len(patch_views),
            subplot_titles=[
                f"{rep} / {view}"
                for rep in representations
                for view in patch_views
            ],
            horizontal_spacing=0.05,
            vertical_spacing=0.12,
            x_title="Model",
            y_title="Metric",
        )

        for i, rep in enumerate(representations, start=1):
            for j, view in enumerate(patch_views, start=1):
                block = sub[
                    (sub["representation"].astype(str) == rep)
                    & (sub["patch_view"].astype(str) == view)
                ].copy()

                mat = (
                    block.pivot_table(
                        index="metric",
                        columns="model",
                        values="spearman",
                        aggfunc="mean",
                        observed=False,
                    )
                    .reindex(index=METRICS, columns=MODELS)
                )

                z = mat.values.astype(float)
                text = np.empty_like(z, dtype=object)
                for yi in range(z.shape[0]):
                    for xi in range(z.shape[1]):
                        val = z[yi, xi]
                        text[yi, xi] = f"ρ={val:.2f}" if pd.notna(val) else ""

                hm = go.Heatmap(
                    z=z,
                    x=[MODEL_LABEL[m] for m in MODELS],
                    y=[METRIC_LABEL[m] for m in METRICS],
                    text=text,
                    texttemplate="%{text}",
                    textfont={"size": 14},
                    zmin=-1,
                    zmax=1,
                    colorscale=HEATMAP_SCALE,
                    colorbar=dict(title="ρ"),
                    showscale=(i == 1 and j == len(patch_views)),
                    hovertemplate="Metric=%{y}<br>Model=%{x}<br>ρ=%{z:.3f}<extra></extra>",
                )
                fig.add_trace(hm, row=i, col=j)
                fig.update_yaxes(showticklabels=(j == 1), row=i, col=j)
                fig.update_xaxes(showticklabels=True, row=i, col=j)

        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            title=dict(text="", x=0.5, font=dict(size=TITLE_FONT_SIZE)),
            font=dict(size=18),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=130, r=80, t=90, b=80),
        )

        write_all(
            fig,
            outdir / f"{output_stem_prefix}_spearman_heatmap_{safe_slug(ds)}",
            formats=formats,
            width=1600,
            height=1000 if len(representations) > 1 else 760,
            scale=scale,
        )


def figure_patch_mae_grid(
    df_in: pd.DataFrame,
    *,
    rq_name: str,
    output_stem_prefix: str,
    outdir: Path,
    formats: list[str],
    width: int,
    height: int,
    scale: int,
) -> None:
    df = prepare_alignment_table(df_in)
    if df.empty:
        print(f"[WARN] {rq_name} HumanAvg table empty; skipping MAE grid.")
        return

    df = df[df["patch_view"].astype(str).isin(PATCH_VIEWS)].copy()
    if df.empty:
        print(f"[WARN] {rq_name} has no patch views; skipping MAE grid.")
        return

    for ds in DATASETS:
        sub = df[df["dataset"].astype(str) == ds].copy()
        if sub.empty:
            continue

        fig = px.bar(
            sub,
            x="metric_label",
            y="mae",
            color="model_label",
            barmode="group",
            facet_row="representation",
            facet_col="patch_view",
            text=sub["mae"].round(2).astype(str),
            category_orders={
                "representation": REPRESENTATIONS,
                "patch_view": PATCH_VIEWS,
                "metric_label": [METRIC_LABEL[m] for m in METRICS],
                "model_label": [MODEL_LABEL[m] for m in MODELS],
            },
            color_discrete_sequence=COLOR_SEQUENCE,
            hover_data=["n", "spearman", "rmse", "mean_bias"],
        )
        fig.update_traces(textposition="outside", textfont_size=12)
        fig.update_yaxes(title="MAE")
        fig.update_xaxes(title="", tickangle=-25)
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=18)))
        apply_base_layout(fig, title="", margin=dict(t=105, b=135, l=105, r=65))
        fig.update_layout(legend_title_text="Model")

        write_all(
            fig,
            outdir / f"{output_stem_prefix}_mae_grid_{safe_slug(ds)}",
            formats=formats,
            width=1700,
            height=1050,
            scale=scale,
        )


def figure_bias_vs_agreement(
    df_in: pd.DataFrame,
    *,
    rq_name: str,
    output_stem_prefix: str,
    outdir: Path,
    formats: list[str],
    width: int,
    height: int,
    scale: int,
) -> None:
    df = prepare_alignment_table(df_in)
    if df.empty:
        print(f"[WARN] {rq_name} HumanAvg table empty; skipping bias scatter.")
        return

    df = df[df["metric"].astype(str).isin(METRICS)].copy()
    if df.empty:
        return

    for ds in DATASETS:
        sub = df[df["dataset"].astype(str) == ds].copy()
        if sub.empty:
            continue

        fig = px.scatter(
            sub,
            x="spearman",
            y="mean_bias",
            color="patch_view" if "patch_view" in sub.columns else "representation",
            symbol="representation" if "representation" in sub.columns else "model_label",
            size="n",
            facet_col="metric_label",
            facet_col_wrap=2,
            hover_data=["model_label", "representation", "patch_view", "n", "mae", "rmse"],
            color_discrete_sequence=COLOR_SEQUENCE,
        )
        fig.add_hline(y=0, line_width=1, line_dash="dot")
        fig.add_vline(x=0, line_width=1, line_dash="dot")
        fig.update_xaxes(title="Spearman ρ")
        fig.update_yaxes(title="Mean bias")
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=18)))
        apply_base_layout(fig, title="", margin=dict(t=110, b=100, l=110, r=70))
        fig.update_layout(legend_title_text="Patch view")

        write_all(
            fig,
            outdir / f"{output_stem_prefix}_bias_vs_agreement_{safe_slug(ds)}",
            formats=formats,
            width=1600,
            height=950,
            scale=scale,
        )


# ---------------------------------------------------------------------
# Global RQ comparison figures
# ---------------------------------------------------------------------

def fig_global_spearman_mae(
    global_df: pd.DataFrame,
    outdir: Path,
    formats: list[str],
    width: int,
    height: int,
    scale: int,
) -> None:
    df = global_df[
        (global_df["rq"].isin(["rq2", "rq3", "rq4_internal"]))
        & (global_df["human"] == "HumanAvg")
    ].copy()

    if df.empty:
        print("[WARN] No global HumanAvg alignment rows; skipping global bars.")
        return

    df["model_label"] = safe_map_labels(df["model"], MODEL_LABEL)
    df["rq_display"] = df["rq"].astype(str).map(RQ_LABEL).fillna(df["rq_label"].astype(str))

    spear = (
        df[df["primary_value_name"] == "spearman"]
        .groupby(["rq", "rq_display", "model_label"], dropna=False)["primary_value"]
        .mean()
        .reset_index(name="value")
    )
    spear["measure"] = "Mean Spearman ρ"

    mae = (
        df[df["secondary_value_name"] == "mae"]
        .groupby(["rq", "rq_display", "model_label"], dropna=False)["secondary_value"]
        .mean()
        .reset_index(name="value")
    )
    mae["measure"] = "Mean MAE"

    plot_df = pd.concat([spear, mae], ignore_index=True)
    plot_df["rq"] = pd.Categorical(plot_df["rq"], categories=RQ_ORDER, ordered=True)
    plot_df = plot_df.sort_values(["measure", "rq", "model_label"])

    fig = px.bar(
        plot_df,
        x="rq_display",
        y="value",
        color="model_label",
        barmode="group",
        facet_col="measure",
        text=plot_df["value"].round(2).astype(str),
        category_orders={
            "rq_display": [RQ_LABEL[r] for r in ["rq2", "rq3", "rq4_internal"]],
            "model_label": [MODEL_LABEL[m] for m in MODELS],
            "measure": ["Mean Spearman ρ", "Mean MAE"],
        },
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="outside", textfont_size=ANNOTATION_FONT_SIZE)
    fig.update_xaxes(title="", tickangle=0)
    fig.update_yaxes(title="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=20)))
    apply_base_layout(fig, title="")
    fig.update_layout(legend_title_text="Model")

    write_all(
        fig,
        outdir / "fig09_global_spearman_mae_by_rq",
        formats=formats,
        width=1500,
        height=780,
        scale=scale,
    )


# ---------------------------------------------------------------------
# Debug figure
# ---------------------------------------------------------------------

def read_debug_summaries(tables_dir: Path) -> pd.DataFrame:
    specs = [
        ("RQ1 Extraction", tables_dir / "rq1_debug_summary.csv"),
        ("RQ2 Report-Test", tables_dir / "RQ2_Report_Test_debug_summary.csv"),
        ("RQ3 Report-Patch", tables_dir / "RQ3_Report_Patch_debug_summary.csv"),
        ("RQ4 Test-Patch", tables_dir / "RQ4_Test_Patch_Internal_debug_summary.csv"),
    ]

    parts = []
    for rq_display, path in specs:
        df = read_csv(path, required=False)
        if df.empty:
            continue

        df = df.copy()
        df["rq_display"] = rq_display
        df["rows"] = pd.to_numeric(df.get("rows", 0), errors="coerce").fillna(0)

        if "source" in df.columns:
            df["debug_type"] = df["source"].astype(str).apply(
                lambda x: "Human without LLM"
                if "human_without_llm" in x
                else ("LLM without human" if "llm_without_human" in x else "Other")
            )
        else:
            df["debug_type"] = "Other"

        parts.append(df[["rq_display", "debug_type", "rows"]])

    if not parts:
        return pd.DataFrame(columns=["rq_display", "debug_type", "rows"])

    out = pd.concat(parts, ignore_index=True)
    return (
        out.groupby(["rq_display", "debug_type"], dropna=False)["rows"]
        .sum()
        .reset_index()
    )


def fig_debug_unmatched(
    tables_dir: Path,
    outdir: Path,
    formats: list[str],
    width: int,
    height: int,
    scale: int,
) -> None:
    df = read_debug_summaries(tables_dir)
    if df.empty:
        print("[WARN] Debug summaries empty; skipping debug figure.")
        return

    rq_order = ["RQ1 Extraction", "RQ2 Report-Test", "RQ3 Report-Patch", "RQ4 Test-Patch"]
    df["rq_display"] = pd.Categorical(df["rq_display"], categories=rq_order, ordered=True)
    df = df.sort_values(["rq_display", "debug_type"])

    fig = px.bar(
        df,
        x="rq_display",
        y="rows",
        color="debug_type",
        barmode="stack",
        text=df["rows"].astype(int).astype(str),
        category_orders={"rq_display": rq_order},
        color_discrete_sequence=ALT_COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="outside", textfont_size=ANNOTATION_FONT_SIZE)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Rows", rangemode="tozero")
    apply_base_layout(fig, title="")
    fig.update_layout(legend_title_text="Debug type")

    write_all(
        fig,
        outdir / "fig10_debug_unmatched_cases",
        formats=formats,
        width=1300,
        height=760,
        scale=scale,
    )


# ---------------------------------------------------------------------
# README index for generated figures
# ---------------------------------------------------------------------

def write_index(outdir: Path) -> None:
    ensure_dir(outdir)

    lines = [
        "# Desc2Fix Human Evaluation Plotly Figures",
        "",
        "Generated with:",
        "",
        "```bash",
        "python scripts/generate_plotly_figures.py",
        "```",
        "",
        "This directory contains Plotly HD figures exported as HTML and static image files.",
        "",
        "Main figures:",
        "",
        "- `fig00_human_eval_row_counts`: global row counts by RQ.",
        "- `fig01_rq1_llm_vs_human_similarity_presence`: RQ1 LLM-vs-HumanAvg extraction agreement.",
        "- `fig02_rq1_pairwise_human_and_llm`: RQ1 human-human and model-model agreement.",
        "- `fig03_rq2_spearman_heatmap_humanavg`: RQ2 HumanAvg Spearman heatmap.",
        "- `fig04_rq2_mae_grid_humanavg`: RQ2 HumanAvg MAE grid.",
        "- `fig05_rq3_spearman_heatmap_<dataset>`: RQ3 HumanAvg Spearman heatmaps.",
        "- `fig06_rq3_mae_grid_<dataset>`: RQ3 HumanAvg MAE grids.",
        "- `fig07_rq4_spearman_heatmap_<dataset>`: RQ4 internal HumanAvg Spearman heatmaps.",
        "- `fig08_rq4_mae_grid_<dataset>`: RQ4 internal HumanAvg MAE grids.",
        "- `fig09_global_spearman_mae_by_rq`: global mean Spearman and MAE by RQ.",
        "- `fig10_debug_unmatched_cases`: unmatched debug cases.",
        "",
        "Notes:",
        "",
        "- Static export requires Kaleido.",
        "- HTML export does not require Kaleido.",
        "- Figures use large fonts and white backgrounds for LaTeX and paper readability.",
    ]

    path = outdir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    tables_dir = Path(args.tables_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    print(f"[Desc2Fix][Plotly] tables_dir={tables_dir.resolve()}")
    print(f"[Desc2Fix][Plotly] output_dir={output_dir.resolve()}")
    print(f"[Desc2Fix][Plotly] formats={args.formats}")
    print(f"[Desc2Fix][Plotly] scale={args.scale}")

    global_df = read_csv(tables_dir / "global_human_evaluation_summary.csv")
    rq1_llm = read_csv(tables_dir / "rq1_llm_vs_human_summary.csv", required=False)
    rq1_human_pairwise = read_csv(tables_dir / "rq1_human_pairwise_summary.csv", required=False)
    rq1_llm_pairwise = read_csv(tables_dir / "rq1_llm_pairwise_summary.csv", required=False)
    rq2 = read_csv(tables_dir / "RQ2_Report_Test_human_llm_agreement.csv", required=False)
    rq3 = read_csv(tables_dir / "RQ3_Report_Patch_human_llm_agreement.csv", required=False)
    rq4 = read_csv(tables_dir / "RQ4_Test_Patch_Internal_human_llm_agreement.csv", required=False)

    fig_overview_row_counts(
        global_df,
        output_dir,
        args.formats,
        args.width,
        args.height,
        args.scale,
    )

    fig_rq1_llm_vs_human_bars(
        rq1_llm,
        output_dir,
        args.formats,
        args.width,
        args.height,
        args.scale,
    )

    fig_rq1_human_and_llm_pairwise(
        rq1_human_pairwise,
        rq1_llm_pairwise,
        output_dir,
        args.formats,
        args.width,
        args.height,
        args.scale,
    )

    fig_rq2_spearman_heatmap(
        rq2,
        output_dir,
        args.formats,
        args.width,
        args.height,
        args.scale,
    )

    fig_rq2_mae_bars(
        rq2,
        output_dir,
        args.formats,
        args.width,
        args.height,
        args.scale,
    )

    figure_patch_spearman_heatmaps(
        rq3,
        rq_name="RQ3",
        output_stem_prefix="fig05_rq3",
        outdir=output_dir,
        formats=args.formats,
        width=args.width,
        height=args.height,
        scale=args.scale,
        representations=REPRESENTATIONS,
        patch_views=PATCH_VIEWS,
    )

    figure_patch_mae_grid(
        rq3,
        rq_name="RQ3",
        output_stem_prefix="fig06_rq3",
        outdir=output_dir,
        formats=args.formats,
        width=args.width,
        height=args.height,
        scale=args.scale,
    )

    figure_bias_vs_agreement(
        rq3,
        rq_name="RQ3",
        output_stem_prefix="fig06b_rq3",
        outdir=output_dir,
        formats=args.formats,
        width=args.width,
        height=args.height,
        scale=args.scale,
    )

    figure_patch_spearman_heatmaps(
        rq4,
        rq_name="RQ4 internal",
        output_stem_prefix="fig07_rq4",
        outdir=output_dir,
        formats=args.formats,
        width=args.width,
        height=args.height,
        scale=args.scale,
        representations=["RAW"],
        patch_views=PATCH_VIEWS,
    )

    figure_patch_mae_grid(
        rq4,
        rq_name="RQ4 internal",
        output_stem_prefix="fig08_rq4",
        outdir=output_dir,
        formats=args.formats,
        width=args.width,
        height=args.height,
        scale=args.scale,
    )

    figure_bias_vs_agreement(
        rq4,
        rq_name="RQ4 internal",
        output_stem_prefix="fig08b_rq4",
        outdir=output_dir,
        formats=args.formats,
        width=args.width,
        height=args.height,
        scale=args.scale,
    )

    fig_global_spearman_mae(
        global_df,
        output_dir,
        args.formats,
        args.width,
        args.height,
        args.scale,
    )

    fig_debug_unmatched(
        tables_dir,
        output_dir,
        args.formats,
        args.width,
        args.height,
        args.scale,
    )

    write_index(output_dir)

    print("[DONE] Plotly figures generated.")


if __name__ == "__main__":
    main()
