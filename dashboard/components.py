#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reusable Streamlit/Plotly components for the Desc2Fix dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


DATASETS = ["Defects4J", "SWTBench"]
MODELS = ["gpt4o", "deepseek"]
MODEL_LABEL = {"gpt4o": "GPT-4o", "deepseek": "DeepSeek"}

REPRESENTATIONS = ["RAW", "STRUCTURED"]
PATCH_VIEWS = ["full", "add", "remove"]
METRICS = ["alignment_score", "coverage", "correctness", "specificity"]

METRIC_LABEL = {
    "alignment_score": "Alignment",
    "coverage": "Coverage",
    "correctness": "Correctness",
    "specificity": "Specificity",
}

RQ_LABEL = {
    "rq1": "RQ1 Extraction",
    "rq2": "RQ2 Report-Test",
    "rq3": "RQ3 Report-Patch",
    "rq4_internal": "RQ4 Test-Patch",
}

COLOR_SEQUENCE = px.colors.qualitative.Set2
ALT_COLOR_SEQUENCE = px.colors.qualitative.Pastel
HEATMAP_SCALE = "Viridis"
PLOTLY_TEMPLATE = "plotly_white"


def page_header(title: str, caption: str | None = None) -> None:
    st.title(title)
    if caption:
        st.caption(caption)


def safe_model_label(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.map(MODEL_LABEL).fillna(s)


def safe_metric_label(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.map(METRIC_LABEL).fillna(s)


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
    return out


def apply_style(fig: go.Figure, height: int = 560) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        font=dict(size=15),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=60, b=75, l=75, r=40),
        legend=dict(
            orientation="h",
            y=1.14,
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=13),
        ),
    )
    fig.update_xaxes(showline=True, linecolor="black", ticks="outside", showgrid=False)
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside", gridcolor="rgba(0,0,0,0.12)")
    return fig


def dataframe_with_download(df: pd.DataFrame, filename: str, height: int = 420) -> None:
    if df.empty:
        st.info("No data available.")
        return

    st.dataframe(df, use_container_width=True, height=height)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def global_metric_cards(global_df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Global rows", f"{len(global_df):,}")
    c2.metric("Research questions", global_df["rq"].nunique() if "rq" in global_df.columns else 0)
    c3.metric("Datasets", global_df["dataset"].nunique() if "dataset" in global_df.columns else 0)
    c4.metric("Models", global_df["model"].nunique() if "model" in global_df.columns else 0)


def row_count_chart(global_df: pd.DataFrame) -> Optional[go.Figure]:
    if global_df.empty:
        return None

    counts = global_df.groupby(["rq", "rq_label"], dropna=False).size().reset_index(name="rows")
    counts["rq_display"] = counts["rq"].astype(str).map(RQ_LABEL).fillna(counts["rq_label"].astype(str))

    fig = px.bar(
        counts,
        x="rq_display",
        y="rows",
        color="rq_display",
        text=counts["rows"].astype(str),
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Rows")
    return apply_style(fig, height=500)


def rq1_extraction_chart(
    rq1_df: pd.DataFrame,
    datasets: list[str],
    models: list[str],
) -> Optional[go.Figure]:
    if rq1_df.empty:
        return None

    df = rq1_df[
        (rq1_df["level"] == "macro")
        & (rq1_df["right_source"] == "HumanAvg")
    ].copy()

    if datasets:
        df = df[df["dataset"].isin(datasets)]
    if models:
        df = df[df["left_source"].isin(models)]

    if df.empty:
        return None

    df["model_label"] = df["left_source"].astype(str).map(MODEL_LABEL).fillna(df["left_source"].astype(str))

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
    fig.update_traces(textposition="outside")
    fig.update_yaxes(range=[0, 1], title="Score")
    fig.update_xaxes(title="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return apply_style(fig, height=560)


def filter_alignment_table(
    df: pd.DataFrame,
    *,
    datasets: list[str],
    models: list[str],
    humans: list[str],
    metrics: list[str],
    representations: Optional[list[str]] = None,
    patch_views: Optional[list[str]] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = clean_strings(df)

    if datasets and "dataset" in out.columns:
        out = out[out["dataset"].isin(datasets)]
    if models and "model" in out.columns:
        out = out[out["model"].isin(models)]
    if humans and "human" in out.columns:
        out = out[out["human"].isin(humans)]
    if metrics and "metric" in out.columns:
        out = out[out["metric"].isin(metrics)]
    if representations and "representation" in out.columns:
        out = out[out["representation"].isin(representations)]
    if patch_views and "patch_view" in out.columns:
        out = out[out["patch_view"].isin(patch_views)]

    return out


def alignment_summary_cards(df: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")

    if not df.empty and "spearman" in df.columns:
        c2.metric("Mean Spearman", f"{pd.to_numeric(df['spearman'], errors='coerce').mean():.3f}")
    else:
        c2.metric("Mean Spearman", "NA")

    if not df.empty and "mae" in df.columns:
        c3.metric("Mean MAE", f"{pd.to_numeric(df['mae'], errors='coerce').mean():.3f}")
    else:
        c3.metric("Mean MAE", "NA")

    if not df.empty and "n" in df.columns:
        c4.metric("Total N", f"{pd.to_numeric(df['n'], errors='coerce').sum():,.0f}")
    else:
        c4.metric("Total N", "NA")


def alignment_heatmap(df: pd.DataFrame, title: str = "") -> Optional[go.Figure]:
    if df.empty or "spearman" not in df.columns:
        return None

    plot_df = clean_strings(df)
    plot_df["row"] = plot_df["dataset"].astype(str) + " / " + safe_model_label(plot_df["model"])
    plot_df["metric_label"] = safe_metric_label(plot_df["metric"])

    if "representation" in plot_df.columns and plot_df["representation"].nunique() > 1:
        col_key = "representation"
        col_order = REPRESENTATIONS
    elif "patch_view" in plot_df.columns and plot_df["patch_view"].nunique() > 1:
        col_key = "patch_view"
        col_order = PATCH_VIEWS
    else:
        col_key = "metric_label"
        col_order = [METRIC_LABEL[m] for m in METRICS]

    row_order = [f"{ds} / {MODEL_LABEL[m]}" for ds in DATASETS for m in MODELS]

    mat = (
        plot_df.pivot_table(
            index="row",
            columns=col_key,
            values="spearman",
            aggfunc="mean",
            observed=False,
        )
        .reindex(index=row_order, columns=col_order)
    )

    fig = px.imshow(
        mat,
        text_auto=".2f",
        zmin=-1,
        zmax=1,
        color_continuous_scale=HEATMAP_SCALE,
        aspect="auto",
        title=title,
    )
    fig.update_layout(coloraxis_colorbar=dict(title="ρ"))
    fig.update_xaxes(title=col_key)
    fig.update_yaxes(title="")
    return apply_style(fig, height=560)


def alignment_bar_chart(df: pd.DataFrame, value: str, title: str) -> Optional[go.Figure]:
    if df.empty or value not in df.columns:
        return None

    plot_df = clean_strings(df)
    plot_df[value] = pd.to_numeric(plot_df[value], errors="coerce")
    plot_df["model_label"] = safe_model_label(plot_df["model"])
    plot_df["metric_label"] = safe_metric_label(plot_df["metric"])

    group_cols = ["dataset", "model_label", "metric_label"]

    if "representation" in plot_df.columns and plot_df["representation"].nunique() > 1:
        group_cols.append("representation")
    if "patch_view" in plot_df.columns and plot_df["patch_view"].nunique() > 1:
        group_cols.append("patch_view")

    summary = plot_df.groupby(group_cols, dropna=False)[value].mean().reset_index()

    facet_col = "representation" if "representation" in summary.columns and summary["representation"].nunique() <= 2 and summary["representation"].nunique() > 1 else None
    facet_row = "patch_view" if "patch_view" in summary.columns and summary["patch_view"].nunique() <= 3 and summary["patch_view"].nunique() > 1 else None

    fig = px.bar(
        summary,
        x="metric_label",
        y=value,
        color="model_label",
        barmode="group",
        facet_col=facet_col,
        facet_row=facet_row,
        text=summary[value].round(2).astype(str),
        category_orders={
            "metric_label": [METRIC_LABEL[m] for m in METRICS],
            "model_label": [MODEL_LABEL[m] for m in MODELS],
            "representation": REPRESENTATIONS,
            "patch_view": PATCH_VIEWS,
        },
        color_discrete_sequence=COLOR_SEQUENCE,
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(title="", tickangle=-20)
    fig.update_yaxes(title=value)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return apply_style(fig, height=720 if facet_row else 560)


def debug_chart(debug_df: pd.DataFrame) -> Optional[go.Figure]:
    if debug_df.empty:
        return None

    df = clean_strings(debug_df)
    if "source" in df.columns:
        df["debug_type"] = df["source"].astype(str).apply(
            lambda x: "Human without LLM"
            if "human_without_llm" in x
            else ("LLM without human" if "llm_without_human" in x else "Other")
        )
    else:
        df["debug_type"] = "Other"

    if "rows" not in df.columns:
        df["rows"] = 0

    df["rows"] = pd.to_numeric(df["rows"], errors="coerce").fillna(0)

    group_cols = ["rq_label", "debug_type"] if "rq_label" in df.columns else ["debug_type"]
    summary = df.groupby(group_cols, dropna=False)["rows"].sum().reset_index()

    x_col = "rq_label" if "rq_label" in summary.columns else "debug_type"

    fig = px.bar(
        summary,
        x=x_col,
        y="rows",
        color="debug_type",
        barmode="stack",
        text=summary["rows"].astype(int).astype(str),
        color_discrete_sequence=ALT_COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Rows", rangemode="tozero")
    return apply_style(fig, height=520)


def sidebar_alignment_filters(
    df: pd.DataFrame,
    page_key: str,
    *,
    allow_representations: bool,
    allow_patch_views: bool,
) -> dict:
    df = clean_strings(df)

    datasets = st.sidebar.multiselect(
        "Datasets",
        DATASETS,
        default=[d for d in DATASETS if d in set(df.get("dataset", pd.Series(dtype=str)).astype(str))] or DATASETS,
        key=f"{page_key}_datasets",
    )

    models = st.sidebar.multiselect(
        "Models",
        MODELS,
        default=[m for m in MODELS if m in set(df.get("model", pd.Series(dtype=str)).astype(str))] or MODELS,
        key=f"{page_key}_models",
        format_func=lambda x: MODEL_LABEL.get(x, x),
    )

    humans_available = sorted(df["human"].dropna().astype(str).unique().tolist()) if "human" in df.columns else []
    default_humans = ["HumanAvg"] if "HumanAvg" in humans_available else humans_available[:1]
    humans = st.sidebar.multiselect(
        "Human reference",
        humans_available,
        default=default_humans,
        key=f"{page_key}_humans",
    )

    metrics_available = [m for m in METRICS if m in set(df.get("metric", pd.Series(dtype=str)).astype(str))]
    metrics = st.sidebar.multiselect(
        "Metrics",
        metrics_available or METRICS,
        default=metrics_available or METRICS,
        key=f"{page_key}_metrics",
        format_func=lambda x: METRIC_LABEL.get(x, x),
    )

    representations = None
    if allow_representations and "representation" in df.columns:
        reps_available = [r for r in REPRESENTATIONS if r in set(df["representation"].astype(str))]
        representations = st.sidebar.multiselect(
            "Representation",
            reps_available,
            default=reps_available,
            key=f"{page_key}_representations",
        )

    patch_views = None
    if allow_patch_views and "patch_view" in df.columns:
        views_available = [p for p in PATCH_VIEWS if p in set(df["patch_view"].astype(str))]
        patch_views = st.sidebar.multiselect(
            "Patch view",
            views_available,
            default=views_available,
            key=f"{page_key}_patch_views",
        )

    return {
        "datasets": datasets,
        "models": models,
        "humans": humans,
        "metrics": metrics,
        "representations": representations,
        "patch_views": patch_views,
    }


def render_alignment_page(
    df: pd.DataFrame,
    *,
    title: str,
    page_key: str,
    allow_representations: bool,
    allow_patch_views: bool,
) -> None:
    page_header(title, "Human-LLM agreement exploration.")

    if df.empty:
        st.warning("The expected table is missing or empty.")
        return

    filters = sidebar_alignment_filters(
        df,
        page_key,
        allow_representations=allow_representations,
        allow_patch_views=allow_patch_views,
    )

    filtered = filter_alignment_table(
        df,
        datasets=filters["datasets"],
        models=filters["models"],
        humans=filters["humans"],
        metrics=filters["metrics"],
        representations=filters["representations"],
        patch_views=filters["patch_views"],
    )

    alignment_summary_cards(filtered)

    fig = alignment_heatmap(filtered)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_s = alignment_bar_chart(filtered, "spearman", "Spearman by metric/model")
        if fig_s is not None:
            st.plotly_chart(fig_s, use_container_width=True)
    with c2:
        fig_m = alignment_bar_chart(filtered, "mae", "MAE by metric/model")
        if fig_m is not None:
            st.plotly_chart(fig_m, use_container_width=True)

    st.subheader("Filtered data")
    dataframe_with_download(filtered, f"{page_key}_filtered.csv")
