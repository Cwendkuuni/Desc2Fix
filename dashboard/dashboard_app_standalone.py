#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Desc2Fix Human Evaluation Dashboard

Interactive Streamlit dashboard for the Desc2Fix human-evaluation package.

Inputs:
  results/tables/human_evaluation/
  results/figures/human_evaluation_plotly/

Run:
  streamlit run dashboard/app.py

Optional:
  streamlit run dashboard/app.py -- --tables-dir results/tables/human_evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit(
        "Streamlit is not installed. Install it with:\n\n"
        "  python -m pip install streamlit\n\n"
        "Then run:\n\n"
        "  streamlit run dashboard/app.py\n"
    ) from exc


# ---------------------------------------------------------------------
# Canonical orders and labels
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Desc2Fix human-evaluation dashboard.")

    parser.add_argument(
        "--tables-dir",
        default="results/tables/human_evaluation",
        help="Directory containing consolidated human-evaluation tables.",
    )

    parser.add_argument(
        "--figures-dir",
        default="results/figures/human_evaluation_plotly",
        help="Directory containing generated Plotly figures.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_csv_cached(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def read_table(tables_dir: Path, filename: str) -> pd.DataFrame:
    return read_csv_cached(str(tables_dir / filename))


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
    return out


def safe_model_label(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.map(MODEL_LABEL).fillna(s)


def safe_metric_label(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    return s.map(METRIC_LABEL).fillna(s)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def apply_style(fig: go.Figure, height: int = 560) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        font=dict(size=15),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=55, b=70, l=70, r=35),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=13),
        ),
    )
    fig.update_xaxes(showline=True, linecolor="black", ticks="outside", showgrid=False)
    fig.update_yaxes(showline=True, linecolor="black", ticks="outside", gridcolor="rgba(0,0,0,0.12)")
    return fig


def show_dataframe_download(df: pd.DataFrame, filename: str, label: str = "Download CSV") -> None:
    if df.empty:
        return

    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def metric_cards(global_df: pd.DataFrame) -> None:
    total_rows = len(global_df)
    rq_count = global_df["rq"].nunique() if "rq" in global_df.columns else 0
    datasets = global_df["dataset"].nunique() if "dataset" in global_df.columns else 0
    models = global_df["model"].nunique() if "model" in global_df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Global rows", f"{total_rows:,}")
    c2.metric("RQs", rq_count)
    c3.metric("Datasets", datasets)
    c4.metric("Models", models)


def row_count_chart(global_df: pd.DataFrame) -> go.Figure:
    counts = (
        global_df.groupby(["rq", "rq_label"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
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


def rq1_chart(rq1_df: pd.DataFrame, selected_datasets: list[str], selected_models: list[str]) -> Optional[go.Figure]:
    if rq1_df.empty:
        return None

    df = rq1_df[
        (rq1_df["level"] == "macro")
        & (rq1_df["right_source"] == "HumanAvg")
    ].copy()

    if selected_datasets:
        df = df[df["dataset"].isin(selected_datasets)]

    if selected_models:
        df = df[df["left_source"].isin(selected_models)]

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


def alignment_table_filter(
    df: pd.DataFrame,
    *,
    datasets: list[str],
    models: list[str],
    representations: Optional[list[str]] = None,
    patch_views: Optional[list[str]] = None,
    metrics: Optional[list[str]] = None,
    humans: Optional[list[str]] = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    out = clean_strings(df)

    if datasets and "dataset" in out.columns:
        out = out[out["dataset"].isin(datasets)]
    if models and "model" in out.columns:
        out = out[out["model"].isin(models)]
    if representations and "representation" in out.columns:
        out = out[out["representation"].isin(representations)]
    if patch_views and "patch_view" in out.columns:
        out = out[out["patch_view"].isin(patch_views)]
    if metrics and "metric" in out.columns:
        out = out[out["metric"].isin(metrics)]
    if humans and "human" in out.columns:
        out = out[out["human"].isin(humans)]

    return out


def alignment_bar_chart(df: pd.DataFrame, value: str, title: str) -> Optional[go.Figure]:
    if df.empty or value not in df.columns:
        return None

    plot_df = df.copy()
    plot_df["model_label"] = safe_model_label(plot_df["model"])
    plot_df["metric_label"] = safe_metric_label(plot_df["metric"])

    group_cols = ["dataset", "model_label", "metric_label"]
    if "patch_view" in plot_df.columns and plot_df["patch_view"].nunique() > 1:
        group_cols.append("patch_view")
    if "representation" in plot_df.columns and plot_df["representation"].nunique() > 1:
        group_cols.append("representation")

    summary = (
        plot_df.groupby(group_cols, dropna=False)[value]
        .mean()
        .reset_index()
    )

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
        color_discrete_sequence=COLOR_SEQUENCE,
        category_orders={
            "metric_label": [METRIC_LABEL[m] for m in METRICS],
            "model_label": [MODEL_LABEL[m] for m in MODELS],
            "representation": REPRESENTATIONS,
            "patch_view": PATCH_VIEWS,
        },
        title=title,
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(title="", tickangle=-20)
    fig.update_yaxes(title=value)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return apply_style(fig, height=700 if facet_row else 560)


def spearman_heatmap(df: pd.DataFrame, title: str) -> Optional[go.Figure]:
    if df.empty:
        return None

    plot_df = df.copy()
    plot_df["row"] = plot_df["dataset"].astype(str) + " / " + safe_model_label(plot_df["model"])
    plot_df["metric_label"] = safe_metric_label(plot_df["metric"])

    if "representation" in plot_df.columns and plot_df["representation"].nunique() > 1:
        col_key = "representation"
    elif "patch_view" in plot_df.columns and plot_df["patch_view"].nunique() > 1:
        col_key = "patch_view"
    else:
        col_key = "metric_label"

    row_order = [f"{ds} / {MODEL_LABEL[m]}" for ds in DATASETS for m in MODELS]
    col_order = REPRESENTATIONS if col_key == "representation" else PATCH_VIEWS if col_key == "patch_view" else [METRIC_LABEL[m] for m in METRICS]

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

    group_cols = ["debug_type"]
    if "rq" in df.columns:
        group_cols.insert(0, "rq")
    elif "source" in df.columns:
        group_cols.insert(0, "source")

    summary = df.groupby(group_cols, dropna=False)["rows"].sum().reset_index()

    x_col = "rq" if "rq" in summary.columns else "source"
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


# ---------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------

def page_overview(global_df: pd.DataFrame, tables_dir: Path) -> None:
    st.header("Overview")

    if global_df.empty:
        st.error("Missing global_human_evaluation_summary.csv")
        return

    metric_cards(global_df)
    st.plotly_chart(row_count_chart(global_df), use_container_width=True)

    st.subheader("Global summary")
    st.dataframe(global_df, use_container_width=True, height=420)
    show_dataframe_download(global_df, "global_human_evaluation_summary.csv")


def page_rq1(rq1_df: pd.DataFrame, rq1_human_pairwise: pd.DataFrame, rq1_llm_pairwise: pd.DataFrame) -> None:
    st.header("RQ1 — Structured bug signal extraction")

    if rq1_df.empty:
        st.warning("RQ1 summary table not found.")
        return

    with st.sidebar:
        st.subheader("RQ1 filters")
        selected_datasets = st.multiselect("RQ1 datasets", DATASETS, default=DATASETS)
        selected_models = st.multiselect("RQ1 models", MODELS, default=MODELS, format_func=lambda x: MODEL_LABEL.get(x, x))

    fig = rq1_chart(rq1_df, selected_datasets, selected_models)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("LLM vs human summary")
    st.dataframe(rq1_df, use_container_width=True, height=360)
    show_dataframe_download(rq1_df, "rq1_llm_vs_human_summary.csv")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Human pairwise")
        st.dataframe(rq1_human_pairwise, use_container_width=True, height=280)
    with c2:
        st.subheader("LLM pairwise")
        st.dataframe(rq1_llm_pairwise, use_container_width=True, height=280)


def page_alignment(
    title: str,
    df: pd.DataFrame,
    *,
    allow_representations: bool,
    allow_patch_views: bool,
    default_human: str = "HumanAvg",
) -> None:
    st.header(title)

    if df.empty:
        st.warning("Table not found.")
        return

    df = clean_strings(df)

    with st.sidebar:
        st.subheader(f"{title} filters")

        datasets = st.multiselect(
            f"{title} datasets",
            DATASETS,
            default=DATASETS,
            key=f"{title}_datasets",
        )
        models = st.multiselect(
            f"{title} models",
            MODELS,
            default=MODELS,
            key=f"{title}_models",
            format_func=lambda x: MODEL_LABEL.get(x, x),
        )
        humans_available = sorted(df["human"].dropna().unique().tolist()) if "human" in df.columns else []
        humans = st.multiselect(
            f"{title} human reference",
            humans_available,
            default=[default_human] if default_human in humans_available else humans_available[:1],
            key=f"{title}_humans",
        )
        metrics = st.multiselect(
            f"{title} metrics",
            METRICS,
            default=METRICS,
            key=f"{title}_metrics",
            format_func=lambda x: METRIC_LABEL.get(x, x),
        )

        representations = None
        if allow_representations and "representation" in df.columns:
            reps_available = [r for r in REPRESENTATIONS if r in set(df["representation"].astype(str))]
            representations = st.multiselect(
                f"{title} representation",
                reps_available,
                default=reps_available,
                key=f"{title}_representations",
            )

        patch_views = None
        if allow_patch_views and "patch_view" in df.columns:
            p_available = [p for p in PATCH_VIEWS if p in set(df["patch_view"].astype(str))]
            patch_views = st.multiselect(
                f"{title} patch view",
                p_available,
                default=p_available,
                key=f"{title}_patch_views",
            )

    filtered = alignment_table_filter(
        df,
        datasets=datasets,
        models=models,
        representations=representations,
        patch_views=patch_views,
        metrics=metrics,
        humans=humans,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(filtered))
    c2.metric("Mean Spearman", f"{filtered['spearman'].mean():.3f}" if "spearman" in filtered and len(filtered) else "NA")
    c3.metric("Mean MAE", f"{filtered['mae'].mean():.3f}" if "mae" in filtered and len(filtered) else "NA")

    fig_h = spearman_heatmap(filtered, "")
    if fig_h is not None:
        st.plotly_chart(fig_h, use_container_width=True)

    fig_s = alignment_bar_chart(filtered, "spearman", "Spearman by metric/model")
    if fig_s is not None:
        st.plotly_chart(fig_s, use_container_width=True)

    fig_m = alignment_bar_chart(filtered, "mae", "MAE by metric/model")
    if fig_m is not None:
        st.plotly_chart(fig_m, use_container_width=True)

    st.subheader("Filtered table")
    st.dataframe(filtered, use_container_width=True, height=420)
    show_dataframe_download(filtered, f"{title.lower().replace(' ', '_').replace('—', '').replace('/', '_')}_filtered.csv")


def page_debug(tables_dir: Path) -> None:
    st.header("Debug summaries")

    files = [
        "rq1_debug_summary.csv",
        "RQ2_Report_Test_debug_summary.csv",
        "RQ3_Report_Patch_debug_summary.csv",
        "RQ4_Test_Patch_Internal_debug_summary.csv",
    ]

    parts = []
    for f in files:
        df = read_table(tables_dir, f)
        if not df.empty:
            df = df.copy()
            df["file"] = f
            parts.append(df)

    if not parts:
        st.warning("No debug summary files found.")
        return

    all_debug = pd.concat(parts, ignore_index=True, sort=False)

    fig = debug_chart(all_debug)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(all_debug, use_container_width=True, height=460)
    show_dataframe_download(all_debug, "debug_summaries_all.csv")


def page_figures(figures_dir: Path) -> None:
    st.header("Generated Plotly figures")

    if not figures_dir.exists():
        st.warning(f"Figures directory not found: {figures_dir}")
        return

    html_files = sorted(figures_dir.glob("*.html"))
    png_files = sorted(figures_dir.glob("*.png"))
    pdf_files = sorted(figures_dir.glob("*.pdf"))

    c1, c2, c3 = st.columns(3)
    c1.metric("HTML figures", len(html_files))
    c2.metric("PNG figures", len(png_files))
    c3.metric("PDF figures", len(pdf_files))

    st.subheader("Available files")
    file_df = pd.DataFrame(
        [
            {"name": p.name, "type": p.suffix.replace(".", ""), "path": str(p)}
            for p in sorted(figures_dir.glob("*"))
            if p.is_file()
        ]
    )
    st.dataframe(file_df, use_container_width=True, height=420)

    st.info(
        "Open the HTML files directly in a browser for full interactivity. "
        "Use the PDF files for LaTeX and the PNG files for README/slides."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)

    st.set_page_config(
        page_title="Desc2Fix Human Evaluation",
        page_icon="📊",
        layout="wide",
    )

    st.title("Desc2Fix Human Evaluation Dashboard")
    st.caption("Interactive exploration of RQ1/RQ2/RQ3/RQ4 human-evaluation outputs.")

    global_df = read_table(tables_dir, "global_human_evaluation_summary.csv")
    rq1 = read_table(tables_dir, "rq1_llm_vs_human_summary.csv")
    rq1_human_pairwise = read_table(tables_dir, "rq1_human_pairwise_summary.csv")
    rq1_llm_pairwise = read_table(tables_dir, "rq1_llm_pairwise_summary.csv")
    rq2 = read_table(tables_dir, "RQ2_Report_Test_human_llm_agreement.csv")
    rq3 = read_table(tables_dir, "RQ3_Report_Patch_human_llm_agreement.csv")
    rq4 = read_table(tables_dir, "RQ4_Test_Patch_Internal_human_llm_agreement.csv")

    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Page",
            [
                "Overview",
                "RQ1 Extraction",
                "RQ2 Report-Test",
                "RQ3 Report-Patch",
                "RQ4 Test-Patch",
                "Debug",
                "Figures",
            ],
        )

        st.divider()
        st.caption(f"Tables: `{tables_dir}`")
        st.caption(f"Figures: `{figures_dir}`")

    if page == "Overview":
        page_overview(global_df, tables_dir)
    elif page == "RQ1 Extraction":
        page_rq1(rq1, rq1_human_pairwise, rq1_llm_pairwise)
    elif page == "RQ2 Report-Test":
        page_alignment(
            "RQ2 Report-Test",
            rq2,
            allow_representations=True,
            allow_patch_views=False,
        )
    elif page == "RQ3 Report-Patch":
        page_alignment(
            "RQ3 Report-Patch",
            rq3,
            allow_representations=True,
            allow_patch_views=True,
        )
    elif page == "RQ4 Test-Patch":
        page_alignment(
            "RQ4 Test-Patch",
            rq4,
            allow_representations=True,
            allow_patch_views=True,
        )
    elif page == "Debug":
        page_debug(tables_dir)
    elif page == "Figures":
        page_figures(figures_dir)


if __name__ == "__main__":
    main()
