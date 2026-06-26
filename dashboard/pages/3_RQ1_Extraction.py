#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st

from components import (
    DATASETS,
    MODELS,
    MODEL_LABEL,
    dataframe_with_download,
    page_header,
    rq1_extraction_chart,
)
from data_loader import load_rq1_tables


page_header("RQ1 Extraction", "Structured bug signal extraction agreement.")

tables = load_rq1_tables()
rq1_df = tables["llm_vs_human"]

if rq1_df.empty:
    st.warning("Missing RQ1 table: `rq1_llm_vs_human_summary.csv`.")
    st.stop()

with st.sidebar:
    st.subheader("RQ1 filters")
    datasets = st.multiselect("Datasets", DATASETS, default=DATASETS)
    models = st.multiselect(
        "Models",
        MODELS,
        default=MODELS,
        format_func=lambda x: MODEL_LABEL.get(x, x),
    )

fig = rq1_extraction_chart(rq1_df, datasets, models)
if fig is not None:
    st.plotly_chart(fig, use_container_width=True)

st.subheader("LLM vs human summary")
dataframe_with_download(rq1_df, "rq1_llm_vs_human_summary.csv", height=360)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Human pairwise summary")
    dataframe_with_download(tables["human_pairwise"], "rq1_human_pairwise_summary.csv", height=320)

with c2:
    st.subheader("LLM pairwise summary")
    dataframe_with_download(tables["llm_pairwise"], "rq1_llm_pairwise_summary.csv", height=320)

st.subheader("RQ1 debug summary")
dataframe_with_download(tables["debug"], "rq1_debug_summary.csv", height=260)
