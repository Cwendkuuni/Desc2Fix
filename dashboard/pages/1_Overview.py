#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st

from components import (
    dataframe_with_download,
    global_metric_cards,
    page_header,
    row_count_chart,
)
from data_loader import load_global_summary


page_header("Overview", "Global consolidated human-evaluation table.")

global_df = load_global_summary()

if global_df.empty:
    st.warning("Missing `results/tables/human_evaluation/global_human_evaluation_summary.csv`.")
    st.stop()

global_metric_cards(global_df)

fig = row_count_chart(global_df)
if fig is not None:
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Rows by RQ")
counts = global_df.groupby(["rq", "rq_label"], dropna=False).size().reset_index(name="rows")
dataframe_with_download(counts, "rows_by_rq.csv", height=220)

st.subheader("Global summary table")
dataframe_with_download(global_df, "global_human_evaluation_summary.csv")
