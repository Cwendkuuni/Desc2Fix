#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import streamlit as st

from components import dataframe_with_download, debug_chart, page_header
from data_loader import (
    DEFAULT_FIGURES_DIR,
    concatenate_debug_tables,
    list_figure_files,
    resolve_path,
)


page_header("Artifact Browser", "Generated figures, debug summaries, and downloadable artifacts.")

st.subheader("Generated Plotly figures")

figures = list_figure_files()
if figures.empty:
    st.warning("No generated figures found. Run `python scripts/generate_plotly_figures.py`.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("HTML", int((figures["type"] == "html").sum()))
    c2.metric("PNG", int((figures["type"] == "png").sum()))
    c3.metric("PDF", int((figures["type"] == "pdf").sum()))

    dataframe_with_download(figures, "generated_figure_files.csv", height=360)

    pngs = figures[figures["type"] == "png"].copy()
    if not pngs.empty:
        st.subheader("PNG preview")
        selected = st.selectbox("Preview figure", pngs["name"].tolist())
        selected_path = Path(pngs[pngs["name"] == selected]["path"].iloc[0])
        st.image(str(selected_path), use_container_width=True)

st.subheader("Debug summaries")
debug_df = concatenate_debug_tables()

fig = debug_chart(debug_df)
if fig is not None:
    st.plotly_chart(fig, use_container_width=True)

dataframe_with_download(debug_df, "debug_summaries_all.csv", height=360)
