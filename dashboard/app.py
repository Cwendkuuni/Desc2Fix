#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main landing page for the Desc2Fix multipage Streamlit dashboard.

Run from the repository root:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from components import global_metric_cards, row_count_chart, dataframe_with_download
from data_loader import (
    DEFAULT_FIGURES_DIR,
    DEFAULT_TABLES_DIR,
    load_global_summary,
    list_processed_artifacts,
    repo_root,
)


st.set_page_config(
    page_title="Desc2Fix Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Desc2Fix Dashboard")
st.caption("Human-evaluation results, Plotly figures, and artifact browser.")

root = repo_root()
tables_dir = root / DEFAULT_TABLES_DIR
figures_dir = root / DEFAULT_FIGURES_DIR

st.info(
    f"Repository root: `{root}`\n\n"
    f"Tables: `{tables_dir}`\n\n"
    f"Figures: `{figures_dir}`"
)

global_df = load_global_summary()

if global_df.empty:
    st.warning(
        "Global summary table not found. Run:\n\n"
        "`python scripts/generate_human_summary_tables.py --human-root Human_Eval`"
    )
else:
    global_metric_cards(global_df)

    fig = row_count_chart(global_df)
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Global human-evaluation summary")
    dataframe_with_download(global_df, "global_human_evaluation_summary.csv", height=360)

st.subheader("Artifact status")
artifacts = list_processed_artifacts()
dataframe_with_download(artifacts, "artifact_status.csv", height=260)

st.markdown(
    """
### Pages

Use the sidebar to navigate:

- **Overview**: global row counts and consolidated table.
- **Corpus**: processed artifacts and available data directories.
- **RQ1 Extraction**: structured signal extraction agreement.
- **RQ2 Report-Test**: human-LLM agreement for report-test alignment.
- **RQ3 Report-Patch**: human-LLM agreement for report-patch alignment.
- **RQ4 Internal Test-Patch**: internal test-patch agreement.
- **Artifact Browser**: generated figures and result files.
"""
)
