#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pandas as pd
import streamlit as st

from components import dataframe_with_download, page_header
from data_loader import list_processed_artifacts, resolve_path


page_header("Corpus", "Processed data, structured extraction files, and generated result directories.")

artifact_df = list_processed_artifacts()
dataframe_with_download(artifact_df, "corpus_artifact_status.csv", height=320)

st.subheader("Directory explorer")

default_dirs = [
    "data/processed",
    "data/structured",
    "results/experiments",
    "results/tables/human_evaluation",
    "results/figures/human_evaluation_plotly",
]

selected = st.selectbox("Directory", default_dirs)
root = resolve_path(selected)

if not root.exists():
    st.warning(f"Directory not found: `{root}`")
    st.stop()

rows = []
for p in sorted(root.rglob("*")):
    if p.is_file():
        rows.append(
            {
                "relative_path": str(p.relative_to(root)),
                "suffix": p.suffix,
                "size_mb": round(p.stat().st_size / (1024 * 1024), 3),
            }
        )

files_df = pd.DataFrame(rows)
st.caption(f"Root: `{root}`")
dataframe_with_download(files_df, f"{selected.replace('/', '_')}_files.csv", height=520)
