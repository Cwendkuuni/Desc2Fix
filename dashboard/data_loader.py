#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Centralized data loading utilities for the Desc2Fix Streamlit dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st


DEFAULT_TABLES_DIR = Path("results/tables/human_evaluation")
DEFAULT_FIGURES_DIR = Path("results/figures/human_evaluation_plotly")
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_STRUCTURED_DIR = Path("data/structured")
DEFAULT_LLM_JUDGES_DIR = Path("results/experiments/llm_judges")


def repo_root() -> Path:
    """
    Resolve the repository root when Streamlit is launched from either:
      - repository root: streamlit run dashboard/app.py
      - dashboard dir:  streamlit run app.py
    """
    cwd = Path.cwd().resolve()
    if (cwd / "dashboard").exists() and (cwd / "results").exists():
        return cwd
    if cwd.name == "dashboard" and (cwd.parent / "results").exists():
        return cwd.parent
    return cwd


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return repo_root() / path


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def read_jsonl_cached(path: str, max_rows: int | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    rows = []
    with p.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(pd.read_json(line, typ="series").to_dict())
            except ValueError:
                continue

    return pd.DataFrame(rows)


def load_table(filename: str, tables_dir: str | Path = DEFAULT_TABLES_DIR) -> pd.DataFrame:
    return read_csv_cached(str(resolve_path(tables_dir) / filename))


def load_global_summary(tables_dir: str | Path = DEFAULT_TABLES_DIR) -> pd.DataFrame:
    return load_table("global_human_evaluation_summary.csv", tables_dir)


def load_rq1_tables(tables_dir: str | Path = DEFAULT_TABLES_DIR) -> Dict[str, pd.DataFrame]:
    return {
        "llm_vs_human": load_table("rq1_llm_vs_human_summary.csv", tables_dir),
        "human_pairwise": load_table("rq1_human_pairwise_summary.csv", tables_dir),
        "llm_pairwise": load_table("rq1_llm_pairwise_summary.csv", tables_dir),
        "debug": load_table("rq1_debug_summary.csv", tables_dir),
    }


def load_rq2_table(tables_dir: str | Path = DEFAULT_TABLES_DIR) -> pd.DataFrame:
    return load_table("RQ2_Report_Test_human_llm_agreement.csv", tables_dir)


def load_rq3_table(tables_dir: str | Path = DEFAULT_TABLES_DIR) -> pd.DataFrame:
    return load_table("RQ3_Report_Patch_human_llm_agreement.csv", tables_dir)


def load_rq4_table(tables_dir: str | Path = DEFAULT_TABLES_DIR) -> pd.DataFrame:
    return load_table("RQ4_Test_Patch_Internal_human_llm_agreement.csv", tables_dir)


def load_debug_tables(tables_dir: str | Path = DEFAULT_TABLES_DIR) -> Dict[str, pd.DataFrame]:
    return {
        "RQ1 Extraction": load_table("rq1_debug_summary.csv", tables_dir),
        "RQ2 Report-Test": load_table("RQ2_Report_Test_debug_summary.csv", tables_dir),
        "RQ3 Report-Patch": load_table("RQ3_Report_Patch_debug_summary.csv", tables_dir),
        "RQ4 Test-Patch": load_table("RQ4_Test_Patch_Internal_debug_summary.csv", tables_dir),
    }


def list_figure_files(figures_dir: str | Path = DEFAULT_FIGURES_DIR) -> pd.DataFrame:
    root = resolve_path(figures_dir)
    if not root.exists():
        return pd.DataFrame(columns=["name", "type", "path", "size_bytes"])

    rows = []
    for p in sorted(root.glob("*")):
        if p.is_file():
            rows.append(
                {
                    "name": p.name,
                    "type": p.suffix.replace(".", ""),
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                }
            )
    return pd.DataFrame(rows)


def list_processed_artifacts() -> pd.DataFrame:
    roots = [
        ("processed", resolve_path(DEFAULT_PROCESSED_DIR)),
        ("structured", resolve_path(DEFAULT_STRUCTURED_DIR)),
        ("llm_judges", resolve_path(DEFAULT_LLM_JUDGES_DIR)),
        ("figures", resolve_path(DEFAULT_FIGURES_DIR)),
        ("tables", resolve_path(DEFAULT_TABLES_DIR)),
    ]

    rows = []
    for kind, root in roots:
        if not root.exists():
            rows.append(
                {
                    "kind": kind,
                    "path": str(root),
                    "exists": False,
                    "files": 0,
                    "size_mb": 0.0,
                }
            )
            continue

        files = [p for p in root.rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files)
        rows.append(
            {
                "kind": kind,
                "path": str(root),
                "exists": True,
                "files": len(files),
                "size_mb": round(size / (1024 * 1024), 2),
            }
        )

    return pd.DataFrame(rows)


def concatenate_debug_tables(tables_dir: str | Path = DEFAULT_TABLES_DIR) -> pd.DataFrame:
    parts = []
    for rq_label, df in load_debug_tables(tables_dir).items():
        if df.empty:
            continue
        tmp = df.copy()
        tmp["rq_label"] = rq_label
        parts.append(tmp)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, sort=False)


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
    return out
