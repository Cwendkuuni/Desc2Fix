#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from desc2fix.utils.config import resolve_path
from desc2fix.utils.io import ensure_dir


RQ_LABELS = {
    "rq2": "RQ2_Report_Test",
    "rq3": "RQ3_Report_Patch",
    "rq4_internal": "RQ4_Test_Patch_Internal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate consolidated human evaluation summary tables."
    )

    parser.add_argument(
        "--human-root",
        default="Human_Eval",
        help="Root containing Human_Eval/RQ1,RQ2,RQ3,RQ4.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/tables/human_evaluation",
        help="Output directory for consolidated tables.",
    )

    parser.add_argument(
        "--rq1-dir",
        default="results/experiments/rq1_human_extraction",
        help="Directory containing RQ1 human extraction agreement outputs.",
    )

    parser.add_argument(
        "--agreement-dir",
        default="results/experiments/human_agreement",
        help="Directory where run_human_agreement.py writes outputs.",
    )

    return parser.parse_args()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[OK] {path}: rows={len(df)}")


def compact_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].round(6)

    return out


def summarize_debug_file(path: Path, label: str) -> pd.DataFrame:
    df = read_csv_if_exists(path)

    if df.empty:
        return pd.DataFrame(
            [
                {
                    "source": label,
                    "status": "empty",
                    "rows": 0,
                }
            ]
        )

    group_cols = [c for c in ["rq", "dataset", "representation", "patch_view"] if c in df.columns]

    if group_cols:
        out = df.groupby(group_cols, dropna=False).size().reset_index(name="rows")
        out.insert(0, "source", label)
        out.insert(1, "status", "non_empty")
        return out

    return pd.DataFrame(
        [
            {
                "source": label,
                "status": "non_empty",
                "rows": len(df),
            }
        ]
    )


def summarize_rq1(rq1_dir: Path, output_dir: Path) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}

    human_pairwise = read_csv_if_exists(rq1_dir / "human_pairwise_summary.csv")
    llm_vs_human = read_csv_if_exists(rq1_dir / "llm_vs_human_summary.csv")
    llm_pairwise = read_csv_if_exists(rq1_dir / "llm_pairwise_summary.csv")

    if not human_pairwise.empty:
        df = human_pairwise.copy()
        df.insert(0, "rq", "rq1")
        df.insert(1, "comparison_type", "human_pairwise")
        outputs["rq1_human_pairwise_summary"] = compact_numeric_columns(df)
        write_csv(output_dir / "rq1_human_pairwise_summary.csv", outputs["rq1_human_pairwise_summary"])

    if not llm_vs_human.empty:
        df = llm_vs_human.copy()
        df.insert(0, "rq", "rq1")
        df.insert(1, "comparison_type", "llm_vs_human")
        outputs["rq1_llm_vs_human_summary"] = compact_numeric_columns(df)
        write_csv(output_dir / "rq1_llm_vs_human_summary.csv", outputs["rq1_llm_vs_human_summary"])

    if not llm_pairwise.empty:
        df = llm_pairwise.copy()
        df.insert(0, "rq", "rq1")
        df.insert(1, "comparison_type", "llm_pairwise")
        outputs["rq1_llm_pairwise_summary"] = compact_numeric_columns(df)
        write_csv(output_dir / "rq1_llm_pairwise_summary.csv", outputs["rq1_llm_pairwise_summary"])

    debug_parts = [
        summarize_debug_file(rq1_dir / "debug_human_without_llm.csv", "rq1_human_without_llm"),
        summarize_debug_file(rq1_dir / "debug_llm_without_human.csv", "rq1_llm_without_human"),
    ]

    debug = pd.concat(debug_parts, ignore_index=True, sort=False)
    outputs["rq1_debug_summary"] = debug
    write_csv(output_dir / "rq1_debug_summary.csv", debug)

    return outputs


def run_human_agreement_for_rq(rq: str, human_root: Path) -> None:
    """
    Regenerate the standard human agreement outputs for a given RQ.

    This keeps the summarizer self-contained: it refreshes
    results/experiments/human_agreement before copying the outputs.
    """
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/run_human_agreement.py",
        "--rq",
        rq,
        "--human-root",
        str(human_root),
    ]

    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def summarize_alignment_rq(
    *,
    rq: str,
    label: str,
    human_root: Path,
    agreement_dir: Path,
    output_dir: Path,
) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}

    run_human_agreement_for_rq(rq, human_root)

    human_flat = read_csv_if_exists(agreement_dir / "human_flat.csv")
    human_agg = read_csv_if_exists(agreement_dir / "human_agg.csv")
    inter_human = read_csv_if_exists(agreement_dir / "inter_human_agreement.csv")
    human_llm = read_csv_if_exists(agreement_dir / "human_llm_agreement.csv")

    if not human_flat.empty:
        group_cols = ["rq", "dataset", "representation", "patch_view", "rater"]
        existing = [c for c in group_cols if c in human_flat.columns]

        df = human_flat.groupby(existing, dropna=False).size().reset_index(name="rows")
        df.insert(0, "rq_label", label)
        outputs[f"{rq}_human_flat_counts"] = df
        write_csv(output_dir / f"{label}_human_flat_counts.csv", df)

    if not human_agg.empty:
        group_cols = ["rq", "dataset", "representation", "patch_view"]
        existing = [c for c in group_cols if c in human_agg.columns]

        df = human_agg.groupby(existing, dropna=False).size().reset_index(name="n_pairs")
        df.insert(0, "rq_label", label)
        outputs[f"{rq}_human_pair_counts"] = df
        write_csv(output_dir / f"{label}_human_pair_counts.csv", df)

    if not inter_human.empty:
        df = inter_human.copy()
        df.insert(0, "rq_label", label)
        df = compact_numeric_columns(df)
        outputs[f"{rq}_inter_human_agreement"] = df
        write_csv(output_dir / f"{label}_inter_human_agreement.csv", df)

    if not human_llm.empty:
        df = human_llm.copy()
        df.insert(0, "rq_label", label)
        df = compact_numeric_columns(df)
        outputs[f"{rq}_human_llm_agreement"] = df
        write_csv(output_dir / f"{label}_human_llm_agreement.csv", df)

        macro = df.copy()
        outputs[f"{rq}_human_llm_agreement_macro"] = macro
        write_csv(output_dir / f"{label}_human_llm_agreement_macro.csv", macro)

    debug_parts = [
        summarize_debug_file(agreement_dir / "debug_human_without_llm.csv", f"{label}_human_without_llm"),
        summarize_debug_file(agreement_dir / "debug_llm_without_human.csv", f"{label}_llm_without_human"),
    ]

    debug = pd.concat(debug_parts, ignore_index=True, sort=False)
    outputs[f"{rq}_debug_summary"] = debug
    write_csv(output_dir / f"{label}_debug_summary.csv", debug)

    return outputs


def build_global_summary(output_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []

    # RQ1
    rq1_llm_vs_human = read_csv_if_exists(output_dir / "rq1_llm_vs_human_summary.csv")
    if not rq1_llm_vs_human.empty:
        macro = rq1_llm_vs_human[rq1_llm_vs_human["level"] == "macro"].copy()
        for _, r in macro.iterrows():
            rows.append(
                {
                    "rq": "rq1",
                    "rq_label": "RQ1_Extraction",
                    "dataset": r.get("dataset"),
                    "representation": "STRUCTURED_EXTRACTION",
                    "patch_view": "none",
                    "model": r.get("left_source"),
                    "human": r.get("right_source"),
                    "metric": "macro_signal_similarity",
                    "n": r.get("n"),
                    "primary_value": r.get("similarity_mean"),
                    "secondary_value": r.get("presence_agreement_mean"),
                    "primary_value_name": "similarity_mean",
                    "secondary_value_name": "presence_agreement_mean",
                }
            )

    # RQ2/RQ3/RQ4 alignment
    for label, rq in [
        ("RQ2_Report_Test", "rq2"),
        ("RQ3_Report_Patch", "rq3"),
        ("RQ4_Test_Patch_Internal", "rq4_internal"),
    ]:
        path = output_dir / f"{label}_human_llm_agreement.csv"
        df = read_csv_if_exists(path)

        if df.empty:
            continue

        for _, r in df.iterrows():
            rows.append(
                {
                    "rq": rq,
                    "rq_label": label,
                    "dataset": r.get("dataset"),
                    "representation": r.get("representation"),
                    "patch_view": r.get("patch_view"),
                    "model": r.get("model"),
                    "human": r.get("human"),
                    "metric": r.get("metric"),
                    "n": r.get("n"),
                    "primary_value": r.get("spearman"),
                    "secondary_value": r.get("mae"),
                    "primary_value_name": "spearman",
                    "secondary_value_name": "mae",
                }
            )

    return compact_numeric_columns(pd.DataFrame(rows))


def main() -> None:
    args = parse_args()

    human_root = resolve_path(args.human_root)
    rq1_dir = resolve_path(args.rq1_dir)
    agreement_dir = resolve_path(args.agreement_dir)
    output_dir = resolve_path(args.output_dir)

    ensure_dir(output_dir)

    print(f"[Desc2Fix][Tables] human_root={human_root}")
    print(f"[Desc2Fix][Tables] rq1_dir={rq1_dir}")
    print(f"[Desc2Fix][Tables] agreement_dir={agreement_dir}")
    print(f"[Desc2Fix][Tables] output_dir={output_dir}")

    summarize_rq1(rq1_dir, output_dir)

    for rq, label in RQ_LABELS.items():
        summarize_alignment_rq(
            rq=rq,
            label=label,
            human_root=human_root,
            agreement_dir=agreement_dir,
            output_dir=output_dir,
        )

    global_summary = build_global_summary(output_dir)
    write_csv(output_dir / "global_human_evaluation_summary.csv", global_summary)

    print("[DONE] Human evaluation summary tables generated.")


if __name__ == "__main__":
    main()
