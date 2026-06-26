#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from desc2fix.human.extraction_agreement import (
    deduplicate_extractions,
    llm_pairwise_agreement,
    llm_vs_human_agreement,
    load_human_extractions,
    load_llm_extractions,
    pairwise_human_agreement,
    serialize_signals_for_csv,
    summarize_agreement,
    unmatched_debug,
)
from desc2fix.utils.config import resolve_path
from desc2fix.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RQ1 human agreement for structured bug signal extraction."
    )

    parser.add_argument(
        "--human-root",
        default="Human_Eval",
        help="Root containing Human_Eval/RQ1.",
    )

    parser.add_argument(
        "--structured-root",
        default="data/structured",
        help="Root containing LLM extracted signals.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/experiments/rq1_human_extraction",
        help="Output directory.",
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["defects4j", "swtbench"],
        choices=["defects4j", "swtbench"],
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt4o", "deepseek"],
        help="LLM extraction model keys to include.",
    )

    return parser.parse_args()


def write_csv(path: Path, df) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    print(f"[OK] {path}: rows={len(df)}")


def filter_llm_to_human_sample(human_df, llm_df):
    """
    Keep only LLM extraction rows whose dataset/pair_id exists in the human RQ1 sample.

    We still write debug_llm_without_human.csv from the full LLM dataframe,
    but all agreement computations should use this filtered dataframe.
    """
    if human_df.empty or llm_df.empty:
        return llm_df.iloc[0:0].copy()

    human_keys = human_df[["dataset", "pair_id"]].drop_duplicates()

    filtered = llm_df.merge(
        human_keys,
        on=["dataset", "pair_id"],
        how="inner",
    )

    print(
        f"[Desc2Fix][RQ1-Human] filtered LLM rows for agreement: "
        f"{len(llm_df)} -> {len(filtered)}"
    )

    return filtered


def main() -> None:
    args = parse_args()

    human_root = resolve_path(args.human_root)
    structured_root = resolve_path(args.structured_root)
    output_dir = resolve_path(args.output_dir)

    ensure_dir(output_dir)

    print(f"[Desc2Fix][RQ1-Human] human_root={human_root}")
    print(f"[Desc2Fix][RQ1-Human] structured_root={structured_root}")
    print(f"[Desc2Fix][RQ1-Human] output_dir={output_dir}")
    print(f"[Desc2Fix][RQ1-Human] datasets={args.datasets}")
    print(f"[Desc2Fix][RQ1-Human] models={args.models}")

    human_df = load_human_extractions(human_root)
    human_df = deduplicate_extractions(human_df)

    llm_df = load_llm_extractions(
        structured_root=structured_root,
        datasets=args.datasets,
        models=args.models,
    )
    llm_df = deduplicate_extractions(llm_df)

    # Write complete flat files for traceability/debugging.
    write_csv(
        output_dir / "human_extractions_flat.csv",
        serialize_signals_for_csv(human_df),
    )
    write_csv(
        output_dir / "llm_extractions_flat.csv",
        serialize_signals_for_csv(llm_df),
    )

    # Debug matching before filtering.
    h_without_l, l_without_h = unmatched_debug(human_df, llm_df)
    write_csv(output_dir / "debug_human_without_llm.csv", h_without_l)
    write_csv(output_dir / "debug_llm_without_human.csv", l_without_h)

    # Critical: use only the human sample for agreement computations.
    llm_df_for_agreement = filter_llm_to_human_sample(human_df, llm_df)

    # Human1 vs Human2.
    human_pairwise = pairwise_human_agreement(human_df)
    write_csv(output_dir / "human_pairwise_field_agreement.csv", human_pairwise)
    write_csv(
        output_dir / "human_pairwise_summary.csv",
        summarize_agreement(human_pairwise),
    )

    # LLMs vs Human1/Human2/HumanAvg.
    llm_vs_human = llm_vs_human_agreement(
        human_df,
        llm_df_for_agreement,
        include_human_avg=True,
    )
    write_csv(output_dir / "llm_vs_human_field_agreement.csv", llm_vs_human)
    write_csv(
        output_dir / "llm_vs_human_summary.csv",
        summarize_agreement(llm_vs_human),
    )

    # GPT-4o vs DeepSeek, restricted to human sample only.
    llm_pairwise = llm_pairwise_agreement(llm_df_for_agreement)
    write_csv(output_dir / "llm_pairwise_field_agreement.csv", llm_pairwise)
    write_csv(
        output_dir / "llm_pairwise_summary.csv",
        summarize_agreement(llm_pairwise),
    )

    if not h_without_l.empty:
        print("[WARN] Some human RQ1 extraction samples have no matching LLM extraction.")

    if not l_without_h.empty:
        print("[WARN] Some LLM extraction outputs have no matching human RQ1 sample.")
        print(
            "[INFO] Extra LLM outputs are kept in llm_extractions_flat.csv for traceability, "
            "but they are excluded from agreement computations."
        )

    print("[DONE] RQ1 human extraction agreement complete.")


if __name__ == "__main__":
    main()
