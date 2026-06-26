#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from desc2fix.human.agreement import (
    human_average,
    human_llm_agreement,
    inter_human_agreement,
    unmatched_debug,
)
from desc2fix.human.loader import load_human_annotations, load_llm_judges
from desc2fix.utils.config import resolve_path
from desc2fix.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute human-human and human-LLM agreement."
    )

    parser.add_argument(
        "--human-root",
        default="DATA",
        help="Root containing RQ2/RQ3/RQ4 human annotation folders.",
    )

    parser.add_argument(
        "--llm-root",
        default="results/experiments/llm_judges",
        help="Root containing LLM judge outputs.",
    )

    parser.add_argument(
        "--rq",
        choices=["rq2", "rq3", "rq4_internal", "all"],
        default="all",
    )

    parser.add_argument(
        "--out-dir",
        default="results/experiments/human_agreement",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    human_root = resolve_path(args.human_root)
    if not human_root.exists() and Path("DATA").exists():
        human_root = Path("DATA")

    llm_root = resolve_path(args.llm_root)
    out_dir = resolve_path(args.out_dir)
    ensure_dir(out_dir)

    rqs = ["rq2", "rq3", "rq4_internal"] if args.rq == "all" else [args.rq]

    print(f"[Desc2Fix][Human] human_root={human_root}")
    print(f"[Desc2Fix][Human] llm_root={llm_root}")
    print(f"[Desc2Fix][Human] rqs={rqs}")

    human_df = load_human_annotations(human_root, rqs=rqs)
    llm_df = load_llm_judges(llm_root, rqs=rqs)

    human_flat_path = out_dir / "human_flat.csv"
    llm_flat_path = out_dir / "llm_flat.csv"

    human_df.to_csv(human_flat_path, index=False)
    llm_df.to_csv(llm_flat_path, index=False)

    print(f"[OK] human_flat: {human_flat_path} rows={len(human_df)}")
    print(f"[OK] llm_flat: {llm_flat_path} rows={len(llm_df)}")

    if human_df.empty:
        print("[WARN] No human annotations found.")
        return

    human_avg = human_average(human_df)
    inter = inter_human_agreement(human_df)

    human_avg_path = out_dir / "human_agg.csv"
    inter_path = out_dir / "inter_human_agreement.csv"

    human_avg.to_csv(human_avg_path, index=False)
    inter.to_csv(inter_path, index=False)

    print(f"[OK] human_agg: {human_avg_path} rows={len(human_avg)}")
    print(f"[OK] inter_human_agreement: {inter_path} rows={len(inter)}")

    if llm_df.empty:
        print("[WARN] No LLM judge outputs found.")
        return

    h_without_l, l_without_h = unmatched_debug(human_df, llm_df)

    debug_h_path = out_dir / "debug_human_without_llm.csv"
    debug_l_path = out_dir / "debug_llm_without_human.csv"

    h_without_l.to_csv(debug_h_path, index=False)
    l_without_h.to_csv(debug_l_path, index=False)

    print(f"[OK] debug_human_without_llm: {debug_h_path} rows={len(h_without_l)}")
    print(f"[OK] debug_llm_without_human: {debug_l_path} rows={len(l_without_h)}")

    human_llm = human_llm_agreement(human_df, llm_df, include_human_avg=True)
    human_llm_path = out_dir / "human_llm_agreement.csv"
    human_llm.to_csv(human_llm_path, index=False)

    print(f"[OK] human_llm_agreement: {human_llm_path} rows={len(human_llm)}")

    if human_llm.empty:
        print(
            "[WARN] No human↔LLM overlap. This is expected if LLM judges were run only "
            "on smoke instances that are not in the human annotation sample."
        )


if __name__ == "__main__":
    main()
