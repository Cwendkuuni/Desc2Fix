#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

from tqdm import tqdm

from desc2fix.rq3_report_patch.metric_alignment import PATCH_VIEWS
from desc2fix.rq4_test_patch_internal.metric_alignment import compute_test_patch_metrics
from desc2fix.utils.config import get_dataset_config, load_configs, load_env, resolve_path
from desc2fix.utils.io import read_json, write_csv_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RQ4 internal metric alignment: test ↔ patch."
    )

    parser.add_argument(
        "--dataset",
        choices=["defects4j", "swtbench", "all"],
        default="all",
    )

    parser.add_argument(
        "--patch-view",
        choices=["full", "add", "remove", "all"],
        default="all",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Shortcut for --dataset all --patch-view all.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use only lightweight metrics.",
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help="Use lightweight metrics + embeddings.",
    )

    parser.add_argument(
        "--include-openai-embeddings",
        action="store_true",
        help="Include OpenAI text-embedding-3-small in --full mode.",
    )

    return parser.parse_args()


def load_triplets(dataset: str, cfg: Dict, limit: Optional[int]) -> List[Dict]:
    dcfg = get_dataset_config(cfg["datasets"], dataset)
    path = resolve_path(dcfg["triplets_path"])

    rows = read_json(path)
    rows = rows.get("rows", rows) if isinstance(rows, dict) else rows

    if not isinstance(rows, list):
        raise ValueError(f"Expected list of triplets in {path}")

    if limit:
        rows = rows[:limit]

    return rows


def summarize_rows(rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "instances": 0,
            "avg_rouge_l": 0.0,
            "avg_jaccard": 0.0,
            "avg_identifier_jaccard": 0.0,
            "avg_pos_overlap": 0.0,
        }

    def avg(key: str) -> float:
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    summary = {
        "instances": len(rows),
        "with_test_text": sum(1 for r in rows if r.get("has_test_text")),
        "with_patch_text": sum(1 for r in rows if r.get("has_patch_text")),
        "avg_rouge_l": avg("rouge_l"),
        "avg_jaccard": avg("jaccard"),
        "avg_identifier_jaccard": avg("identifier_jaccard"),
        "avg_pos_overlap": avg("pos_overlap"),
    }

    for key in ["sbert", "codebert", "openai_embedding"]:
        if any(key in r for r in rows):
            summary[f"avg_{key}"] = avg(key)

    return summary


def format_summary_metrics(summary: Dict) -> str:
    keys = [
        "avg_rouge_l",
        "avg_jaccard",
        "avg_identifier_jaccard",
        "avg_pos_overlap",
        "avg_sbert",
        "avg_codebert",
        "avg_openai_embedding",
    ]

    return ", ".join(
        f"{key}={summary[key]}"
        for key in keys
        if key in summary
    )


def run_dataset_patch_view(
    *,
    dataset: str,
    patch_view: str,
    triplets: List[Dict],
    args: argparse.Namespace,
) -> None:
    rows = []

    print(f"[Desc2Fix][RQ4-Internal] dataset={dataset}, patch_view={patch_view}")

    for triplet in tqdm(
        triplets,
        desc=f"rq4_internal/{dataset}/{patch_view}",
        ncols=100,
    ):
        row = compute_test_patch_metrics(
            triplet=triplet,
            patch_view=patch_view,
            full=args.full,
            include_openai_embeddings=args.include_openai_embeddings,
        )
        rows.append(row)

    out_dir = resolve_path(f"results/experiments/rq4_internal/{dataset}/{patch_view}")
    write_csv_rows(out_dir / "per_instance.csv", rows)

    summary = summarize_rows(rows)
    summary.update(
        {
            "dataset": dataset,
            "scenario": "raw",
            "artifact_pair": "test_patch",
            "patch_view": patch_view,
            "internal": True,
            "paper_rq": False,
        }
    )

    write_json(out_dir / "summary.json", summary)

    print(
        f"[OK][RQ4-Internal] {dataset}/{patch_view}: "
        f"instances={summary['instances']}, "
        f"{format_summary_metrics(summary)}"
    )


def run_dataset(dataset: str, cfg: Dict, args: argparse.Namespace) -> None:
    triplets = load_triplets(dataset, cfg, args.limit)
    patch_views = PATCH_VIEWS if args.patch_view == "all" else [args.patch_view]

    for patch_view in patch_views:
        run_dataset_patch_view(
            dataset=dataset,
            patch_view=patch_view,
            triplets=triplets,
            args=args,
        )


def main() -> None:
    args = parse_args()

    if args.all:
        args.dataset = "all"
        args.patch_view = "all"

    if not args.fast and not args.full:
        args.fast = True

    load_env()
    cfg = load_configs()

    datasets = ["defects4j", "swtbench"] if args.dataset == "all" else [args.dataset]

    for dataset in datasets:
        run_dataset(dataset, cfg, args)


if __name__ == "__main__":
    main()
