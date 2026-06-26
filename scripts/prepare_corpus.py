#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from desc2fix.corpus.defects4j import build_defects4j_existing, build_defects4j_raw
from desc2fix.corpus.swtbench import build_swtbench_existing, build_swtbench_raw
from desc2fix.utils.config import get_dataset_config, load_configs, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare canonical Desc2Fix corpora from raw benchmarks."
    )

    parser.add_argument(
        "--dataset",
        choices=["defects4j", "swtbench", "all"],
        default="all",
        help="Dataset to prepare.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Shortcut for --dataset all.",
    )

    parser.add_argument(
        "--mode",
        choices=["raw", "existing"],
        default="raw",
        help="Corpus build mode. raw is the official replication mode.",
    )

    parser.add_argument(
        "--defects4j-projects-root",
        default=None,
        help="Path to Defects4J framework/projects.",
    )

    parser.add_argument(
        "--defects4j-cache-dir",
        default=None,
        help="Cache directory for fetched Defects4J bug reports.",
    )

    parser.add_argument(
        "--existing-defects4j-triplets",
        default=None,
        help="Debug mode only: historical Defects4J_v3_triplets.json.",
    )

    parser.add_argument(
        "--existing-swtbench-triplets",
        default=None,
        help="Debug mode only: historical swtbench_corpus_v6_triplets.json.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Delay between external bug-report requests for Defects4J.",
    )

    return parser.parse_args()


def prepare_defects4j(args: argparse.Namespace, cfg: dict) -> None:
    dcfg = get_dataset_config(cfg["datasets"], "defects4j")

    output_path = resolve_path(dcfg["triplets_path"])
    stats_path = resolve_path(dcfg["stats_path"])

    print("[Desc2Fix] Preparing Defects4J corpus")
    print(f"  mode: {args.mode}")
    print(f"  output: {output_path}")

    if args.mode == "raw":
        projects_root = resolve_path(
            args.defects4j_projects_root
            or dcfg.get("projects_root")
            or "external/defects4j/framework/projects"
        )

        cache_dir = resolve_path(
            args.defects4j_cache_dir
            or dcfg.get("cache_dir")
            or "data/cache/defects4j_bug_reports"
        )

        triplets = build_defects4j_raw(
            projects_root=projects_root,
            output_path=output_path,
            stats_path=stats_path,
            cache_dir=cache_dir,
            projects=dcfg.get("projects"),
            sleep_seconds=args.sleep_seconds,
        )

    else:
        if not args.existing_defects4j_triplets:
            raise ValueError(
                "--mode existing requires --existing-defects4j-triplets"
            )

        triplets = build_defects4j_existing(
            existing_triplets_path=Path(args.existing_defects4j_triplets).expanduser().resolve(),
            output_path=output_path,
            stats_path=stats_path,
        )

    print(f"[OK] Defects4J triplets: {len(triplets)}")
    print(f"[OK] Wrote: {output_path}")
    print(f"[OK] Wrote: {stats_path}")


def prepare_swtbench(args: argparse.Namespace, cfg: dict) -> None:
    dcfg = get_dataset_config(cfg["datasets"], "swtbench")

    output_path = resolve_path(dcfg["triplets_path"])
    full_output_path = resolve_path(dcfg.get("full_path", "data/processed/swtbench/full.json"))
    stats_path = resolve_path(dcfg["stats_path"])

    print("[Desc2Fix] Preparing SWT-Bench corpus")
    print(f"  mode: {args.mode}")
    print(f"  output: {output_path}")

    if args.mode == "raw":
        triplets = build_swtbench_raw(
            dataset_name=dcfg.get("hf_dataset", "eth-sri/SWT-bench_bm25_27k_zsp"),
            splits=dcfg.get("splits", ["dev", "test"]),
            output_path=output_path,
            full_output_path=full_output_path,
            stats_path=stats_path,
            cache_dir=resolve_path("data/cache/huggingface"),
        )

    else:
        if not args.existing_swtbench_triplets:
            raise ValueError(
                "--mode existing requires --existing-swtbench-triplets"
            )

        triplets = build_swtbench_existing(
            existing_triplets_path=Path(args.existing_swtbench_triplets).expanduser().resolve(),
            output_path=output_path,
            full_output_path=full_output_path,
            stats_path=stats_path,
        )

    print(f"[OK] SWT-Bench triplets: {len(triplets)}")
    print(f"[OK] Wrote: {output_path}")
    print(f"[OK] Wrote: {full_output_path}")
    print(f"[OK] Wrote: {stats_path}")


def main() -> None:
    args = parse_args()

    if args.all:
        args.dataset = "all"

    cfg = load_configs()

    if args.dataset in {"defects4j", "all"}:
        prepare_defects4j(args, cfg)

    if args.dataset in {"swtbench", "all"}:
        prepare_swtbench(args, cfg)


if __name__ == "__main__":
    main()
