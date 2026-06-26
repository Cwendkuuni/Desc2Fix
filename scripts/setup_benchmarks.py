#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from desc2fix.benchmarks.defects4j import setup_defects4j
from desc2fix.benchmarks.swtbench import setup_swtbench
from desc2fix.utils.config import get_dataset_config, load_configs, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Setup external benchmarks required by Desc2Fix."
    )

    parser.add_argument(
        "--dataset",
        choices=["defects4j", "swtbench", "all"],
        default="all",
        help="Benchmark to setup.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="Reserved for future use. Config files are loaded from config/*.yaml.",
    )

    parser.add_argument(
        "--defects4j-root",
        default=None,
        help="Where Defects4J should be located. Default from config/datasets.yaml.",
    )

    parser.add_argument(
        "--no-clone",
        action="store_true",
        help="Do not clone Defects4J if missing; only inspect.",
    )

    parser.add_argument(
        "--force-clone",
        action="store_true",
        help="Remove and re-clone Defects4J if it already exists.",
    )

    parser.add_argument(
        "--init-defects4j",
        action="store_true",
        help="Run Defects4J init.sh after cloning/checking.",
    )

    parser.add_argument(
        "--skip-swtbench-download",
        action="store_true",
        help="Do not download SWT-Bench now; only create/check cache dirs.",
    )

    return parser.parse_args()


def setup_one_defects4j(args: argparse.Namespace, cfg: dict) -> None:
    dcfg = get_dataset_config(cfg["datasets"], "defects4j")

    root = (
        Path(args.defects4j_root)
        if args.defects4j_root
        else resolve_path(dcfg.get("external_root", "external/defects4j"))
    )

    setup_defects4j(
        root=root,
        clone_if_missing=not args.no_clone,
        initialize=args.init_defects4j,
        force_clone=args.force_clone,
    )


def setup_one_swtbench(args: argparse.Namespace, cfg: dict) -> None:
    dcfg = get_dataset_config(cfg["datasets"], "swtbench")

    cache_dir = resolve_path("data/cache/huggingface")
    dataset_name = dcfg.get("hf_dataset", "eth-sri/SWT-bench_bm25_27k_zsp")

    setup_swtbench(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        download=not args.skip_swtbench_download,
    )


def main() -> None:
    args = parse_args()
    cfg = load_configs()

    if args.dataset in {"defects4j", "all"}:
        setup_one_defects4j(args, cfg)

    if args.dataset in {"swtbench", "all"}:
        setup_one_swtbench(args, cfg)


if __name__ == "__main__":
    main()
