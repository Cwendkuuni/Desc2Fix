#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from desc2fix.extraction.schema import SIGNAL_FIELDS, normalize_signals
from desc2fix.extraction.io import signal_output_path
from desc2fix.llm.registry import ModelRegistry
from desc2fix.utils.config import get_dataset_config, load_configs, resolve_path
from desc2fix.utils.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate structured bug-signal extraction outputs."
    )

    parser.add_argument(
        "--dataset",
        choices=["defects4j", "swtbench", "all"],
        default="all",
    )

    parser.add_argument(
        "--model",
        default="all",
        help="Model key from config/models.yaml, or all.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Shortcut for --dataset all --model all.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    return parser.parse_args()


def get_structured_root(cfg: Dict) -> Path:
    paths_cfg = cfg.get("paths", {}) or {}
    data_cfg = paths_cfg.get("data", {}) or {}

    if isinstance(data_cfg, dict) and data_cfg.get("structured"):
        return resolve_path(data_cfg["structured"])

    return resolve_path("data/structured")


def load_triplets(dataset: str, cfg: Dict) -> List[Dict]:
    dcfg = get_dataset_config(cfg["datasets"], dataset)
    path = resolve_path(dcfg["triplets_path"])

    rows = read_json(path)
    rows = rows.get("rows", rows) if isinstance(rows, dict) else rows

    if not isinstance(rows, list):
        raise ValueError(f"Expected list in {path}")

    return rows


def validate_pair(dataset: str, model: str, cfg: Dict, limit: int | None = None) -> Dict:
    triplets = load_triplets(dataset, cfg)
    if limit:
        triplets = triplets[:limit]

    structured_root = get_structured_root(cfg)

    missing_files = []
    invalid_files = []
    valid_files = 0
    non_empty_counts = {field: 0 for field in SIGNAL_FIELDS}

    for t in triplets:
        path = signal_output_path(
            structured_root=structured_root,
            dataset=dataset,
            model=model,
            triplet=t,
        )

        if not path.exists():
            missing_files.append(
                {
                    "instance_id": t.get("instance_id"),
                    "path": str(path),
                }
            )
            continue

        try:
            obj = read_json(path)
            signals = normalize_signals(obj.get("signals", {}))

            missing_keys = [f for f in SIGNAL_FIELDS if f not in signals]
            if missing_keys:
                invalid_files.append(
                    {
                        "instance_id": t.get("instance_id"),
                        "path": str(path),
                        "error": f"missing keys: {missing_keys}",
                    }
                )
                continue

            for field in SIGNAL_FIELDS:
                if signals.get(field):
                    non_empty_counts[field] += 1

            valid_files += 1

        except Exception as exc:
            invalid_files.append(
                {
                    "instance_id": t.get("instance_id"),
                    "path": str(path),
                    "error": str(exc),
                }
            )

    summary = {
        "dataset": dataset,
        "model": model,
        "expected_instances": len(triplets),
        "valid_files": valid_files,
        "missing_files": len(missing_files),
        "invalid_files": len(invalid_files),
        "non_empty_counts": non_empty_counts,
        "missing_examples": missing_files[:20],
        "invalid_examples": invalid_files[:20],
    }

    out_dir = resolve_path(f"results/experiments/extraction_validation/{dataset}/{model}")
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "missing_files.json", missing_files)
    write_json(out_dir / "invalid_files.json", invalid_files)

    return summary


def main() -> None:
    args = parse_args()

    if args.all:
        args.dataset = "all"
        args.model = "all"

    cfg = load_configs()

    datasets = ["defects4j", "swtbench"] if args.dataset == "all" else [args.dataset]

    registry = ModelRegistry(cfg["models"])
    models = registry.enabled_keys() if args.model == "all" else [args.model]

    for dataset in datasets:
        for model in models:
            summary = validate_pair(dataset, model, cfg, args.limit)

            print(
                f"[OK] {dataset}/{model}: "
                f"valid={summary['valid_files']}, "
                f"missing={summary['missing_files']}, "
                f"invalid={summary['invalid_files']}"
            )
            print(f"     non_empty={summary['non_empty_counts']}")


if __name__ == "__main__":
    main()
