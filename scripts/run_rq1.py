#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from desc2fix.extraction.io import signal_output_path
from desc2fix.extraction.schema import SIGNAL_FIELDS, normalize_signals
from desc2fix.llm.registry import ModelRegistry
from desc2fix.rq1_extraction.completeness import (
    completeness_ratio,
    signal_count,
    summarize_model_outputs,
    summarize_pairwise_consistency,
)
from desc2fix.utils.config import get_dataset_config, load_configs, resolve_path
from desc2fix.utils.io import read_json, write_csv_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RQ1 analysis: structured bug signal extraction."
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
        help="Limit instances for smoke testing.",
    )

    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail if some signal files are missing.",
    )

    return parser.parse_args()


def get_structured_root(cfg: Dict) -> Path:
    paths_cfg = cfg.get("paths", {}) or {}
    data_cfg = paths_cfg.get("data", {}) or {}

    if isinstance(data_cfg, dict) and data_cfg.get("structured"):
        return resolve_path(data_cfg["structured"])

    return resolve_path("data/structured")


def load_triplets(dataset: str, cfg: Dict, limit: int | None = None) -> List[Dict]:
    dcfg = get_dataset_config(cfg["datasets"], dataset)
    path = resolve_path(dcfg["triplets_path"])

    rows = read_json(path)
    rows = rows.get("rows", rows) if isinstance(rows, dict) else rows

    if not isinstance(rows, list):
        raise ValueError(f"Expected list of triplets in {path}")

    if limit:
        rows = rows[:limit]

    return rows


def load_signal_rows(
    *,
    dataset: str,
    model: str,
    triplets: List[Dict],
    cfg: Dict,
    require_complete: bool = False,
) -> List[Dict]:
    structured_root = get_structured_root(cfg)
    rows = []
    missing = []

    for t in triplets:
        out_path = signal_output_path(
            structured_root=structured_root,
            dataset=dataset,
            model=model,
            triplet=t,
        )

        base = {
            "dataset": dataset,
            "model": model,
            "instance_id": t.get("instance_id"),
            "project": t.get("project"),
            "repo": t.get("repo"),
            "path": str(out_path),
        }

        if not out_path.exists():
            missing.append(base)
            rows.append(
                {
                    **base,
                    "available": False,
                    "signals": {},
                    "signal_count": 0,
                    "field_completeness": 0.0,
                    **{f"{field}_count": 0 for field in SIGNAL_FIELDS},
                    **{f"{field}_non_empty": False for field in SIGNAL_FIELDS},
                }
            )
            continue

        obj = read_json(out_path)
        signals = normalize_signals(obj.get("signals", {}))

        row = {
            **base,
            "available": True,
            "signals": signals,
            "signal_count": signal_count(signals),
            "field_completeness": round(completeness_ratio(signals), 6),
        }

        for field in SIGNAL_FIELDS:
            row[f"{field}_count"] = len(signals.get(field, []))
            row[f"{field}_non_empty"] = bool(signals.get(field))

        rows.append(row)

    if missing and require_complete:
        examples = "\n".join(m["path"] for m in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} signal files for {dataset}/{model}. "
            f"First examples:\n{examples}"
        )

    return rows


def csv_safe_rows(rows: List[Dict]) -> List[Dict]:
    """
    Remove nested dict/list fields for CSV writing.
    """
    clean = []

    for r in rows:
        out = {}
        for k, v in r.items():
            if k == "signals":
                continue
            if isinstance(v, (dict, list)):
                out[k] = str(v)
            else:
                out[k] = v
        clean.append(out)

    return clean


def run_dataset(dataset: str, models: List[str], cfg: Dict, args: argparse.Namespace) -> None:
    triplets = load_triplets(dataset, cfg, args.limit)

    out_dir = resolve_path(f"results/experiments/rq1/{dataset}")
    all_summaries = {}
    model_rows = {}

    print(f"[Desc2Fix][RQ1] dataset={dataset}, instances={len(triplets)}")

    for model in models:
        rows = load_signal_rows(
            dataset=dataset,
            model=model,
            triplets=triplets,
            cfg=cfg,
            require_complete=args.require_complete,
        )

        model_rows[model] = rows

        summary = summarize_model_outputs(
            dataset=dataset,
            model=model,
            rows=rows,
        )

        all_summaries[model] = summary

        write_csv_rows(out_dir / f"per_instance_{model}.csv", csv_safe_rows(rows))
        write_json(out_dir / f"summary_{model}.json", summary)

        print(
            f"[OK][RQ1] {dataset}/{model}: "
            f"available={summary['available_instances']}/"
            f"{summary['expected_instances']}, "
            f"avg_completeness={summary['avg_field_completeness']}"
        )

    consistency_outputs = {}

    if len(models) >= 2:
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                a = models[i]
                b = models[j]

                comp = summarize_pairwise_consistency(
                    dataset=dataset,
                    model_a=a,
                    model_b=b,
                    rows_a=model_rows[a],
                    rows_b=model_rows[b],
                )

                key = f"{a}_vs_{b}"
                consistency_outputs[key] = comp["summary"]

                write_csv_rows(
                    out_dir / f"consistency_{key}.csv",
                    comp["per_instance"],
                )
                write_json(
                    out_dir / f"consistency_{key}.json",
                    comp["summary"],
                )

                print(
                    f"[OK][RQ1] {dataset}/{key}: "
                    f"common={comp['summary']['common_instances']}, "
                    f"avg_macro_jaccard={comp['summary']['avg_macro_jaccard']}"
                )

    global_summary = {
        "dataset": dataset,
        "instances": len(triplets),
        "models": all_summaries,
        "consistency": consistency_outputs,
    }

    write_json(out_dir / "summary.json", global_summary)


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
        run_dataset(dataset, models, cfg, args)


if __name__ == "__main__":
    main()
