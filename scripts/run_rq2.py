#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from desc2fix.extraction.io import signal_output_path
from desc2fix.llm.registry import ModelRegistry
from desc2fix.rq2_report_test.metric_alignment import compute_report_test_metrics
from desc2fix.utils.config import get_dataset_config, load_configs, load_env, resolve_path
from desc2fix.utils.io import read_json, write_csv_rows, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RQ2 metric alignment: bug report ↔ test."
    )

    parser.add_argument(
        "--dataset",
        choices=["defects4j", "swtbench", "all"],
        default="all",
    )

    parser.add_argument(
        "--model",
        default="all",
        help="Model key for structured scenario, or all.",
    )

    parser.add_argument(
        "--scenario",
        choices=["raw", "structured", "both"],
        default="both",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Shortcut for --dataset all --model all --scenario both.",
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

    parser.add_argument(
        "--require-structured",
        action="store_true",
        help="Fail if structured signals are missing in structured scenario.",
    )

    return parser.parse_args()


def get_structured_root(cfg: Dict) -> Path:
    paths_cfg = cfg.get("paths", {}) or {}
    data_cfg = paths_cfg.get("data", {}) or {}

    if isinstance(data_cfg, dict) and data_cfg.get("structured"):
        return resolve_path(data_cfg["structured"])

    return resolve_path("data/structured")


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


def load_signals_for_triplet(
    *,
    dataset: str,
    model: str,
    triplet: Dict,
    cfg: Dict,
    require: bool = False,
) -> Optional[Dict]:
    structured_root = get_structured_root(cfg)

    path = signal_output_path(
        structured_root=structured_root,
        dataset=dataset,
        model=model,
        triplet=triplet,
    )

    if not path.exists():
        if require:
            raise FileNotFoundError(f"Missing structured signal file: {path}")
        return None

    obj = read_json(path)
    return obj.get("signals", {})


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
        "with_report_text": sum(1 for r in rows if r.get("has_report_text")),
        "with_test_text": sum(1 for r in rows if r.get("has_test_text")),
        "avg_rouge_l": avg("rouge_l"),
        "avg_jaccard": avg("jaccard"),
        "avg_identifier_jaccard": avg("identifier_jaccard"),
        "avg_pos_overlap": avg("pos_overlap"),
    }

    structured_keys = [
        "structured_macro_jaccard",
        "structured_all_jaccard",
    ]

    for key in structured_keys:
        if any(key in r for r in rows):
            summary[f"avg_{key}"] = avg(key)

    embedding_keys = ["sbert", "codebert", "openai_embedding"]
    for key in embedding_keys:
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
        "avg_structured_macro_jaccard",
        "avg_structured_all_jaccard",
    ]

    parts = []
    for key in keys:
        if key in summary:
            parts.append(f"{key}={summary[key]}")

    return ", ".join(parts)

def run_raw_scenario(
    *,
    dataset: str,
    triplets: List[Dict],
    cfg: Dict,
    args: argparse.Namespace,
) -> None:
    rows = []

    print(f"[Desc2Fix][RQ2] dataset={dataset}, scenario=raw")

    for triplet in tqdm(triplets, desc=f"rq2/{dataset}/raw", ncols=100):
        row = compute_report_test_metrics(
            triplet=triplet,
            scenario="raw",
            signals=None,
            full=args.full,
            include_openai_embeddings=args.include_openai_embeddings,
        )
        row["model"] = "none"
        rows.append(row)

    out_dir = resolve_path(f"results/experiments/rq2/{dataset}/raw")
    write_csv_rows(out_dir / "per_instance.csv", rows)

    summary = summarize_rows(rows)
    summary.update(
        {
            "dataset": dataset,
            "scenario": "raw",
            "model": "none",
            "artifact_pair": "report_test",
        }
    )

    write_json(out_dir / "summary.json", summary)

    # print(
    #     f"[OK][RQ2] {dataset}/raw: "
    #     f"instances={summary['instances']}, "
    #     f"avg_rouge_l={summary['avg_rouge_l']}, "
    #     f"avg_jaccard={summary['avg_jaccard']}"
    # )

    print(
        f"[OK][RQ2] {dataset}/raw: "
        f"instances={summary['instances']}, "
        f"{format_summary_metrics(summary)}"
    )


def run_structured_scenario(
    *,
    dataset: str,
    model: str,
    triplets: List[Dict],
    cfg: Dict,
    args: argparse.Namespace,
) -> None:
    rows = []

    print(f"[Desc2Fix][RQ2] dataset={dataset}, scenario=structured, model={model}")

    for triplet in tqdm(triplets, desc=f"rq2/{dataset}/structured/{model}", ncols=100):
        signals = load_signals_for_triplet(
            dataset=dataset,
            model=model,
            triplet=triplet,
            cfg=cfg,
            require=args.require_structured,
        )

        if signals is None:
            continue

        row = compute_report_test_metrics(
            triplet=triplet,
            scenario="structured",
            signals=signals,
            full=args.full,
            include_openai_embeddings=args.include_openai_embeddings,
        )
        row["model"] = model
        rows.append(row)

    out_dir = resolve_path(f"results/experiments/rq2/{dataset}/structured/{model}")
    write_csv_rows(out_dir / "per_instance.csv", rows)

    summary = summarize_rows(rows)
    summary.update(
        {
            "dataset": dataset,
            "scenario": "structured",
            "model": model,
            "artifact_pair": "report_test",
        }
    )

    write_json(out_dir / "summary.json", summary)

    # print(
    #     f"[OK][RQ2] {dataset}/structured/{model}: "
    #     f"instances={summary['instances']}, "
    #     f"avg_rouge_l={summary['avg_rouge_l']}, "
    #     f"avg_jaccard={summary['avg_jaccard']}"
    # )

    print(
        f"[OK][RQ2] {dataset}/structured/{model}: "
        f"instances={summary['instances']}, "
        f"{format_summary_metrics(summary)}"
    )


def run_dataset(dataset: str, models: List[str], cfg: Dict, args: argparse.Namespace) -> None:
    triplets = load_triplets(dataset, cfg, args.limit)

    scenarios = (
        ["raw", "structured"]
        if args.scenario == "both"
        else [args.scenario]
    )

    for scenario in scenarios:
        if scenario == "raw":
            run_raw_scenario(
                dataset=dataset,
                triplets=triplets,
                cfg=cfg,
                args=args,
            )

        elif scenario == "structured":
            for model in models:
                run_structured_scenario(
                    dataset=dataset,
                    model=model,
                    triplets=triplets,
                    cfg=cfg,
                    args=args,
                )


def main() -> None:
    args = parse_args()

    if args.all:
        args.dataset = "all"
        args.model = "all"
        args.scenario = "both"

    if not args.fast and not args.full:
        args.fast = True

    load_env()
    cfg = load_configs()

    datasets = ["defects4j", "swtbench"] if args.dataset == "all" else [args.dataset]

    registry = ModelRegistry(cfg["models"])
    models = registry.enabled_keys() if args.model == "all" else [args.model]

    for dataset in datasets:
        run_dataset(dataset, models, cfg, args)


if __name__ == "__main__":
    main()
