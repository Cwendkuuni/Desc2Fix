#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set

from tqdm import tqdm

from desc2fix.extraction.io import signal_output_path, write_signal_file
from desc2fix.extraction.signal_extractor import extract_signals_for_triplet
from desc2fix.llm.clients import make_llm_client
from desc2fix.llm.registry import ModelRegistry
from desc2fix.utils.config import get_dataset_config, load_configs, load_env, resolve_path
from desc2fix.utils.io import ensure_dir, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured bug signals with configured LLMs."
    )

    parser.add_argument(
        "--dataset",
        choices=["defects4j", "swtbench", "all"],
        default="all",
    )

    parser.add_argument(
        "--model",
        default="all",
        help="Model key from config/models.yaml, e.g., gpt4o, deepseek, or all.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Shortcut for --dataset all --model all.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip instances whose output already exists.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke tests. Applied after --sample-from-human filtering.",
    )

    parser.add_argument(
        "--prompt",
        default="prompts/extraction/structured_bug_signal_extraction.txt",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--sample-from-human",
        default=None,
        help=(
            "Optional Human_Eval root. If provided, extract signals only for "
            "instances present in human annotations."
        ),
    )

    return parser.parse_args()


def load_triplets_for_dataset(dataset: str, cfg: Dict) -> List[Dict]:
    dcfg = get_dataset_config(cfg["datasets"], dataset)
    triplets_path = resolve_path(dcfg["triplets_path"])

    if not triplets_path.exists():
        raise FileNotFoundError(
            f"Missing triplets for {dataset}: {triplets_path}. "
            f"Run scripts/prepare_corpus.py first."
        )

    rows = read_json(triplets_path)
    rows = rows.get("rows", rows) if isinstance(rows, dict) else rows

    if not isinstance(rows, list):
        raise ValueError(f"Expected list of triplets in {triplets_path}")

    return rows


def get_structured_root(cfg: Dict) -> Path:
    """
    Resolve structured output root robustly.

    Expected config:
      cfg["paths"]["data"]["structured"]

    Fallback:
      data/structured
    """
    paths_cfg = cfg.get("paths", {}) or {}

    data_cfg = paths_cfg.get("data", {}) or {}
    if isinstance(data_cfg, dict) and data_cfg.get("structured"):
        return resolve_path(data_cfg["structured"])

    if paths_cfg.get("structured"):
        return resolve_path(paths_cfg["structured"])

    return resolve_path("data/structured")


def human_dataset_name(dataset: str) -> str:
    if dataset == "defects4j":
        return "Defects4J"
    if dataset == "swtbench":
        return "SWTBench"
    return dataset


def normalize_human_dataset(value: str) -> str:
    s = str(value or "").strip().lower()

    if "defects" in s or s == "d4j":
        return "Defects4J"

    if "swt" in s or "swe" in s:
        return "SWTBench"

    return str(value or "").strip()


def triplet_pair_id(dataset: str, triplet: Dict) -> str:
    if dataset == "defects4j":
        project = triplet.get("project")
        bug_id = triplet.get("bug_id")

        if not bug_id:
            instance = str(triplet.get("instance_id") or "")
            if "-" in instance:
                project, bug_id = instance.split("-", 1)

        return f"{project}__{bug_id}"

    return str(triplet.get("instance_id"))


def human_pair_id_from_record(obj: Dict) -> str:
    dataset = normalize_human_dataset(obj.get("dataset"))

    if dataset == "Defects4J":
        return f"{obj.get('project')}__{obj.get('bug_id')}"

    return str(obj.get("instance_id"))


def read_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def human_target_pair_ids_for_dataset(
    *,
    human_root: Path,
    dataset: str,
) -> Set[str]:
    """
    For extraction, we need the union of all instances that have human
    annotations in RQ2/RQ3/RQ4. Extracted signals are reusable across RQs.
    """
    target: Set[str] = set()
    dataset_h = human_dataset_name(dataset)

    for rq_dir_name in ["RQ2", "RQ3", "RQ4"]:
        rq_dir = human_root / rq_dir_name
        if not rq_dir.exists():
            continue

        for path in sorted(rq_dir.glob("*human*_alignment_*.jsonl")):
            for obj in read_jsonl_records(path):
                if normalize_human_dataset(obj.get("dataset")) != dataset_h:
                    continue

                pid = human_pair_id_from_record(obj)
                if pid and pid != "None__None":
                    target.add(pid)

    return target


def filter_triplets_from_human_sample(
    *,
    triplets: List[Dict],
    human_root: Optional[str],
    dataset: str,
) -> List[Dict]:
    if not human_root:
        return triplets

    root = resolve_path(human_root)

    targets = human_target_pair_ids_for_dataset(
        human_root=root,
        dataset=dataset,
    )

    if not targets:
        print(f"[WARN] No human targets found for dataset={dataset} in {root}")
        return []

    filtered = [
        t for t in triplets
        if triplet_pair_id(dataset, t) in targets
    ]

    matched = {triplet_pair_id(dataset, t) for t in filtered}
    missing = sorted(targets - matched)

    print(
        f"[Desc2Fix][Extract] human sample filter: "
        f"dataset={dataset}, targets={len(targets)}, matched_triplets={len(filtered)}"
    )

    if missing:
        debug_dir = resolve_path("results/experiments/extraction/debug")
        ensure_dir(debug_dir)
        p = debug_dir / f"missing_human_targets_{dataset}.txt"
        p.write_text("\n".join(missing), encoding="utf-8")
        print(f"[WARN] missing human targets written to {p}: {len(missing)}")

    return filtered


def run_extraction_for_pair(
    *,
    dataset: str,
    model_key: str,
    args: argparse.Namespace,
    cfg: Dict,
) -> None:
    registry = ModelRegistry(cfg["models"])
    spec = registry.get(model_key)

    if not spec.enabled:
        print(f"[SKIP] Model disabled in config: {model_key}")
        return

    client = make_llm_client(spec, retry=True, max_retries=args.max_retries)

    triplets = load_triplets_for_dataset(dataset, cfg)

    triplets = filter_triplets_from_human_sample(
        triplets=triplets,
        human_root=args.sample_from_human,
        dataset=dataset,
    )

    if args.limit:
        triplets = triplets[: args.limit]

    structured_root = get_structured_root(cfg)

    errors = []
    written = 0
    skipped = 0

    print(f"[Desc2Fix] Extracting signals: dataset={dataset}, model={model_key}")
    print(f"[Desc2Fix] Instances: {len(triplets)}")

    for triplet in tqdm(triplets, desc=f"{dataset}/{model_key}", ncols=100):
        out_path = signal_output_path(
            structured_root=structured_root,
            dataset=dataset,
            model=model_key,
            triplet=triplet,
        )

        if args.resume and out_path.exists():
            skipped += 1
            continue

        try:
            payload = extract_signals_for_triplet(
                triplet=triplet,
                client=client,
                prompt_path=args.prompt,
            )
            write_signal_file(out_path, payload)
            written += 1

        except Exception as exc:
            err = {
                "dataset": dataset,
                "model": model_key,
                "instance_id": triplet.get("instance_id"),
                "project": triplet.get("project"),
                "repo": triplet.get("repo"),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            print(f"[ERR] {dataset}/{model_key}/{triplet.get('instance_id')}: {exc}")

    log_dir = resolve_path(f"results/experiments/extraction_logs/{dataset}/{model_key}")
    ensure_dir(log_dir)

    write_json(
        log_dir / "summary.json",
        {
            "dataset": dataset,
            "model": model_key,
            "instances": len(triplets),
            "written": written,
            "skipped": skipped,
            "errors": len(errors),
            "sample_from_human": args.sample_from_human,
        },
    )
    write_json(log_dir / "errors.json", errors)

    print(
        f"[OK] dataset={dataset}, model={model_key}, "
        f"written={written}, skipped={skipped}, errors={len(errors)}"
    )


def main() -> None:
    args = parse_args()

    if args.all:
        args.dataset = "all"
        args.model = "all"

    load_env()
    cfg = load_configs()

    datasets = ["defects4j", "swtbench"] if args.dataset == "all" else [args.dataset]

    registry = ModelRegistry(cfg["models"])
    models = registry.enabled_keys() if args.model == "all" else [args.model]

    for dataset in datasets:
        for model_key in models:
            run_extraction_for_pair(
                dataset=dataset,
                model_key=model_key,
                args=args,
                cfg=cfg,
            )


if __name__ == "__main__":
    main()
