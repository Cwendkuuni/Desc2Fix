#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from desc2fix.human.schema import (
    SCORE_FIELDS,
    infer_rater_from_path,
    infer_rq_from_path,
    normalize_dataset,
    pair_id_from_record,
    parse_scenario,
)


def preferred_human_dirs(root: Path, rq: str) -> List[Path]:
    """
    Return possible source folders for human annotations.

    The replication package uses:
      Human_Eval/RQ2
      Human_Eval/RQ3
      Human_Eval/RQ4

    Older local experiments may use:
      DATA/HUMAN-RQ2
      DATA/HUMAN-RQ3
      DATA/HUMAN-RQ4
    """
    if rq == "rq2":
        names = ["RQ2", "HUMAN-RQ2"]

    elif rq == "rq3":
        names = ["RQ3", "HUMAN-RQ3"]

    elif rq == "rq4_internal":
        names = ["RQ4", "HUMAN-RQ4"]

    else:
        raise ValueError(f"Unknown RQ: {rq}")

    return [root / name for name in names if (root / name).exists()]


def find_human_files(root: Path, rq: str) -> List[Path]:
    """
    Find raw human annotation files for one RQ.

    Preference:
      1. JSONL files, because they preserve the original score structure.
      2. CSV files, only if no JSONL files exist in the selected directory.

    This avoids loading both CSV and JSONL duplicates for the same annotations.
    """
    dirs = preferred_human_dirs(root, rq)

    for d in dirs:
        jsonl_files = sorted(
            p for p in d.rglob("*human*_alignment*.jsonl")
            if p.is_file()
        )

        if jsonl_files:
            return jsonl_files

        csv_files = sorted(
            p for p in d.rglob("*human*_alignment*.csv")
            if p.is_file()
        )

        if csv_files:
            return csv_files

    return []


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def flatten_human_record(obj: Dict, *, path: Path) -> Dict:
    rq = infer_rq_from_path(str(path))
    rater = infer_rater_from_path(path.name)

    scores = obj.get("scores", {}) if isinstance(obj.get("scores"), dict) else obj

    representation, patch_view, scenario_norm = parse_scenario(obj.get("scenario"))
    dataset = normalize_dataset(obj.get("dataset"))

    row = {
        "rq": rq,
        "dataset": dataset,
        "pair_id": pair_id_from_record(obj),
        "project": obj.get("project"),
        "bug_id": obj.get("bug_id"),
        "repo": obj.get("repo"),
        "instance_id": obj.get("instance_id"),
        "scenario": scenario_norm,
        "representation": representation,
        "patch_view": patch_view,
        "rater": rater,
        "source_file": str(path),
        "justification": scores.get("justification", obj.get("justification", "")),
    }

    for field in SCORE_FIELDS:
        row[field] = scores.get(field)

    return row


def load_human_file(path: Path) -> List[Dict]:
    rows: List[Dict] = []

    if path.suffix.lower() == ".jsonl":
        for obj in read_jsonl(path):
            rows.append(flatten_human_record(obj, path=path))

    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

        for obj in df.to_dict(orient="records"):
            rows.append(flatten_human_record(obj, path=path))

    return rows


def load_human_annotations(root: Path, rqs: Optional[List[str]] = None) -> pd.DataFrame:
    if rqs is None:
        rqs = ["rq2", "rq3", "rq4_internal"]

    all_rows: List[Dict] = []

    for rq in rqs:
        files = find_human_files(root, rq)

        if not files:
            print(f"[Desc2Fix][Human] No human files found for {rq} under {root}")

        for path in files:
            all_rows.extend(load_human_file(path))

    df = pd.DataFrame(all_rows)

    if df.empty:
        return df

    for field in SCORE_FIELDS:
        df[field] = pd.to_numeric(df[field], errors="coerce")

    # RQ2 has no patch view. RQ3/RQ4 use full/add/remove.
    df["patch_view"] = df["patch_view"].where(df["patch_view"].notna(), "none")

    # Defensive de-duplication in case the same annotations were copied both
    # as CSV and JSONL, or duplicated during local file moves.
    dedup_keys = [
        "rq",
        "dataset",
        "representation",
        "patch_view",
        "pair_id",
        "rater",
    ]

    before = len(df)
    df = df.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"[Desc2Fix][Human] deduplicated human rows: {before} -> {after}")

    return df


def flatten_llm_record(obj: Dict, *, path: Path, fallback_rq: str) -> Dict:
    representation, patch_view, scenario_norm = parse_scenario(
        obj.get("scenario", obj.get("scenario_base", ""))
    )

    row = {
        "rq": obj.get("rq", fallback_rq),
        "dataset": normalize_dataset(obj.get("dataset")),
        "pair_id": pair_id_from_record(obj),
        "project": obj.get("project"),
        "bug_id": obj.get("bug_id"),
        "repo": obj.get("repo"),
        "instance_id": obj.get("instance_id"),
        "scenario": scenario_norm,
        "representation": representation,
        "patch_view": patch_view or obj.get("patch_view") or "none",
        "model": obj.get("model"),
        "model_name": obj.get("model_name"),
        "source_file": str(path),
        "justification": obj.get("justification", ""),
    }

    for field in SCORE_FIELDS:
        row[field] = obj.get(field)

    return row


def load_llm_judges(root: Path, rqs: Optional[List[str]] = None) -> pd.DataFrame:
    if rqs is None:
        rqs = ["rq2", "rq3", "rq4_internal"]

    rows: List[Dict] = []

    for rq in rqs:
        rq_dir = root / rq
        if not rq_dir.exists():
            continue

        for path in sorted(rq_dir.rglob("scores.jsonl")):
            for obj in read_jsonl(path):
                rows.append(flatten_llm_record(obj, path=path, fallback_rq=rq))

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    for field in SCORE_FIELDS:
        df[field] = pd.to_numeric(df[field], errors="coerce")

    df["patch_view"] = df["patch_view"].where(df["patch_view"].notna(), "none")

    # Critical: scores.jsonl files are append-only in our current runner.
    # If a configuration is re-run without cleaning its output file first,
    # the same pair_id/model/scenario can appear multiple times.
    # We keep the last row as the latest run result.
    dedup_keys = [
        "rq",
        "dataset",
        "representation",
        "patch_view",
        "pair_id",
        "model",
    ]

    before = len(df)
    df = df.drop_duplicates(subset=dedup_keys, keep="last").reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"[Desc2Fix][Human] deduplicated LLM rows: {before} -> {after}")

    return df
