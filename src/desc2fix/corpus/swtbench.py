#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import html
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from desc2fix.corpus.schema import compute_corpus_stats, make_triplet, validate_triplet
from desc2fix.utils.io import read_json, write_json


DEFAULT_HF_DATASET = "eth-sri/SWT-bench_bm25_27k_zsp"


def clean_text(text: Any) -> str:
    if not text:
        return ""

    text = html.unescape(str(text))
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"!\[[^\]]*\]\([^\)]+\)", "", text)
    text = re.sub(r"\[[^\]]*\]\([^\)]+\)", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def safe_json_loads(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value

    if not isinstance(value, str):
        return []

    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


def extract_test_blocks(test_patch: str) -> tuple[str, str]:
    if not test_patch:
        return "", ""

    added_lines = []

    for line in str(test_patch).splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:].rstrip())

    code = "\n".join(added_lines).strip()

    if not code:
        return "", ""

    python_test = ""
    declarative_test = ""

    if re.search(r"def test_|assert|pytest\.|unittest\.|self\.assert", code):
        python_test = code

    if re.search(r"(pass_str|fail_str|test_pass_|test_fail_|: \|)", code):
        declarative_test = code

    if not python_test and not declarative_test and "import " in code:
        python_test = code

    return python_test.strip(), declarative_test.strip()


def extract_patch_variants(patch: str) -> Dict[str, Any]:
    if not patch:
        return {
            "add": "",
            "remove": "",
            "full": "",
            "add_lines": 0,
            "remove_lines": 0,
        }

    patch = str(patch).replace("<patch>", "").replace("</patch>", "").strip()

    added = []
    removed = []

    for line in patch.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].rstrip())
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].rstrip())

    code_add = "\n".join(added).strip()
    code_remove = "\n".join(removed).strip()

    if code_add or code_remove:
        code_full = (
            "# --- REMOVED CODE ---\n"
            + code_remove
            + "\n\n# +++ ADDED CODE +++\n"
            + code_add
        ).strip()
    else:
        code_full = ""

    return {
        "add": code_add,
        "remove": code_remove,
        "full": code_full,
        "add_lines": len(added),
        "remove_lines": len(removed),
    }


def record_to_triplet(ex: Dict[str, Any]) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    bug_report = clean_text(ex.get("problem_statement", ""))
    patch_raw = str(ex.get("patch", "") or "").replace("<patch>", "").replace("</patch>", "").strip()
    test_patch = str(ex.get("test_patch", "") or "").strip()

    python_test, yaml_test = extract_test_blocks(test_patch)
    patch_variants = extract_patch_variants(patch_raw)

    record = {
        "instance_id": ex.get("instance_id"),
        "repo": ex.get("repo"),
        "created_at": ex.get("created_at"),
        "bug_report": bug_report,
        "patch": patch_raw,
        "patch_code_add": patch_variants["add"],
        "patch_code_remove": patch_variants["remove"],
        "patch_code_full": patch_variants["full"],
        "test_code": python_test,
        "test_yaml": yaml_test,
        "fail_to_pass": safe_json_loads(ex.get("FAIL_TO_PASS", "[]")),
        "pass_to_pass": safe_json_loads(ex.get("PASS_TO_PASS", "[]")),
        "base_commit": ex.get("base_commit"),
        "environment_commit": ex.get("environment_setup_commit"),
        "version": ex.get("version"),
        "patch_add_lines": patch_variants["add_lines"],
        "patch_remove_lines": patch_variants["remove_lines"],
    }

    if not bug_report or not patch_raw:
        return record, None

    if not python_test or not patch_variants["full"]:
        return record, None

    triplet = make_triplet(
        dataset="swtbench",
        instance_id=str(ex.get("instance_id") or ""),
        project=None,
        repo=ex.get("repo"),
        language="python",
        bug_report={
            "summary": "",
            "description": bug_report,
            "comments": [],
            "raw_text": bug_report,
            "source": "github_issue",
        },
        tests={
            "trigger": safe_json_loads(ex.get("FAIL_TO_PASS", "[]")),
            "relevant": safe_json_loads(ex.get("PASS_TO_PASS", "[]")),
            "test_code": python_test,
            "test_yaml": yaml_test,
        },
        patch={
            "src": {
                "raw": patch_variants["full"],
                "add": patch_variants["add"],
                "remove": patch_variants["remove"],
                "files": [],
            },
            "test": None,
            "full": patch_variants["full"],
            "add": patch_variants["add"],
            "remove": patch_variants["remove"],
        },
        metadata={
            "repo": ex.get("repo"),
            "created_at": ex.get("created_at"),
            "base_commit": ex.get("base_commit"),
            "environment_commit": ex.get("environment_setup_commit"),
            "version": ex.get("version"),
        },
    )

    validate_triplet(triplet)
    return record, triplet


def compute_swtbench_full_stats(records: List[Dict[str, Any]], triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "total_instances": 0,
            "triplets": 0,
            "repos": 0,
            "repo_distribution": {},
            "test_type_counts": {"python": 0, "declarative": 0, "none": 0},
        }

    repo_counter = Counter([r.get("repo") for r in records if r.get("repo")])

    test_type_counts = {"python": 0, "declarative": 0, "none": 0}
    patch_sizes = []

    for record in records:
        if record.get("test_code"):
            test_type_counts["python"] += 1
        elif record.get("test_yaml"):
            test_type_counts["declarative"] += 1
        else:
            test_type_counts["none"] += 1

        patch_sizes.append(
            int(record.get("patch_add_lines", 0) or 0)
            + int(record.get("patch_remove_lines", 0) or 0)
        )

    bug_lengths = [
        len(str(r.get("bug_report", "")).split())
        for r in records
        if r.get("bug_report")
    ]

    patch_lines = [
        len(str(r.get("patch", "")).splitlines())
        for r in records
        if r.get("patch")
    ]

    test_lines = [
        len(str(r.get("test_code", "")).splitlines())
        for r in records
        if r.get("test_code")
    ]

    ratios = []
    for record in records:
        add = int(record.get("patch_add_lines", 0) or 0)
        remove = int(record.get("patch_remove_lines", 0) or 0)
        if add or remove:
            ratios.append(add / remove if remove > 0 else 1.0)

    return {
        "total_instances": len(records),
        "triplets": len(triplets),
        "repos": len(repo_counter),
        "repo_distribution": dict(repo_counter.most_common(15)),
        "avg_bug_report_words": round(mean(bug_lengths), 2) if bug_lengths else 0.0,
        "avg_patch_lines": round(mean(patch_lines), 2) if patch_lines else 0.0,
        "avg_test_lines": round(mean(test_lines), 2) if test_lines else 0.0,
        "avg_patch_size_lines": round(mean(patch_sizes), 2) if patch_sizes else 0.0,
        "avg_add_remove_ratio": round(mean(ratios), 2) if ratios else 0.0,
        "test_type_counts": test_type_counts,
    }


def build_swtbench_raw(
    *,
    dataset_name: str,
    splits: List[str],
    output_path: Path,
    full_output_path: Path,
    stats_path: Path,
    cache_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError as exc:
        raise ImportError(
            "Missing dependency 'datasets'. Install it with: python -m pip install datasets"
        ) from exc

    datasets = []

    for split in splits:
        print(f"[Desc2Fix] Loading SWT-Bench split: {split}")
        kwargs = {
            "path": dataset_name,
            "split": split,
            "verification_mode": "no_checks",
        }
        if cache_dir:
            kwargs["cache_dir"] = str(cache_dir)

        ds = load_dataset(**kwargs)
        datasets.append(ds)

    combined = concatenate_datasets(datasets)
    print(f"[Desc2Fix] SWT-Bench combined rows: {len(combined)}")

    records: List[Dict[str, Any]] = []
    triplets: List[Dict[str, Any]] = []

    for ex in tqdm(combined, desc="SWT-Bench", ncols=100):
        record, triplet = record_to_triplet(dict(ex))
        if record:
            records.append(record)
        if triplet:
            triplets.append(triplet)

    write_json(full_output_path, records)
    write_json(output_path, triplets)

    full_stats = compute_swtbench_full_stats(records, triplets)
    canonical_stats = compute_corpus_stats(triplets)

    stats = {
        **full_stats,
        "canonical_triplet_stats": canonical_stats,
    }

    write_json(stats_path, stats)

    return triplets


def build_swtbench_existing(
    *,
    existing_triplets_path: Path,
    output_path: Path,
    full_output_path: Path,
    stats_path: Path,
) -> List[Dict[str, Any]]:
    rows = read_json(existing_triplets_path)
    rows = rows.get("rows", rows) if isinstance(rows, dict) else rows

    records = []
    triplets = []

    for item in rows:
        record = {
            "instance_id": item.get("instance_id"),
            "repo": item.get("repo"),
            "bug_report": item.get("bug_report"),
            "test_code": item.get("test_code"),
            "test_yaml": item.get("test_yaml"),
            "patch_code_add": item.get("patch_code_add"),
            "patch_code_remove": item.get("patch_code_remove"),
            "patch_code_full": item.get("patch_code_full"),
        }
        records.append(record)

        triplet = make_triplet(
            dataset="swtbench",
            instance_id=str(item.get("instance_id") or ""),
            project=None,
            repo=item.get("repo"),
            language="python",
            bug_report={
                "summary": "",
                "description": clean_text(item.get("bug_report", "")),
                "comments": [],
                "raw_text": clean_text(item.get("bug_report", "")),
                "source": "github_issue",
            },
            tests={
                "trigger": item.get("fail_to_pass", []),
                "relevant": item.get("pass_to_pass", []),
                "test_code": item.get("test_code", ""),
                "test_yaml": item.get("test_yaml", ""),
            },
            patch={
                "src": {
                    "raw": item.get("patch_code_full", "") or "",
                    "add": item.get("patch_code_add", "") or "",
                    "remove": item.get("patch_code_remove", "") or "",
                    "files": [],
                },
                "test": None,
                "full": item.get("patch_code_full", "") or "",
                "add": item.get("patch_code_add", "") or "",
                "remove": item.get("patch_code_remove", "") or "",
            },
            metadata={
                "repo": item.get("repo"),
                "source_file": str(existing_triplets_path),
            },
        )

        validate_triplet(triplet)
        triplets.append(triplet)

    write_json(full_output_path, records)
    write_json(output_path, triplets)
    write_json(stats_path, compute_corpus_stats(triplets))

    return triplets
