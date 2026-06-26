#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional


SIGNAL_FIELDS = [
    "exception",
    "api_involved",
    "input_conditions",
    "reproduction_steps",
    "expected",
    "actual",
]


def empty_patch_block() -> Dict[str, Any]:
    return {
        "raw": "",
        "add": "",
        "remove": "",
        "files": [],
    }


def make_triplet(
    *,
    dataset: str,
    instance_id: str,
    language: str,
    project: Optional[str] = None,
    repo: Optional[str] = None,
    bug_report: Optional[Dict[str, Any]] = None,
    tests: Optional[Dict[str, Any]] = None,
    patch: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    bug_report = bug_report or {}
    tests = tests or {}
    patch = patch or {}
    metadata = metadata or {}

    summary = bug_report.get("summary", "") or ""
    description = bug_report.get("description", "") or ""
    comments = bug_report.get("comments", []) or []

    if not isinstance(comments, list):
        comments = [str(comments)]

    raw_text = bug_report.get("raw_text")
    if not raw_text:
        parts = [summary, description] + [str(c) for c in comments]
        raw_text = "\n\n".join([p for p in parts if str(p).strip()])

    src_patch = patch.get("src") or empty_patch_block()
    test_patch = patch.get("test")

    full_patch = patch.get("full") or src_patch.get("raw", "") or ""
    add_patch = patch.get("add") or src_patch.get("add", "") or ""
    remove_patch = patch.get("remove") or src_patch.get("remove", "") or ""

    return {
        "dataset": dataset,
        "instance_id": str(instance_id),
        "project": project,
        "repo": repo,
        "language": language,
        "bug_report": {
            "summary": summary,
            "description": description,
            "comments": comments,
            "raw_text": raw_text or "",
            "source": bug_report.get("source"),
        },
        "tests": {
            "trigger": tests.get("trigger", []) or [],
            "relevant": tests.get("relevant", []) or [],
            "test_code": tests.get("test_code"),
            "test_yaml": tests.get("test_yaml"),
        },
        "patch": {
            "src": {
                "raw": src_patch.get("raw", "") or "",
                "add": src_patch.get("add", "") or "",
                "remove": src_patch.get("remove", "") or "",
                "files": src_patch.get("files", []) or [],
            },
            "test": test_patch,
            "full": full_patch,
            "add": add_patch,
            "remove": remove_patch,
        },
        "metadata": metadata,
    }


def validate_triplet(triplet: Dict[str, Any]) -> None:
    required = [
        "dataset",
        "instance_id",
        "language",
        "bug_report",
        "tests",
        "patch",
        "metadata",
    ]

    missing = [k for k in required if k not in triplet]
    if missing:
        raise ValueError(f"Triplet missing required keys: {missing}")

    if not str(triplet.get("instance_id", "")).strip():
        raise ValueError("Triplet has empty instance_id")

    bug_report = triplet.get("bug_report", {})
    if "raw_text" not in bug_report:
        raise ValueError("Triplet bug_report must contain raw_text")

    patch = triplet.get("patch", {})
    if "full" not in patch:
        raise ValueError("Triplet patch must contain full")


def compute_corpus_stats(triplets: List[Dict[str, Any]]) -> Dict[str, Any]:
    datasets: Dict[str, int] = {}
    languages: Dict[str, int] = {}
    projects: Dict[str, int] = {}
    repos: Dict[str, int] = {}

    with_report = 0
    with_test = 0
    with_patch = 0
    with_structured_patch_views = 0

    for t in triplets:
        dataset = t.get("dataset", "unknown")
        language = t.get("language", "unknown")
        project = t.get("project")
        repo = t.get("repo")

        datasets[dataset] = datasets.get(dataset, 0) + 1
        languages[language] = languages.get(language, 0) + 1

        if project:
            projects[project] = projects.get(project, 0) + 1

        if repo:
            repos[repo] = repos.get(repo, 0) + 1

        if str(t.get("bug_report", {}).get("raw_text", "")).strip():
            with_report += 1

        tests = t.get("tests", {})
        if tests.get("test_code") or tests.get("trigger") or tests.get("relevant"):
            with_test += 1

        patch = t.get("patch", {})
        if str(patch.get("full", "")).strip():
            with_patch += 1

        if (
            str(patch.get("full", "")).strip()
            and str(patch.get("add", "")).strip()
            and str(patch.get("remove", "")).strip()
        ):
            with_structured_patch_views += 1

    total = len(triplets)

    def ratio(x: int) -> float:
        return round(x / total, 6) if total else 0.0

    return {
        "total_instances": total,
        "datasets": datasets,
        "languages": languages,
        "projects": projects,
        "repos": repos,
        "with_bug_report": with_report,
        "with_tests": with_test,
        "with_patch": with_patch,
        "with_structured_patch_views": with_structured_patch_views,
        "coverage": {
            "bug_report": ratio(with_report),
            "tests": ratio(with_test),
            "patch": ratio(with_patch),
            "structured_patch_views": ratio(with_structured_patch_views),
        },
    }
