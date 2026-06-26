#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import html
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

from desc2fix.corpus.schema import compute_corpus_stats, make_triplet, validate_triplet
from desc2fix.utils.io import ensure_dir, write_json


USER_AGENT = {"User-Agent": "Desc2Fix/1.0"}

DEFAULT_PROJECTS = [
    "Cli",
    "Closure",
    "Codec",
    "Collections",
    "Compress",
    "Csv",
    "Gson",
    "JacksonCore",
    "JacksonDatabind",
    "JacksonXml",
    "Jsoup",
    "JxPath",
    "Lang",
    "Math",
    "Mockito",
    "Time",
]


# ---------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------

def get_cache_path(cache_dir: Path, project: str, report_id: str) -> Path:
    safe_report_id = str(report_id).replace("/", "_")
    return cache_dir / f"{project}_{safe_report_id}.json"


def cache_load(cache_dir: Path, project: str, report_id: str) -> Optional[Dict[str, Any]]:
    path = get_cache_path(cache_dir, project, report_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def cache_save(cache_dir: Path, project: str, report_id: str, data: Dict[str, Any]) -> None:
    ensure_dir(cache_dir)
    path = get_cache_path(cache_dir, project, report_id)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------
# Bug report fetchers
# ---------------------------------------------------------------------

def fetch_jira_report(report_id: str) -> Optional[Dict[str, Any]]:
    url = f"https://issues.apache.org/jira/rest/api/2/issue/{report_id}"
    response = requests.get(url, headers=USER_AGENT, timeout=30)

    if response.status_code != 200:
        return None

    fields = response.json().get("fields", {})

    return {
        "summary": fields.get("summary", "") or "",
        "description": fields.get("description", "") or "",
        "comments": [
            c.get("body", "")
            for c in fields.get("comment", {}).get("comments", [])
            if c.get("body")
        ],
        "source": "jira",
    }


def fetch_googlecode_report(url: str) -> Optional[Dict[str, Any]]:
    response = requests.get(url, headers=USER_AGENT, timeout=30)

    if response.status_code != 200:
        return None

    try:
        data = response.json()
    except Exception:
        return None

    issue = data.get("issue", data)

    title = (
        issue.get("title")
        or issue.get("summary")
        or data.get("title")
        or data.get("summary")
        or ""
    )

    content = (
        issue.get("content")
        or issue.get("description")
        or data.get("content")
        or data.get("description")
        or ""
    )

    comments = []
    for comment in data.get("comments", []):
        text = comment.get("content", "").strip()
        if text:
            comments.append(html.unescape(text))

    if not content and comments and len(comments[0]) >= 50:
        content = comments[0]

    return {
        "summary": html.unescape(str(title).strip()),
        "description": html.unescape(str(content).strip()),
        "comments": comments,
        "source": "googlecode",
    }


def fetch_github_issue(url: str) -> Optional[Dict[str, Any]]:
    match = re.match(r"https://github.com/([^/]+)/([^/]+)/issues/(\d+)", url)

    if not match:
        return None

    owner, repo, issue_number = match.groups()
    api_url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"

    response = requests.get(
        api_url,
        headers={"Accept": "application/vnd.github+json", **USER_AGENT},
        timeout=30,
    )

    if response.status_code != 200:
        return None

    data = response.json()
    comments = []

    if data.get("comments_url") and data.get("comments", 0) > 0:
        comments_response = requests.get(data["comments_url"], headers=USER_AGENT, timeout=30)
        if comments_response.status_code == 200:
            comments = [
                c.get("body", "")
                for c in comments_response.json()
                if c.get("body")
            ]

    return {
        "summary": data.get("title", "") or "",
        "description": data.get("body", "") or "",
        "comments": comments,
        "source": "github",
    }


def fetch_bug_report(
    *,
    project: str,
    report_id: str,
    report_url: str,
    cache_dir: Path,
    sleep_seconds: float = 0.5,
) -> Optional[Dict[str, Any]]:
    cached = cache_load(cache_dir, project, report_id)
    if cached:
        return cached

    report = None

    try:
        if "jira" in report_url.lower():
            report = fetch_jira_report(report_id)
        elif "googleapis.com" in report_url.lower():
            report = fetch_googlecode_report(report_url)
        elif "github.com" in report_url.lower():
            report = fetch_github_issue(report_url)
    except Exception as exc:
        print(f"[WARN] Failed to fetch report {project}/{report_id}: {exc}")
        report = None

    if report:
        cache_save(cache_dir, project, report_id, report)

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return report


# ---------------------------------------------------------------------
# Patch and tests extraction
# ---------------------------------------------------------------------

def parse_patch(patch_text: str) -> Dict[str, Any]:
    added: List[str] = []
    removed: List[str] = []
    files = re.findall(r"diff --git a/(.*?) b/", patch_text)

    for line in patch_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:].rstrip())
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:].rstrip())

    return {
        "raw": patch_text,
        "add": "\n".join(added).strip(),
        "remove": "\n".join(removed).strip(),
        "files": files,
        "added_lines": added,
        "removed_lines": removed,
    }


def load_patches(project_dir: Path, bug_id: str) -> Dict[str, Any]:
    patches_dir = project_dir / "patches"
    patches: Dict[str, Any] = {}

    for kind in ["src", "test"]:
        patch_path = patches_dir / f"{bug_id}.{kind}.patch"

        if patch_path.exists():
            text = patch_path.read_text(encoding="utf-8", errors="ignore")
            patches[kind] = parse_patch(text)

    return patches


def load_tests(project_dir: Path, bug_id: str) -> Dict[str, List[str]]:
    tests = {}

    for name in ["trigger", "relevant"]:
        path = project_dir / f"{name}_tests" / str(bug_id)
        if path.exists():
            tests[name] = [
                line.strip()
                for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
        else:
            tests[name] = []

    return tests


# ---------------------------------------------------------------------
# Raw builder
# ---------------------------------------------------------------------

def process_project_raw(
    *,
    project_name: str,
    projects_root: Path,
    cache_dir: Path,
    sleep_seconds: float = 0.5,
) -> List[Dict[str, Any]]:
    project_dir = projects_root / project_name
    active_bugs_csv = project_dir / "active-bugs.csv"

    if not active_bugs_csv.exists():
        print(f"[WARN] Missing active-bugs.csv for {project_name}: {active_bugs_csv}")
        return []

    triplets: List[Dict[str, Any]] = []

    with active_bugs_csv.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows, desc=f"Defects4J/{project_name}", ncols=100):
        bug_id = str(row.get("bug.id", "")).strip()
        if not bug_id:
            continue

        report_id = str(row.get("report.id") or f"{project_name.upper()}-{bug_id}").strip()
        report_url = str(row.get("report.url") or "").strip()

        report = fetch_bug_report(
            project=project_name,
            report_id=report_id,
            report_url=report_url,
            cache_dir=cache_dir,
            sleep_seconds=sleep_seconds,
        )

        if not report:
            print(f"[WARN] Missing bug report for {project_name}-{bug_id} ({report_id})")
            continue

        patches = load_patches(project_dir, bug_id)
        tests = load_tests(project_dir, bug_id)

        src_patch = patches.get(
            "src",
            {"raw": "", "add": "", "remove": "", "files": []},
        )
        test_patch = patches.get("test")

        instance_id = f"{project_name}-{bug_id}"

        triplet = make_triplet(
            dataset="defects4j",
            instance_id=instance_id,
            project=project_name,
            repo=None,
            language="java",
            bug_report={
                "summary": report.get("summary", ""),
                "description": report.get("description", ""),
                "comments": report.get("comments", []),
                "source": report.get("source", "unknown"),
            },
            tests={
                "trigger": tests.get("trigger", []),
                "relevant": tests.get("relevant", []),
                "test_code": None,
                "test_yaml": None,
            },
            patch={
                "src": src_patch,
                "test": test_patch,
                "full": src_patch.get("raw", ""),
                "add": src_patch.get("add", ""),
                "remove": src_patch.get("remove", ""),
            },
            metadata={
                "bug_id": bug_id,
                "project": project_name,
                "report_id": report_id,
                "report_url": report_url,
            },
        )

        validate_triplet(triplet)
        triplets.append(triplet)

    return triplets


def build_defects4j_raw(
    *,
    projects_root: Path,
    output_path: Path,
    stats_path: Path,
    cache_dir: Path,
    projects: Optional[List[str]] = None,
    sleep_seconds: float = 0.5,
) -> List[Dict[str, Any]]:
    projects_root = projects_root.expanduser().resolve()
    cache_dir = cache_dir.expanduser().resolve()

    if not projects_root.exists():
        raise FileNotFoundError(
            f"Defects4J projects root not found: {projects_root}. "
            "Run scripts/setup_benchmarks.py first or pass the correct --defects4j-projects-root."
        )

    projects = projects or DEFAULT_PROJECTS

    all_triplets: List[Dict[str, Any]] = []

    for project_name in projects:
        project_triplets = process_project_raw(
            project_name=project_name,
            projects_root=projects_root,
            cache_dir=cache_dir,
            sleep_seconds=sleep_seconds,
        )
        all_triplets.extend(project_triplets)

    write_json(output_path, all_triplets)
    write_json(stats_path, compute_corpus_stats(all_triplets))

    return all_triplets


# ---------------------------------------------------------------------
# Existing adapter, kept only as optional debug mode
# ---------------------------------------------------------------------

def build_defects4j_existing(
    *,
    existing_triplets_path: Path,
    output_path: Path,
    stats_path: Path,
) -> List[Dict[str, Any]]:
    from desc2fix.utils.io import read_json

    data = read_json(existing_triplets_path)
    rows = data.get("rows", data) if isinstance(data, dict) else data

    triplets = []

    for item in rows:
        project = item.get("project")
        bug_id = str(item.get("bug_id") or item.get("id") or item.get("instance_id"))
        instance_id = item.get("instance_id") or f"{project}-{bug_id}"

        br = item.get("bug_report", {}) or {}
        patches = item.get("patches", item.get("patch", {})) or {}
        tests = item.get("tests", {}) or {}

        src_patch = patches.get("src", {}) or {}
        test_patch = patches.get("test")

        src_norm = {
            "raw": src_patch.get("raw", "") or "",
            "add": "\n".join(src_patch.get("added", [])) if isinstance(src_patch.get("added"), list) else src_patch.get("add", ""),
            "remove": "\n".join(src_patch.get("removed", [])) if isinstance(src_patch.get("removed"), list) else src_patch.get("remove", ""),
            "files": src_patch.get("files", []) or [],
        }

        if test_patch:
            test_patch = {
                "raw": test_patch.get("raw", "") or "",
                "add": "\n".join(test_patch.get("added", [])) if isinstance(test_patch.get("added"), list) else test_patch.get("add", ""),
                "remove": "\n".join(test_patch.get("removed", [])) if isinstance(test_patch.get("removed"), list) else test_patch.get("remove", ""),
                "files": test_patch.get("files", []) or [],
            }

        triplet = make_triplet(
            dataset="defects4j",
            instance_id=instance_id,
            project=project,
            repo=None,
            language="java",
            bug_report={
                "summary": br.get("summary", ""),
                "description": br.get("description", ""),
                "comments": br.get("comments", []),
                "source": br.get("source", "unknown"),
            },
            tests={
                "trigger": tests.get("trigger", []),
                "relevant": tests.get("relevant", []),
                "test_code": None,
                "test_yaml": None,
            },
            patch={
                "src": src_norm,
                "test": test_patch,
                "full": src_norm.get("raw", ""),
                "add": src_norm.get("add", ""),
                "remove": src_norm.get("remove", ""),
            },
            metadata={
                "bug_id": bug_id,
                "project": project,
                "report_id": item.get("report_id"),
                "report_url": item.get("report_url"),
                "source_file": str(existing_triplets_path),
            },
        )

        validate_triplet(triplet)
        triplets.append(triplet)

    write_json(output_path, triplets)
    write_json(stats_path, compute_corpus_stats(triplets))

    return triplets
