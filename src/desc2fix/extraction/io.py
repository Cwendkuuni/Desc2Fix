#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from desc2fix.utils.io import safe_filename, write_json, read_json


def signal_output_path(
    *,
    structured_root: Path,
    dataset: str,
    model: str,
    triplet: Dict[str, Any],
) -> Path:
    if dataset == "defects4j":
        project = safe_filename(str(triplet.get("project") or "unknown"))
        bug_id = str(triplet.get("metadata", {}).get("bug_id") or triplet.get("instance_id"))
        bug_id = safe_filename(bug_id.replace(f"{project}-", ""))
        return structured_root / dataset / model / project / bug_id / "bug_signals.json"

    if dataset == "swtbench":
        repo = safe_filename(str(triplet.get("repo") or "unknown"))
        instance_id = safe_filename(str(triplet.get("instance_id")))
        return structured_root / dataset / model / repo / instance_id / "bug_signals.json"

    instance_id = safe_filename(str(triplet.get("instance_id")))
    return structured_root / dataset / model / instance_id / "bug_signals.json"


def write_signal_file(path: Path, payload: Dict[str, Any]) -> None:
    write_json(path, payload)


def read_signal_file(path: Path) -> Dict[str, Any]:
    return read_json(path)
