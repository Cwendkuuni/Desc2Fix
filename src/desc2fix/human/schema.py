#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


SCORE_FIELDS = [
    "alignment_score",
    "coverage",
    "correctness",
    "specificity",
]


def normalize_dataset(value: Any) -> str:
    s = str(value or "").strip().lower()

    if s in {"defects4j", "d4j"}:
        return "Defects4J"

    if s in {"swtbench", "swt-bench", "swebench", "swe-bench"}:
        return "SWTBench"

    if "defects" in s:
        return "Defects4J"

    if "swt" in s or "swe" in s:
        return "SWTBench"

    return str(value or "").strip()


def infer_rq_from_path(path: str) -> str:
    p = path.replace("\\", "/").lower()

    if "rq2" in p:
        return "rq2"

    if "rq3" in p:
        return "rq3"

    if "rq4" in p:
        return "rq4_internal"

    raise ValueError(f"Cannot infer RQ from path: {path}")


def infer_rater_from_path(path: str) -> str:
    name = path.lower()

    m = re.search(r"human[_-]?(\d+)", name)
    if m:
        return f"human{m.group(1)}"

    if "human_2" in name or "human2" in name:
        return "human2"

    if "human" in name:
        return "human1"

    return "unknown"


def parse_scenario(value: Any) -> Tuple[str, Optional[str], str]:
    """
    Return:
      representation: RAW or STRUCTURED
      patch_view: full/add/remove/None
      scenario_norm: original normalized scenario label
    """
    s = str(value or "").strip()
    s_upper = s.upper()

    if s_upper.startswith("STRUCTURED"):
        representation = "STRUCTURED"
    elif s_upper.startswith("RAW"):
        representation = "RAW"
    else:
        representation = s_upper

    patch_view = None
    low = s.lower()

    for view in ["full", "add", "remove"]:
        if low.endswith(f"_{view}") or low == view:
            patch_view = view
            break

    return representation, patch_view, s_upper


def pair_id_from_record(obj: Dict[str, Any]) -> str:
    dataset = normalize_dataset(obj.get("dataset"))

    if dataset == "Defects4J":
        project = obj.get("project")
        bug_id = obj.get("bug_id")

        if not bug_id and obj.get("instance_id"):
            instance = str(obj.get("instance_id"))
            if "-" in instance:
                project, bug_id = instance.split("-", 1)

        return f"{project}__{bug_id}"

    instance_id = obj.get("instance_id")
    if instance_id:
        return str(instance_id)

    repo = str(obj.get("repo") or "").replace("/", "__")
    bug_id = obj.get("bug_id")
    if repo and bug_id:
        return f"{repo}__{bug_id}"

    return str(obj.get("pair_id") or "")


def score_value(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None
