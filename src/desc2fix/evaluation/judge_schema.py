#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict


JUDGE_FIELDS = [
    "alignment_score",
    "coverage",
    "correctness",
    "specificity",
]


def clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except Exception:
        score = 1

    return max(1, min(5, score))


def normalize_judge_output(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = {}

    for field in JUDGE_FIELDS:
        out[field] = clamp_score(raw.get(field, 1))

    # IMPORTANT: old experiments used "justification", not "rationale".
    out["justification"] = str(
        raw.get("justification", raw.get("rationale", "")) or ""
    ).strip()

    return out
