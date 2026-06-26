#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List


SIGNAL_FIELDS = [
    "exception",
    "api_involved",
    "input_conditions",
    "reproduction_steps",
    "expected",
    "actual",
]


def normalize_signal_value(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    if isinstance(value, list):
        out = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out

    s = str(value).strip()
    return [s] if s else []


def normalize_signals(raw: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        field: normalize_signal_value(raw.get(field))
        for field in SIGNAL_FIELDS
    }


def empty_signals() -> Dict[str, List[str]]:
    return {field: [] for field in SIGNAL_FIELDS}


def signals_to_text(signals: Dict[str, Any]) -> str:
    lines = []

    norm = normalize_signals(signals)

    for field in SIGNAL_FIELDS:
        values = norm.get(field, [])
        if values:
            lines.append(f"{field}:")
            for v in values:
                lines.append(f"- {v}")

    return "\n".join(lines).strip()
