#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: Any) -> List[str]:
    text = normalize_text(text).lower()

    if not text:
        return []

    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]", text)


def tokenize_identifiers(text: Any) -> List[str]:
    text = normalize_text(text)

    if not text:
        return []

    return re.findall(r"[A-Za-z_][A-Za-z0-9_$.]*", text)


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def cosine_similarity(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    a = list(vec_a)
    b = list(vec_b)

    if not a or not b or len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))

    if na == 0.0 or nb == 0.0:
        return 0.0

    return dot / (na * nb)


def flatten_signal_dict(signals: Dict[str, Any]) -> str:
    parts = []

    for key, value in signals.items():
        if isinstance(value, list):
            for item in value:
                if item:
                    parts.append(str(item))
        elif value:
            parts.append(str(value))

    return "\n".join(parts).strip()


def metric_result(
    *,
    metric: str,
    value: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out = {
        "metric": metric,
        "value": round(float(value), 6),
    }

    if extra:
        out.update(extra)

    return out


def compute_basic_lengths(text_a: Any, text_b: Any) -> Dict[str, int]:
    a = normalize_text(text_a)
    b = normalize_text(text_b)

    return {
        "a_chars": len(a),
        "b_chars": len(b),
        "a_tokens": len(tokenize_words(a)),
        "b_tokens": len(tokenize_words(b)),
        "a_identifiers": len(tokenize_identifiers(a)),
        "b_identifiers": len(tokenize_identifiers(b)),
    }
