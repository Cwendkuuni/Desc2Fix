#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List

from desc2fix.extraction.schema import SIGNAL_FIELDS, normalize_signals
from desc2fix.metrics.common import normalize_text, tokenize_words
from desc2fix.metrics.lexical import jaccard_similarity


def signal_field_text(signals: Dict[str, Any], field: str) -> str:
    norm = normalize_signals(signals)
    values = norm.get(field, [])
    return "\n".join(values).strip()


def all_signal_text(signals: Dict[str, Any]) -> str:
    norm = normalize_signals(signals)
    parts: List[str] = []

    for field in SIGNAL_FIELDS:
        values = norm.get(field, [])
        parts.extend(values)

    return "\n".join(parts).strip()


def signal_coverage_in_text(signals: Dict[str, Any], target_text: Any) -> Dict[str, float]:
    """
    For each signal field, compute token Jaccard between field signals and target text.
    """
    target = normalize_text(target_text)

    out: Dict[str, float] = {}

    for field in SIGNAL_FIELDS:
        field_text = signal_field_text(signals, field)
        score = jaccard_similarity(
            tokenize_words(field_text),
            tokenize_words(target),
        )
        out[f"structured_{field}_jaccard"] = round(score, 6)

    macro = sum(out.values()) / len(out) if out else 0.0
    out["structured_macro_jaccard"] = round(macro, 6)

    all_text = all_signal_text(signals)
    out["structured_all_jaccard"] = round(
        jaccard_similarity(
            tokenize_words(all_text),
            tokenize_words(target),
        ),
        6,
    )

    return out


def structured_metrics(
    *,
    signals: Dict[str, Any],
    target_text: Any,
) -> Dict[str, float]:
    return signal_coverage_in_text(signals, target_text)
