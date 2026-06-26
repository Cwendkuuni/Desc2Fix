#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, Optional

from desc2fix.extraction.schema import normalize_signals
from desc2fix.extraction.signal_extractor import build_bug_report_text
from desc2fix.metrics.common import compute_basic_lengths, normalize_text
from desc2fix.metrics.embeddings import embedding_metrics
from desc2fix.metrics.lexical import lexical_metrics
from desc2fix.metrics.pos import pos_metrics
from desc2fix.metrics.structured import all_signal_text, structured_metrics


def get_test_text(triplet: Dict[str, Any]) -> str:
    """
    Return the best available test artifact text.

    Defects4J:
      - trigger/relevant test identifiers
      - test patch if available

    SWT-Bench:
      - test_code
      - test_yaml
      - trigger/relevant identifiers
    """
    tests = triplet.get("tests", {}) or {}
    patch = triplet.get("patch", {}) or {}

    parts = []

    test_code = tests.get("test_code")
    test_yaml = tests.get("test_yaml")

    if test_code:
        parts.append(str(test_code))

    if test_yaml:
        parts.append(str(test_yaml))

    trigger = tests.get("trigger") or []
    relevant = tests.get("relevant") or []

    if trigger:
        parts.append("\n".join([str(x) for x in trigger]))

    if relevant:
        parts.append("\n".join([str(x) for x in relevant]))

    test_patch = patch.get("test")
    if isinstance(test_patch, dict):
        raw = test_patch.get("raw", "")
        add = test_patch.get("add", "")
        remove = test_patch.get("remove", "")

        for value in [raw, add, remove]:
            if value:
                parts.append(str(value))

    return "\n\n".join([p.strip() for p in parts if p and str(p).strip()])


def get_report_text_for_scenario(
    *,
    triplet: Dict[str, Any],
    signals: Optional[Dict[str, Any]],
    scenario: str,
) -> str:
    if scenario == "raw":
        return build_bug_report_text(triplet)

    if scenario == "structured":
        if not signals:
            return ""
        return all_signal_text(normalize_signals(signals))

    raise ValueError(f"Unknown scenario: {scenario}")


def compute_report_test_metrics(
    *,
    triplet: Dict[str, Any],
    scenario: str,
    signals: Optional[Dict[str, Any]] = None,
    full: bool = False,
    include_openai_embeddings: bool = False,
) -> Dict[str, Any]:
    report_text = get_report_text_for_scenario(
        triplet=triplet,
        signals=signals,
        scenario=scenario,
    )

    test_text = get_test_text(triplet)

    out: Dict[str, Any] = {
        "dataset": triplet.get("dataset"),
        "instance_id": triplet.get("instance_id"),
        "project": triplet.get("project"),
        "repo": triplet.get("repo"),
        "scenario": scenario,
        "artifact_pair": "report_test",
        "has_report_text": bool(normalize_text(report_text)),
        "has_test_text": bool(normalize_text(test_text)),
    }

    out.update(compute_basic_lengths(report_text, test_text))

    out.update(lexical_metrics(report_text, test_text))
    out.update(pos_metrics(report_text, test_text))

    if scenario == "structured" and signals:
        out.update(
            structured_metrics(
                signals=signals,
                target_text=test_text,
            )
        )

    if full:
        out.update(
            embedding_metrics(
                report_text,
                test_text,
                include_sbert=True,
                include_codebert=True,
                include_openai=include_openai_embeddings,
            )
        )

    return out
