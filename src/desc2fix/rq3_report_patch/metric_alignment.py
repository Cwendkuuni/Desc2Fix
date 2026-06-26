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


PATCH_VIEWS = ["full", "add", "remove"]


def get_patch_text(
    triplet: Dict[str, Any],
    *,
    patch_view: str = "full",
) -> str:
    """
    Return patch text according to the requested view.

    Supported views:
      - full: complete normalized patch representation
      - add: added code only
      - remove: removed code only
    """
    if patch_view not in PATCH_VIEWS:
        raise ValueError(f"Unknown patch_view={patch_view}. Expected one of {PATCH_VIEWS}")

    patch = triplet.get("patch", {}) or {}

    value = patch.get(patch_view)
    if value:
        return str(value)

    src = patch.get("src", {}) or {}
    value = src.get(patch_view)
    if value:
        return str(value)

    if patch_view == "full":
        return str(src.get("raw", "") or "")

    return ""


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


def compute_report_patch_metrics(
    *,
    triplet: Dict[str, Any],
    scenario: str,
    patch_view: str,
    signals: Optional[Dict[str, Any]] = None,
    full: bool = False,
    include_openai_embeddings: bool = False,
) -> Dict[str, Any]:
    report_text = get_report_text_for_scenario(
        triplet=triplet,
        signals=signals,
        scenario=scenario,
    )

    patch_text = get_patch_text(triplet, patch_view=patch_view)

    out: Dict[str, Any] = {
        "dataset": triplet.get("dataset"),
        "instance_id": triplet.get("instance_id"),
        "project": triplet.get("project"),
        "repo": triplet.get("repo"),
        "scenario": scenario,
        "artifact_pair": "report_patch",
        "patch_view": patch_view,
        "has_report_text": bool(normalize_text(report_text)),
        "has_patch_text": bool(normalize_text(patch_text)),
    }

    out.update(compute_basic_lengths(report_text, patch_text))

    out.update(lexical_metrics(report_text, patch_text))
    out.update(pos_metrics(report_text, patch_text))

    if scenario == "structured" and signals:
        out.update(
            structured_metrics(
                signals=signals,
                target_text=patch_text,
            )
        )

    if full:
        out.update(
            embedding_metrics(
                report_text,
                patch_text,
                include_sbert=True,
                include_codebert=True,
                include_openai=include_openai_embeddings,
            )
        )

    return out
