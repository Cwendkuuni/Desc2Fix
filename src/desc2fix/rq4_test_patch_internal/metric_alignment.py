#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict

from desc2fix.metrics.common import compute_basic_lengths, normalize_text
from desc2fix.metrics.embeddings import embedding_metrics
from desc2fix.metrics.lexical import lexical_metrics
from desc2fix.metrics.pos import pos_metrics
from desc2fix.rq2_report_test.metric_alignment import get_test_text
from desc2fix.rq3_report_patch.metric_alignment import PATCH_VIEWS, get_patch_text


def compute_test_patch_metrics(
    *,
    triplet: Dict[str, Any],
    patch_view: str,
    full: bool = False,
    include_openai_embeddings: bool = False,
) -> Dict[str, Any]:
    """
    Internal RQ4:
      test artifact ↔ patch artifact

    This is useful to inspect whether generated or trigger tests are structurally
    and semantically close to the code changes.
    """
    if patch_view not in PATCH_VIEWS:
        raise ValueError(f"Unknown patch_view={patch_view}. Expected one of {PATCH_VIEWS}")

    test_text = get_test_text(triplet)
    patch_text = get_patch_text(triplet, patch_view=patch_view)

    out: Dict[str, Any] = {
        "dataset": triplet.get("dataset"),
        "instance_id": triplet.get("instance_id"),
        "project": triplet.get("project"),
        "repo": triplet.get("repo"),
        "scenario": "raw",
        "artifact_pair": "test_patch",
        "patch_view": patch_view,
        "has_test_text": bool(normalize_text(test_text)),
        "has_patch_text": bool(normalize_text(patch_text)),
    }

    out.update(compute_basic_lengths(test_text, patch_text))

    out.update(lexical_metrics(test_text, patch_text))
    out.update(pos_metrics(test_text, patch_text))

    if full:
        out.update(
            embedding_metrics(
                test_text,
                patch_text,
                include_sbert=True,
                include_codebert=True,
                include_openai=include_openai_embeddings,
            )
        )

    return out
