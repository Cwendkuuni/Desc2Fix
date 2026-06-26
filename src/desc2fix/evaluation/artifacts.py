#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from desc2fix.extraction.schema import normalize_signals
from desc2fix.extraction.signal_extractor import build_bug_report_text
from desc2fix.metrics.structured import all_signal_text
from desc2fix.rq2_report_test.metric_alignment import get_test_text
from desc2fix.rq3_report_patch.metric_alignment import get_patch_text


def structured_signals_json(signals: Optional[Dict[str, Any]]) -> str:
    if not signals:
        return ""
    return json.dumps(
        normalize_signals(signals),
        ensure_ascii=False,
        indent=2,
    )


def artifact_pair_for_judge(
    *,
    rq: str,
    triplet: Dict[str, Any],
    scenario: str,
    patch_view: str = "full",
    signals: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], str]:
    """
    Return:
      - variables for the legacy prompt template
      - artifact_pair name

    RQ mapping:
      rq2          = bug report ↔ test
      rq3          = bug report ↔ patch
      rq4_internal = test ↔ patch
    """

    bug_report_raw = build_bug_report_text(triplet)
    signals_json = structured_signals_json(signals)
    test_code = get_test_text(triplet)
    patch_code = get_patch_text(triplet, patch_view=patch_view)

    if rq == "rq2":
        variables = {
            "BUG_REPORT_RAW": bug_report_raw,
            "BUG_REPORT_STRUCTURED_JSON": signals_json,
            "TEST_CODE": test_code,
        }
        return variables, "report_test"

    if rq == "rq3":
        variables = {
            "BUG_REPORT_RAW": bug_report_raw,
            "BUG_REPORT_STRUCTURED_JSON": signals_json,
            "PATCH_CODE": patch_code,
            "PATCH_TYPE": patch_view,
        }
        return variables, "report_patch"

    if rq == "rq4_internal":
        variables = {
            "BUG_REPORT_STRUCTURED_JSON": signals_json,
            "TEST_CODE": test_code,
            "PATCH_CODE": patch_code,
            "PATCH_TYPE": patch_view,
        }
        return variables, "test_patch"

    raise ValueError(f"Unknown RQ: {rq}")


def prompt_path_for_rq_scenario(rq: str, scenario: str) -> str:
    if scenario not in {"raw", "structured"}:
        raise ValueError(f"Unknown scenario: {scenario}")

    if rq == "rq2":
        return f"prompts/report_test_alignment/{scenario}_prompt.txt"

    if rq == "rq3":
        return f"prompts/report_patch_alignment/{scenario}_prompt.txt"

    if rq == "rq4_internal":
        return f"prompts/test_patch_alignment_internal/{scenario}_prompt.txt"

    raise ValueError(f"Unknown RQ: {rq}")
