#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from desc2fix.extraction.schema import SIGNAL_FIELDS, normalize_signals


def jaccard(a: List[str], b: List[str]) -> float:
    sa = set()
    sb = set()

    for x in a:
        sa.update(str(x).lower().split())

    for x in b:
        sb.update(str(x).lower().split())

    if not sa and not sb:
        return 1.0

    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def signal_count(signals: Dict[str, Any]) -> int:
    norm = normalize_signals(signals)
    return sum(len(norm[field]) for field in SIGNAL_FIELDS)


def non_empty_fields(signals: Dict[str, Any]) -> List[str]:
    norm = normalize_signals(signals)
    return [field for field in SIGNAL_FIELDS if norm.get(field)]


def completeness_ratio(signals: Dict[str, Any]) -> float:
    fields = non_empty_fields(signals)
    return len(fields) / len(SIGNAL_FIELDS)


def compare_two_models(
    signals_a: Dict[str, Any],
    signals_b: Dict[str, Any],
) -> Dict[str, Any]:
    a = normalize_signals(signals_a)
    b = normalize_signals(signals_b)

    field_scores = {
        field: round(jaccard(a.get(field, []), b.get(field, [])), 6)
        for field in SIGNAL_FIELDS
    }

    macro = sum(field_scores.values()) / len(SIGNAL_FIELDS)

    return {
        "field_jaccard": field_scores,
        "macro_jaccard": round(macro, 6),
    }


def summarize_model_outputs(
    *,
    dataset: str,
    model: str,
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    total = len(rows)
    available = [r for r in rows if r.get("available")]

    field_non_empty = Counter()
    total_signal_items = 0
    completeness_values = []

    for r in available:
        signals = normalize_signals(r.get("signals", {}))

        total_signal_items += signal_count(signals)
        completeness_values.append(completeness_ratio(signals))

        for field in SIGNAL_FIELDS:
            if signals.get(field):
                field_non_empty[field] += 1

    def rate(x: int) -> float:
        return round(x / len(available), 6) if available else 0.0

    return {
        "dataset": dataset,
        "model": model,
        "expected_instances": total,
        "available_instances": len(available),
        "missing_instances": total - len(available),
        "availability_rate": round(len(available) / total, 6) if total else 0.0,
        "avg_signal_items_per_instance": round(total_signal_items / len(available), 6)
        if available
        else 0.0,
        "avg_field_completeness": round(sum(completeness_values) / len(completeness_values), 6)
        if completeness_values
        else 0.0,
        "field_non_empty_counts": {field: field_non_empty[field] for field in SIGNAL_FIELDS},
        "field_non_empty_rates": {field: rate(field_non_empty[field]) for field in SIGNAL_FIELDS},
    }


def summarize_pairwise_consistency(
    *,
    dataset: str,
    model_a: str,
    model_b: str,
    rows_a: List[Dict[str, Any]],
    rows_b: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_id_a = {r["instance_id"]: r for r in rows_a if r.get("available")}
    by_id_b = {r["instance_id"]: r for r in rows_b if r.get("available")}

    common_ids = sorted(set(by_id_a) & set(by_id_b))

    field_totals = Counter()
    macro_scores = []

    per_instance = []

    for iid in common_ids:
        comp = compare_two_models(
            by_id_a[iid].get("signals", {}),
            by_id_b[iid].get("signals", {}),
        )

        macro_scores.append(comp["macro_jaccard"])

        for field, score in comp["field_jaccard"].items():
            field_totals[field] += score

        per_instance.append(
            {
                "dataset": dataset,
                "instance_id": iid,
                "model_a": model_a,
                "model_b": model_b,
                "macro_jaccard": comp["macro_jaccard"],
                **{
                    f"{field}_jaccard": comp["field_jaccard"][field]
                    for field in SIGNAL_FIELDS
                },
            }
        )

    field_avg = {
        field: round(field_totals[field] / len(common_ids), 6) if common_ids else 0.0
        for field in SIGNAL_FIELDS
    }

    return {
        "summary": {
            "dataset": dataset,
            "model_a": model_a,
            "model_b": model_b,
            "common_instances": len(common_ids),
            "avg_macro_jaccard": round(sum(macro_scores) / len(macro_scores), 6)
            if macro_scores
            else 0.0,
            "avg_field_jaccard": field_avg,
        },
        "per_instance": per_instance,
    }
