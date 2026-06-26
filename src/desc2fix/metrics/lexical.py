#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from desc2fix.metrics.common import normalize_text, safe_div, tokenize_identifiers, tokenize_words


def jaccard_similarity(tokens_a: Sequence[str], tokens_b: Sequence[str]) -> float:
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if not set_a and not set_b:
        return 1.0

    if not set_a or not set_b:
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def token_jaccard(text_a: Any, text_b: Any) -> float:
    return jaccard_similarity(
        tokenize_words(text_a),
        tokenize_words(text_b),
    )


def identifier_jaccard(text_a: Any, text_b: Any) -> float:
    ids_a = [x.lower() for x in tokenize_identifiers(text_a)]
    ids_b = [x.lower() for x in tokenize_identifiers(text_b)]

    return jaccard_similarity(ids_a, ids_b)


def lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)

    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)

        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])

        prev = curr

    return prev[-1]


def rouge_l_f1(text_a: Any, text_b: Any) -> float:
    """
    ROUGE-L F1 between text_a and text_b.
    """
    a = tokenize_words(text_a)
    b = tokenize_words(text_b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    lcs = lcs_length(a, b)

    precision = safe_div(lcs, len(b))
    recall = safe_div(lcs, len(a))

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def lexical_metrics(text_a: Any, text_b: Any) -> Dict[str, float]:
    a = normalize_text(text_a)
    b = normalize_text(text_b)

    return {
        "rouge_l": round(rouge_l_f1(a, b), 6),
        "jaccard": round(token_jaccard(a, b), 6),
        "identifier_jaccard": round(identifier_jaccard(a, b), 6),
    }
