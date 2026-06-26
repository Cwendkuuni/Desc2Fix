#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

from desc2fix.metrics.common import normalize_text, tokenize_words
from desc2fix.metrics.lexical import jaccard_similarity


@lru_cache(maxsize=1)
def load_spacy_model():
    try:
        import spacy

        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return spacy.blank("en")
    except Exception:
        return None


def fallback_pos_tags(text: Any) -> List[str]:
    """
    Lightweight fallback when spaCy is unavailable.

    This is not linguistically rich, but it keeps smoke tests and replication
    package validation running.
    """
    tokens = tokenize_words(text)
    tags = []

    for tok in tokens:
        if tok.isdigit():
            tags.append("NUM")
        elif tok in {".", ",", ":", ";", "(", ")", "{", "}", "[", "]"}:
            tags.append("PUNCT")
        elif tok.lower() in {"if", "for", "while", "return", "try", "catch", "except"}:
            tags.append("KEYWORD")
        elif tok[:1].isupper():
            tags.append("PROPN")
        else:
            tags.append("WORD")

    return tags


def pos_tags(text: Any) -> List[str]:
    text = normalize_text(text)

    if not text:
        return []

    nlp = load_spacy_model()

    if nlp is None:
        return fallback_pos_tags(text)

    try:
        doc = nlp(text[:100000])
        tags = []

        for token in doc:
            if token.is_space:
                continue

            tag = token.pos_ or token.tag_ or "X"
            tags.append(tag)

        if not tags:
            return fallback_pos_tags(text)

        return tags

    except Exception:
        return fallback_pos_tags(text)


def pos_overlap(text_a: Any, text_b: Any) -> float:
    return jaccard_similarity(pos_tags(text_a), pos_tags(text_b))


def pos_metrics(text_a: Any, text_b: Any) -> Dict[str, float]:
    return {
        "pos_overlap": round(pos_overlap(text_a, text_b), 6),
    }
