#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from typing import Any, Dict


def strip_code_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def extract_json_object(text: str) -> str:
    text = strip_code_fences(text)

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response.")

    candidate = text[start : end + 1]
    json.loads(candidate)
    return candidate


def parse_json_response(text: str) -> Dict[str, Any]:
    obj_text = extract_json_object(text)
    obj = json.loads(obj_text)

    if not isinstance(obj, dict):
        raise ValueError("Expected a JSON object.")

    return obj
