#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from desc2fix.extraction.schema import normalize_signals
from desc2fix.llm.clients import BaseLLMClient
from desc2fix.llm.json_parser import parse_json_response
from desc2fix.llm.prompts import load_prompt, render_prompt


DEFAULT_EXTRACTION_PROMPT = "prompts/extraction/structured_bug_signal_extraction.txt"


def build_bug_report_text(triplet: Dict[str, Any]) -> str:
    br = triplet.get("bug_report", {}) or {}

    raw_text = br.get("raw_text")
    if raw_text and str(raw_text).strip():
        return str(raw_text).strip()

    parts = [
        br.get("summary", ""),
        br.get("description", ""),
    ]

    comments = br.get("comments", []) or []
    if isinstance(comments, list):
        parts.extend(comments)

    return "\n\n".join([str(p).strip() for p in parts if str(p).strip()])


def extract_signals_for_triplet(
    *,
    triplet: Dict[str, Any],
    client: BaseLLMClient,
    prompt_path: str | Path = DEFAULT_EXTRACTION_PROMPT,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    bug_report_text = build_bug_report_text(triplet)

    if not bug_report_text:
        raise ValueError(f"Empty bug report for instance {triplet.get('instance_id')}")

    template = load_prompt(prompt_path)
    prompt = render_prompt(template, BUG_REPORT=bug_report_text)

    raw_response = client.generate(
        prompt=prompt,
        system_prompt=system_prompt,
    )

    parsed = parse_json_response(raw_response)
    signals = normalize_signals(parsed)

    # return {
    #     "dataset": triplet.get("dataset"),
    #     "model": client.spec.key,
    #     "model_name": client.spec.model_name,
    #     "instance_id": triplet.get("instance_id"),
    #     "project": triplet.get("project"),
    #     "repo": triplet.get("repo"),
    #     "signals": signals,
    #     "raw_response": raw_response,
    #     "prompt_path": str(prompt_path),
    # }
    return {
        "dataset": triplet.get("dataset"),
        "model": client.spec.key,
        "model_name": client.spec.model_name,
        "instance_id": triplet.get("instance_id"),
        "project": triplet.get("project"),
        "repo": triplet.get("repo"),

        # Nouveau format canonique
        "signals": signals,

        # Compatibilité avec les anciens scripts
        "extracted_signals": signals,

        "raw_response": raw_response,
        "llm_response": raw_response,
        "prompt_path": str(prompt_path),
    }
