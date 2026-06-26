#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    provider: str
    model_name: str
    api_key_env: str
    base_url: Optional[str]
    enabled: bool = True
    max_total_tokens: int = 128000
    reserved_response_tokens: int = 16000
    temperature: float = 1.0
    top_p: float = 1.0
    tokenizer: str = "cl100k_base"


def model_spec_from_config(key: str, cfg: Dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        key=key,
        display_name=cfg.get("display_name", key),
        provider=cfg.get("provider", "openai"),
        model_name=cfg["model_name"],
        api_key_env=cfg["api_key_env"],
        base_url=cfg.get("base_url"),
        enabled=bool(cfg.get("enabled", True)),
        max_total_tokens=int(cfg.get("max_total_tokens", 128000)),
        reserved_response_tokens=int(cfg.get("reserved_response_tokens", 16000)),
        temperature=float(cfg.get("temperature", 1.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        tokenizer=cfg.get("tokenizer", "cl100k_base"),
    )


class ModelRegistry:
    """
    Registry of LLM models.

    It is config-driven, so adding a new LLM means adding one block to
    config/models.yaml, not changing RQ scripts.
    """

    def __init__(self, models_cfg: Dict[str, Any]):
        self._specs: Dict[str, ModelSpec] = {}

        raw_models = models_cfg.get("models", {})
        for key, cfg in raw_models.items():
            spec = model_spec_from_config(key, cfg)
            self._specs[key] = spec

    def get(self, key: str) -> ModelSpec:
        if key not in self._specs:
            known = ", ".join(sorted(self._specs.keys()))
            raise KeyError(f"Unknown model '{key}'. Known models: {known}")
        return self._specs[key]

    def enabled(self) -> Dict[str, ModelSpec]:
        return {
            key: spec
            for key, spec in self._specs.items()
            if spec.enabled
        }

    def keys(self) -> list[str]:
        return sorted(self._specs.keys())

    def enabled_keys(self) -> list[str]:
        return sorted(self.enabled().keys())
