#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from desc2fix.utils.config import resolve_path


def load_prompt(path: str | Path) -> str:
    path = resolve_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")

    return path.read_text(encoding="utf-8")


def render_prompt(template: str, **kwargs) -> str:
    rendered = template

    for key, value in kwargs.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))

    return rendered
