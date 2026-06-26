#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


def project_root() -> Path:
    """
    Find the repository root robustly by walking upward from this file.

    A valid Desc2Fix root contains config/ and either src/, pyproject.toml,
    reproduce.sh, or README.md.
    """
    here = Path(__file__).resolve()

    for parent in here.parents:
        has_config = (parent / "config").is_dir()
        has_src = (parent / "src").is_dir()
        has_marker = any(
            (parent / marker).exists()
            for marker in ["pyproject.toml", "reproduce.sh", "README.md"]
        )

        if has_config and (has_src or has_marker):
            return parent

    raise RuntimeError(
        f"Could not locate Desc2Fix project root from {here}. "
        "Check that config/ exists in the repository root."
    )


def load_env(env_path: Optional[str | Path] = None) -> None:
    """
    Load environment variables from .env if available.
    """
    if env_path is None:
        env_path = project_root() / ".env"
    env_path = Path(env_path)

    if env_path.exists():
        load_dotenv(env_path)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file and return a dictionary.
    """
    path = Path(path)

    if not path.is_absolute():
        path = project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Missing YAML config file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def load_configs(
    *,
    paths_config: str | Path = "config/paths.yaml",
    datasets_config: str | Path = "config/datasets.yaml",
    models_config: str | Path = "config/models.yaml",
    experiments_config: str | Path = "config/experiments.yaml",
    metrics_config: str | Path = "config/metrics.yaml",
    prompts_config: str | Path = "config/prompts.yaml",
    dashboard_config: str | Path = "config/dashboard.yaml",
) -> Dict[str, Any]:
    """
    Load all standard Desc2Fix config files.

    Missing optional files are returned as empty dictionaries.
    """
    cfg: Dict[str, Any] = {}

    required = {
        "paths": paths_config,
        "datasets": datasets_config,
        "models": models_config,
        "experiments": experiments_config,
    }

    optional = {
        "metrics": metrics_config,
        "prompts": prompts_config,
        "dashboard": dashboard_config,
    }

    for key, path in required.items():
        cfg[key] = load_yaml(path)

    for key, path in optional.items():
        try:
            cfg[key] = load_yaml(path)
        except FileNotFoundError:
            cfg[key] = {}

    return cfg


def resolve_path(path: str | Path, *, base: Optional[str | Path] = None) -> Path:
    """
    Resolve a path relative to the project root or a custom base.
    """
    p = Path(path).expanduser()

    if p.is_absolute():
        return p

    if base is None:
        base = project_root()

    return Path(base).expanduser().resolve() / p


def get_required_env(var_name: str) -> str:
    """
    Get a required environment variable.
    """
    value = os.getenv(var_name)

    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {var_name}. "
            f"Set it in your shell or in .env."
        )

    return value


def get_optional_env(var_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an optional environment variable.
    """
    return os.getenv(var_name, default)


def enabled_models(models_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return models marked enabled=true in config/models.yaml.
    """
    models = models_cfg.get("models", {})
    return {
        name: cfg
        for name, cfg in models.items()
        if bool(cfg.get("enabled", True))
    }


def get_model_config(models_cfg: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """
    Return one model configuration.
    """
    models = models_cfg.get("models", {})

    if model_name not in models:
        known = ", ".join(sorted(models.keys()))
        raise KeyError(f"Unknown model '{model_name}'. Known models: {known}")

    return models[model_name]


def get_dataset_config(datasets_cfg: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Return one dataset configuration.
    """
    datasets = datasets_cfg.get("datasets", {})

    if dataset_name not in datasets:
        known = ", ".join(sorted(datasets.keys()))
        raise KeyError(f"Unknown dataset '{dataset_name}'. Known datasets: {known}")

    return datasets[dataset_name]
