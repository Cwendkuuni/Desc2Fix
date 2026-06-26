#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def read_json(path: str | Path) -> Any:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e

            if isinstance(obj, dict):
                rows.append(obj)
            else:
                rows.append({"value": obj})

    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv_rows(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    ensure_dir(path.parent)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path):
    """
    Optional pandas helper.

    Pandas is imported lazily so JSON-only pipelines do not fail when pandas
    is not installed yet.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "pandas is required for read_csv(). Install with: "
            "python -m pip install pandas"
        ) from exc

    return pd.read_csv(path)


def write_csv(path: str | Path, df) -> None:
    """
    Optional pandas helper.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pandas is required for write_csv(). Install with: "
            "python -m pip install pandas"
        ) from exc

    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def append_csv_row(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    file_exists = path.exists()
    fieldnames = list(row.keys())

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def list_files(
    root: str | Path,
    *,
    pattern: str = "*",
    recursive: bool = True,
) -> List[Path]:
    root = Path(root)

    if not root.exists():
        return []

    if recursive:
        return sorted(root.rglob(pattern))

    return sorted(root.glob(pattern))


def safe_filename(value: str, *, replacement: str = "_") -> str:
    bad_chars = ["/", "\\", ":", "*", "?", '"', "<", ">", "|", " "]
    out = str(value)

    for ch in bad_chars:
        out = out.replace(ch, replacement)

    while replacement * 2 in out:
        out = out.replace(replacement * 2, replacement)

    return out.strip(replacement)


def load_rows_from_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)

    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)

    obj = read_json(path)

    if isinstance(obj, dict) and isinstance(obj.get("rows"), list):
        rows = obj["rows"]
    elif isinstance(obj, list):
        rows = obj
    else:
        rows = [obj]

    clean_rows: List[Dict[str, Any]] = []
    for row in rows:
        if isinstance(row, dict):
            clean_rows.append(row)
        else:
            clean_rows.append({"value": row})

    return clean_rows