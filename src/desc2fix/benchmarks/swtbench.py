#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json


DEFAULT_HF_DATASET = "eth-sri/SWT-bench_bm25_27k_zsp"


@dataclass
class SWTBenchStatus:
    dataset_name: str
    cache_dir: Path
    available: bool
    dev_rows: Optional[int] = None
    test_rows: Optional[int] = None
    error: Optional[str] = None

    @property
    def ready_for_corpus_build(self) -> bool:
        return self.available

    def as_dict(self) -> dict:
        return {
            "dataset_name": self.dataset_name,
            "cache_dir": str(self.cache_dir),
            "available": self.available,
            "dev_rows": self.dev_rows,
            "test_rows": self.test_rows,
            "error": self.error,
            "ready_for_corpus_build": self.ready_for_corpus_build,
        }


def setup_swtbench(
    *,
    dataset_name: str = DEFAULT_HF_DATASET,
    cache_dir: Path = Path("data/cache/huggingface"),
    download: bool = True,
) -> SWTBenchStatus:
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not download:
        status = SWTBenchStatus(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            available=True,
        )
        print("[Desc2Fix] SWT-Bench download skipped.")
        return status

    try:
        from datasets import load_dataset

        dev = load_dataset(dataset_name, split="dev", cache_dir=str(cache_dir))
        test = load_dataset(dataset_name, split="test", cache_dir=str(cache_dir))

        status = SWTBenchStatus(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            available=True,
            dev_rows=len(dev),
            test_rows=len(test),
        )

        manifest = {
            "dataset_name": dataset_name,
            "cache_dir": str(cache_dir),
            "dev_rows": len(dev),
            "test_rows": len(test),
        }

        manifest_path = cache_dir / "swtbench_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    except Exception as e:
        status = SWTBenchStatus(
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            available=False,
            error=str(e),
        )

    print("[Desc2Fix] SWT-Bench status:")
    for k, v in status.as_dict().items():
        print(f"  {k}: {v}")

    return status
