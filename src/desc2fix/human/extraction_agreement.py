#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


SIGNAL_FIELDS = [
    "exception",
    "api_involved",
    "input_conditions",
    "expected",
    "actual",
    "reproduction_steps",
]


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_dataset(value: object) -> str:
    s = str(value or "").strip().lower()

    if "defects" in s or s == "d4j":
        return "Defects4J"

    if "swt" in s or "swe" in s:
        return "SWTBench"

    return str(value or "").strip()


def infer_dataset_from_path(path: Path) -> str:
    s = str(path).lower()

    if "defects4j" in s:
        return "Defects4J"

    if "swtbench" in s or "swt-bench" in s:
        return "SWTBench"

    return ""


def infer_rater_from_path(path: Path) -> str:
    name = path.name.lower()

    m = re.search(r"human[_-]?(\d+)", name)
    if m:
        return f"human{m.group(1)}"

    if "human1" in name:
        return "human1"

    if "human2" in name:
        return "human2"

    if "human" in name:
        return "human"

    return "unknown"


def pair_id_from_record(obj: Dict, dataset: Optional[str] = None) -> str:
    dataset = normalize_dataset(dataset or obj.get("dataset") or infer_dataset_from_path(Path("")))

    if dataset == "Defects4J":
        project = obj.get("project")
        bug_id = obj.get("bug_id")

        if not bug_id and obj.get("instance_id"):
            instance = str(obj.get("instance_id"))
            if "-" in instance:
                project, bug_id = instance.split("-", 1)

        return f"{project}__{bug_id}"

    return str(obj.get("instance_id"))


def normalize_scalar(value: object) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    if isinstance(value, list):
        return " ".join(normalize_scalar(v) for v in value if normalize_scalar(v))

    if isinstance(value, dict):
        return " ".join(
            f"{normalize_scalar(k)} {normalize_scalar(v)}"
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        )

    s = str(value).strip()

    if s.lower() in {"none", "null", "nan", "n/a", "na", ""}:
        return ""

    return s


def normalize_list(value: object) -> List[str]:
    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    if isinstance(value, list):
        out = []
        for v in value:
            s = normalize_scalar(v)
            if s:
                out.append(s)
        return out

    if isinstance(value, dict):
        s = normalize_scalar(value)
        return [s] if s else []

    s = normalize_scalar(value)
    return [s] if s else []


def tokenize(value: object) -> set[str]:
    s = normalize_scalar(value).lower()
    if not s:
        return set()

    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|[^\s]", s))


def token_jaccard(a: object, b: object) -> float:
    ta = tokenize(a)
    tb = tokenize(b)

    if not ta and not tb:
        return 1.0

    if not ta or not tb:
        return 0.0

    return float(len(ta & tb) / len(ta | tb))


def list_jaccard(a: object, b: object) -> float:
    la = {normalize_scalar(x).lower() for x in normalize_list(a)}
    lb = {normalize_scalar(x).lower() for x in normalize_list(b)}

    la = {x for x in la if x}
    lb = {x for x in lb if x}

    if not la and not lb:
        return 1.0

    if not la or not lb:
        return 0.0

    return float(len(la & lb) / len(la | lb))


def is_present(value: object) -> bool:
    if isinstance(value, list):
        return any(is_present(v) for v in value)

    if isinstance(value, dict):
        return bool(value)

    return bool(normalize_scalar(value))


def signal_value(signals: Dict, field: str):
    if not isinstance(signals, dict):
        return None

    return signals.get(field)


def extract_signals_payload(obj: Dict) -> Dict:
    if isinstance(obj.get("extracted_signals"), dict):
        return obj["extracted_signals"]

    if isinstance(obj.get("signals"), dict):
        return obj["signals"]

    # Some old files may store the six fields at top level.
    if any(field in obj for field in SIGNAL_FIELDS):
        return {field: obj.get(field) for field in SIGNAL_FIELDS}

    return {field: None for field in SIGNAL_FIELDS}


def normalize_signals(signals: Dict) -> Dict:
    out = {}

    for field in SIGNAL_FIELDS:
        value = signal_value(signals, field)

        if field in {"api_involved", "input_conditions", "reproduction_steps"}:
            out[field] = normalize_list(value)
        else:
            s = normalize_scalar(value)
            out[field] = s if s else None

    return out


def signal_count(signals: Dict) -> int:
    return sum(1 for field in SIGNAL_FIELDS if is_present(signal_value(signals, field)))


def field_similarity(field: str, a: object, b: object) -> float:
    if field in {"api_involved", "input_conditions", "reproduction_steps"}:
        return list_jaccard(a, b)

    return token_jaccard(a, b)


def compare_signal_dicts(
    *,
    dataset: str,
    pair_id: str,
    left_source: str,
    right_source: str,
    left_signals: Dict,
    right_signals: Dict,
) -> List[Dict]:
    rows = []

    left_signals = normalize_signals(left_signals)
    right_signals = normalize_signals(right_signals)

    similarities = []
    presence_agreements = []

    for field in SIGNAL_FIELDS:
        left_value = left_signals.get(field)
        right_value = right_signals.get(field)

        left_present = is_present(left_value)
        right_present = is_present(right_value)

        presence_agreement = int(left_present == right_present)
        sim = field_similarity(field, left_value, right_value)

        similarities.append(sim)
        presence_agreements.append(presence_agreement)

        rows.append(
            {
                "dataset": dataset,
                "pair_id": pair_id,
                "left_source": left_source,
                "right_source": right_source,
                "level": "field",
                "field": field,
                "left_present": left_present,
                "right_present": right_present,
                "presence_agreement": presence_agreement,
                "similarity": sim,
                "left_signal_count": signal_count(left_signals),
                "right_signal_count": signal_count(right_signals),
                "signal_count_diff": abs(signal_count(left_signals) - signal_count(right_signals)),
            }
        )

    rows.append(
        {
            "dataset": dataset,
            "pair_id": pair_id,
            "left_source": left_source,
            "right_source": right_source,
            "level": "macro",
            "field": "ALL",
            "left_present": np.nan,
            "right_present": np.nan,
            "presence_agreement": float(np.mean(presence_agreements)),
            "similarity": float(np.mean(similarities)),
            "left_signal_count": signal_count(left_signals),
            "right_signal_count": signal_count(right_signals),
            "signal_count_diff": abs(signal_count(left_signals) - signal_count(right_signals)),
        }
    )

    return rows


def load_human_extractions(human_root: Path) -> pd.DataFrame:
    rows = []

    rq1_dir = human_root / "RQ1"

    if not rq1_dir.exists():
        return pd.DataFrame(rows)

    files = sorted(rq1_dir.glob("*human*_extracted_signals.jsonl"))

    for path in files:
        dataset = infer_dataset_from_path(path)
        rater = infer_rater_from_path(path)

        for obj in read_jsonl(path):
            dataset_obj = normalize_dataset(obj.get("dataset") or dataset)
            pair_id = pair_id_from_record(obj, dataset_obj)

            rows.append(
                {
                    "dataset": dataset_obj,
                    "pair_id": pair_id,
                    "project": obj.get("project"),
                    "bug_id": obj.get("bug_id"),
                    "repo": obj.get("repo"),
                    "instance_id": obj.get("instance_id"),
                    "source": rater,
                    "source_type": "human",
                    "source_file": str(path),
                    "signals": normalize_signals(extract_signals_payload(obj)),
                }
            )

    return pd.DataFrame(rows)


def load_llm_extractions(
    *,
    structured_root: Path,
    datasets: List[str],
    models: List[str],
) -> pd.DataFrame:
    rows = []

    for dataset_key in datasets:
        dataset_dir = structured_root / dataset_key
        if not dataset_dir.exists():
            continue

        for model in models:
            model_dir = dataset_dir / model
            if not model_dir.exists():
                continue

            for path in sorted(model_dir.rglob("*.json")):
                try:
                    obj = read_json(path)
                except Exception:
                    continue

                dataset_obj = normalize_dataset(obj.get("dataset") or dataset_key)
                pair_id = pair_id_from_record(obj, dataset_obj)

                if not pair_id or pair_id in {"None__None", "None"}:
                    continue

                rows.append(
                    {
                        "dataset": dataset_obj,
                        "pair_id": pair_id,
                        "project": obj.get("project"),
                        "bug_id": obj.get("bug_id"),
                        "repo": obj.get("repo"),
                        "instance_id": obj.get("instance_id"),
                        "source": model,
                        "source_type": "llm",
                        "source_file": str(path),
                        "signals": normalize_signals(extract_signals_payload(obj)),
                    }
                )

    return pd.DataFrame(rows)


def deduplicate_extractions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    before = len(df)
    df = df.drop_duplicates(
        subset=["dataset", "pair_id", "source", "source_type"],
        keep="last",
    ).reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"[Desc2Fix][RQ1] deduplicated extraction rows: {before} -> {after}")

    return df


def pairwise_human_agreement(human_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if human_df.empty:
        return pd.DataFrame(rows)

    for (dataset, pair_id), g in human_df.groupby(["dataset", "pair_id"], dropna=False):
        sources = sorted(g["source"].dropna().unique())

        if len(sources) < 2:
            continue

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                s1 = sources[i]
                s2 = sources[j]

                a = g[g["source"] == s1].iloc[-1]
                b = g[g["source"] == s2].iloc[-1]

                rows.extend(
                    compare_signal_dicts(
                        dataset=dataset,
                        pair_id=pair_id,
                        left_source=s1,
                        right_source=s2,
                        left_signals=a["signals"],
                        right_signals=b["signals"],
                    )
                )

    return pd.DataFrame(rows)


def llm_vs_human_agreement(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    *,
    include_human_avg: bool = True,
) -> pd.DataFrame:
    rows = []

    if human_df.empty or llm_df.empty:
        return pd.DataFrame(rows)

    human_sources = sorted(human_df["source"].dropna().unique())

    for (dataset, pair_id), hgroup in human_df.groupby(["dataset", "pair_id"], dropna=False):
        lgroup = llm_df[(llm_df["dataset"] == dataset) & (llm_df["pair_id"] == pair_id)]

        if lgroup.empty:
            continue

        for _, lrow in lgroup.iterrows():
            model = lrow["source"]

            # LLM vs each individual human.
            for human_source in human_sources:
                hs = hgroup[hgroup["source"] == human_source]

                if hs.empty:
                    continue

                hrow = hs.iloc[-1]

                rows.extend(
                    compare_signal_dicts(
                        dataset=dataset,
                        pair_id=pair_id,
                        left_source=model,
                        right_source=human_source,
                        left_signals=lrow["signals"],
                        right_signals=hrow["signals"],
                    )
                )

            # LLM vs HumanAvg = mean of similarities against available human raters.
            if include_human_avg:
                tmp_rows = []

                for _, hrow in hgroup.iterrows():
                    tmp_rows.extend(
                        compare_signal_dicts(
                            dataset=dataset,
                            pair_id=pair_id,
                            left_source=model,
                            right_source=hrow["source"],
                            left_signals=lrow["signals"],
                            right_signals=hrow["signals"],
                        )
                    )

                if tmp_rows:
                    tmp = pd.DataFrame(tmp_rows)

                    for field, fg in tmp.groupby(["level", "field"], dropna=False):
                        level, field_name = field

                        rows.append(
                            {
                                "dataset": dataset,
                                "pair_id": pair_id,
                                "left_source": model,
                                "right_source": "HumanAvg",
                                "level": level,
                                "field": field_name,
                                "left_present": np.nan,
                                "right_present": np.nan,
                                "presence_agreement": fg["presence_agreement"].mean(),
                                "similarity": fg["similarity"].mean(),
                                "left_signal_count": fg["left_signal_count"].iloc[0],
                                "right_signal_count": fg["right_signal_count"].mean(),
                                "signal_count_diff": fg["signal_count_diff"].mean(),
                            }
                        )

    return pd.DataFrame(rows)


def llm_pairwise_agreement(llm_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if llm_df.empty:
        return pd.DataFrame(rows)

    for (dataset, pair_id), g in llm_df.groupby(["dataset", "pair_id"], dropna=False):
        sources = sorted(g["source"].dropna().unique())

        if len(sources) < 2:
            continue

        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                s1 = sources[i]
                s2 = sources[j]

                a = g[g["source"] == s1].iloc[-1]
                b = g[g["source"] == s2].iloc[-1]

                rows.extend(
                    compare_signal_dicts(
                        dataset=dataset,
                        pair_id=pair_id,
                        left_source=s1,
                        right_source=s2,
                        left_signals=a["signals"],
                        right_signals=b["signals"],
                    )
                )

    return pd.DataFrame(rows)


def summarize_agreement(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = [
        "dataset",
        "left_source",
        "right_source",
        "level",
        "field",
    ]

    rows = []

    for key, g in df.groupby(group_cols, dropna=False):
        dataset, left_source, right_source, level, field = key

        rows.append(
            {
                "dataset": dataset,
                "left_source": left_source,
                "right_source": right_source,
                "level": level,
                "field": field,
                "n": int(g["pair_id"].nunique()),
                "presence_agreement_mean": g["presence_agreement"].mean(),
                "presence_agreement_std": g["presence_agreement"].std(),
                "similarity_mean": g["similarity"].mean(),
                "similarity_std": g["similarity"].std(),
                "left_signal_count_mean": g["left_signal_count"].mean(),
                "right_signal_count_mean": g["right_signal_count"].mean(),
                "signal_count_diff_mean": g["signal_count_diff"].mean(),
            }
        )

    return pd.DataFrame(rows)


def unmatched_debug(human_df: pd.DataFrame, llm_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if human_df.empty:
        human_keys = pd.DataFrame(columns=["dataset", "pair_id"])
    else:
        human_keys = human_df[["dataset", "pair_id"]].drop_duplicates()

    if llm_df.empty:
        llm_keys = pd.DataFrame(columns=["dataset", "pair_id"])
    else:
        llm_keys = llm_df[["dataset", "pair_id"]].drop_duplicates()

    h_without_l = human_keys.merge(llm_keys, on=["dataset", "pair_id"], how="left", indicator=True)
    h_without_l = h_without_l[h_without_l["_merge"] == "left_only"].drop(columns=["_merge"])

    l_without_h = llm_keys.merge(human_keys, on=["dataset", "pair_id"], how="left", indicator=True)
    l_without_h = l_without_h[l_without_h["_merge"] == "left_only"].drop(columns=["_merge"])

    return h_without_l, l_without_h


def serialize_signals_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "signals" not in df.columns:
        return df

    out = df.copy()
    out["signals"] = out["signals"].apply(
        lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True)
    )
    return out
