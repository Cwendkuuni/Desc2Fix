#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from desc2fix.evaluation.artifacts import (
    artifact_pair_for_judge,
    prompt_path_for_rq_scenario,
)
from desc2fix.evaluation.judge_schema import normalize_judge_output
from desc2fix.extraction.io import signal_output_path
from desc2fix.llm.clients import make_llm_client
from desc2fix.llm.json_parser import parse_json_response
from desc2fix.llm.prompts import load_prompt
from desc2fix.llm.registry import ModelRegistry
from desc2fix.rq3_report_patch.metric_alignment import PATCH_VIEWS
from desc2fix.utils.config import get_dataset_config, load_configs, load_env, resolve_path
from desc2fix.utils.io import ensure_dir, read_json, write_json


RQS = ["rq2", "rq3", "rq4_internal"]
JUDGE_RESERVED_RESPONSE_TOKENS = 4000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM judges using the original Desc2Fix prompts."
    )

    parser.add_argument("--rq", choices=["rq2", "rq3", "rq4_internal", "all"], default="all")
    parser.add_argument("--dataset", choices=["defects4j", "swtbench", "all"], default="all")
    parser.add_argument("--model", default="all")
    parser.add_argument("--scenario", choices=["raw", "structured", "both"], default="both")
    parser.add_argument("--patch-view", choices=["full", "add", "remove", "all"], default="full")

    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--sample-from-human", default=None, help="Optional Human_Eval root. If provided, run only instances present in human annotations for the requested RQ/dataset/scenario/patch-view.")

    return parser.parse_args()


def render_legacy_prompt(template: str, variables: Dict[str, str]) -> str:
    """
    Original prompts use Python .format placeholders:
      {BUG_REPORT_RAW}, {TEST_CODE}, etc.

    JSON braces are escaped as {{ }} in prompt files.
    """
    return template.format(**variables)


def trim_prompt(prompt: str, *, model_key: str, model_name: str, max_total_tokens: int) -> str:
    """
    Reproduce old behavior:
      GPT-4o: tiktoken.encoding_for_model("gpt-4o")
      DeepSeek: cl100k_base
      reserved response tokens: 4000
    """
    max_prompt_tokens = max_total_tokens - JUDGE_RESERVED_RESPONSE_TOKENS

    try:
        import tiktoken

        if model_key == "gpt4o" or "gpt-4o" in model_name:
            encoding = tiktoken.encoding_for_model("gpt-4o")
        else:
            encoding = tiktoken.get_encoding("cl100k_base")

        toks = encoding.encode(prompt)
        if len(toks) > max_prompt_tokens:
            return encoding.decode(toks[:max_prompt_tokens])

    except Exception:
        # Fallback approximate char trimming
        max_chars = max_prompt_tokens * 4
        if len(prompt) > max_chars:
            return prompt[:max_chars]

    return prompt


def get_structured_root(cfg: Dict) -> Path:
    data_cfg = (cfg.get("paths", {}) or {}).get("data", {}) or {}
    if isinstance(data_cfg, dict) and data_cfg.get("structured"):
        return resolve_path(data_cfg["structured"])
    return resolve_path("data/structured")


def load_triplets(dataset: str, cfg: Dict, limit: Optional[int]) -> List[Dict]:
    dcfg = get_dataset_config(cfg["datasets"], dataset)
    path = resolve_path(dcfg["triplets_path"])

    rows = read_json(path)
    rows = rows.get("rows", rows) if isinstance(rows, dict) else rows

    if not isinstance(rows, list):
        raise ValueError(f"Expected list of triplets in {path}")

    if limit:
        rows = rows[:limit]

    return rows


def normalize_dataset_for_human(value: str) -> str:
    s = str(value or "").strip().lower()

    if s in {"defects4j", "defects4j"} or "defects" in s:
        return "Defects4J"

    if s in {"swtbench", "swt-bench", "swebench", "swe-bench"} or "swt" in s or "swe" in s:
        return "SWTBench"

    return str(value or "").strip()


def human_dataset_name(dataset: str) -> str:
    if dataset == "defects4j":
        return "Defects4J"
    if dataset == "swtbench":
        return "SWTBench"
    return dataset


def human_rq_dir_name(rq: str) -> str:
    if rq == "rq2":
        return "RQ2"
    if rq == "rq3":
        return "RQ3"
    if rq == "rq4_internal":
        return "RQ4"
    raise ValueError(rq)


def triplet_pair_id(dataset: str, triplet: Dict) -> str:
    if dataset == "defects4j":
        project = triplet.get("project")
        bug_id = triplet.get("bug_id")

        if not bug_id:
            instance = str(triplet.get("instance_id") or "")
            if "-" in instance:
                project, bug_id = instance.split("-", 1)

        return f"{project}__{bug_id}"

    return str(triplet.get("instance_id"))


def parse_human_scenario_label(value: str):
    s = str(value or "").strip()
    up = s.upper()
    low = s.lower()

    if up.startswith("STRUCTURED"):
        scenario_base = "structured"
    elif up.startswith("RAW"):
        scenario_base = "raw"
    else:
        scenario_base = low

    patch_view = None
    for view in ["full", "add", "remove"]:
        if low.endswith(f"_{view}") or low == view:
            patch_view = view
            break

    return scenario_base, patch_view


def read_human_jsonl_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def human_pair_id_from_record(obj: Dict) -> str:
    dataset = normalize_dataset_for_human(obj.get("dataset"))

    if dataset == "Defects4J":
        return f"{obj.get('project')}__{obj.get('bug_id')}"

    return str(obj.get("instance_id"))


def human_target_pair_ids(
    *,
    human_root: Path,
    rq: str,
    dataset: str,
    scenario: str,
    patch_view: Optional[str],
) -> set[str]:
    """
    Extract the exact annotated pair_ids for a given rq/dataset/scenario/patch_view.
    Uses Human_Eval/RQ2,RQ3,RQ4 jsonl files.
    """
    target = set()

    rq_dir = human_root / human_rq_dir_name(rq)
    if not rq_dir.exists():
        return target

    dataset_h = human_dataset_name(dataset)

    for path in sorted(rq_dir.glob("*human*_alignment_*.jsonl")):
        for obj in read_human_jsonl_records(path):
            if normalize_dataset_for_human(obj.get("dataset")) != dataset_h:
                continue

            scenario_base, scenario_patch_view = parse_human_scenario_label(obj.get("scenario"))

            if scenario_base != scenario:
                continue

            if rq in {"rq3", "rq4_internal"}:
                if (scenario_patch_view or "none") != (patch_view or "none"):
                    continue

            target.add(human_pair_id_from_record(obj))

    return target


def filter_triplets_from_human_sample(
    *,
    triplets: List[Dict],
    human_root: Optional[str],
    rq: str,
    dataset: str,
    scenario: str,
    patch_view: Optional[str],
) -> List[Dict]:
    if not human_root:
        return triplets

    root = resolve_path(human_root)
    targets = human_target_pair_ids(
        human_root=root,
        rq=rq,
        dataset=dataset,
        scenario=scenario,
        patch_view=patch_view,
    )

    if not targets:
        print(
            f"[WARN] No human targets found for rq={rq}, dataset={dataset}, "
            f"scenario={scenario}, patch_view={patch_view} in {root}"
        )
        return []

    filtered = [
        t for t in triplets
        if triplet_pair_id(dataset, t) in targets
    ]

    print(
        f"[Desc2Fix][LLM-Judge] human sample filter: "
        f"targets={len(targets)}, matched_triplets={len(filtered)}"
    )

    missing = sorted(targets - {triplet_pair_id(dataset, t) for t in filtered})
    if missing:
        out_dir = resolve_path("results/experiments/llm_judges/debug")
        ensure_dir(out_dir)
        debug_path = out_dir / f"missing_human_targets_{rq}_{dataset}_{scenario}_{patch_view or 'none'}.txt"
        debug_path.write_text("\n".join(missing), encoding="utf-8")
        print(f"[WARN] missing human targets written to {debug_path}: {len(missing)}")

    return filtered

def load_signals(
    *,
    dataset: str,
    model: str,
    triplet: Dict,
    cfg: Dict,
) -> Optional[Dict]:
    structured_root = get_structured_root(cfg)

    path = signal_output_path(
        structured_root=structured_root,
        dataset=dataset,
        model=model,
        triplet=triplet,
    )

    if not path.exists():
        return None

    obj = read_json(path)
    return obj.get("signals", {})


def output_dir_for_run(
    *,
    rq: str,
    dataset: str,
    scenario: str,
    model: str,
    patch_view: Optional[str],
) -> Path:
    if rq == "rq2":
        return resolve_path(f"results/experiments/llm_judges/{rq}/{dataset}/{scenario}/{model}")

    if rq == "rq3":
        return resolve_path(
            f"results/experiments/llm_judges/{rq}/{dataset}/{scenario}/{model}/{patch_view}"
        )

    if rq == "rq4_internal":
        return resolve_path(
            f"results/experiments/llm_judges/{rq}/{dataset}/{scenario}/{model}/{patch_view}"
        )

    raise ValueError(f"Unknown RQ: {rq}")


def read_done_keys(jsonl_path: Path) -> set[str]:
    done = set()

    if not jsonl_path.exists():
        return done

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                key = make_done_key(
                    instance_id=str(obj.get("instance_id")),
                    scenario=str(obj.get("scenario")),
                    patch_view=obj.get("patch_view"),
                )
                done.add(key)
            except Exception:
                continue

    return done


def make_done_key(*, instance_id: str, scenario: str, patch_view: Optional[str]) -> str:
    return f"{instance_id}::{scenario}::{patch_view or 'none'}"


def append_jsonl(path: Path, obj: Dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_llm_json(raw_response: str) -> Dict:
    """
    Use existing parser first, then tolerate old triple-backtick style.
    """
    try:
        return parse_json_response(raw_response)
    except Exception:
        pass

    m = re.search(r"```(?:json)?\s*({.*?})\s*```", raw_response or "", flags=re.DOTALL)
    if not m:
        m = re.search(r"({.*})", raw_response or "", flags=re.DOTALL)

    if not m:
        raise ValueError("Could not parse JSON response.")

    js = re.sub(r",(\s*[}\]])", r"\1", m.group(1))
    return json.loads(js)


def should_skip_empty_patch(rq: str, patch_view: Optional[str], variables: Dict[str, str]) -> bool:
    if rq in {"rq3", "rq4_internal"}:
        patch_code = variables.get("PATCH_CODE", "") or ""
        return len(patch_code.strip()) < 10
    return False


def should_skip_empty_artifacts(rq: str, scenario: str, variables: Dict[str, str]) -> bool:
    if rq == "rq2":
        if scenario == "raw":
            return not variables.get("BUG_REPORT_RAW", "").strip() or not variables.get("TEST_CODE", "").strip()
        return not variables.get("BUG_REPORT_STRUCTURED_JSON", "").strip() or not variables.get("TEST_CODE", "").strip()

    if rq == "rq3":
        if scenario == "raw":
            return not variables.get("BUG_REPORT_RAW", "").strip() or not variables.get("PATCH_CODE", "").strip()
        return not variables.get("BUG_REPORT_STRUCTURED_JSON", "").strip() or not variables.get("PATCH_CODE", "").strip()

    if rq == "rq4_internal":
        if scenario == "raw":
            return not variables.get("TEST_CODE", "").strip() or not variables.get("PATCH_CODE", "").strip()
        return (
            not variables.get("BUG_REPORT_STRUCTURED_JSON", "").strip()
            or not variables.get("TEST_CODE", "").strip()
            or not variables.get("PATCH_CODE", "").strip()
        )

    return True


def run_one_configuration(
    *,
    rq: str,
    dataset: str,
    scenario: str,
    model_key: str,
    patch_view: Optional[str],
    cfg: Dict,
    args: argparse.Namespace,
) -> None:
    registry = ModelRegistry(cfg["models"])
    spec = registry.get(model_key)

    if not spec.enabled:
        print(f"[SKIP] disabled model: {model_key}")
        return

    client = make_llm_client(spec, retry=True, max_retries=args.max_retries)
    #triplets = load_triplets(dataset, cfg, args.limit)
    triplets = load_triplets(dataset, cfg, args.limit)

    triplets = filter_triplets_from_human_sample(
        triplets=triplets,
        human_root=args.sample_from_human,
        rq=rq,
        dataset=dataset,
        scenario=scenario,
        patch_view=patch_view,
    )

    prompt_path = prompt_path_for_rq_scenario(rq, scenario)
    template = load_prompt(prompt_path)

    out_dir = output_dir_for_run(
        rq=rq,
        dataset=dataset,
        scenario=scenario,
        model=model_key,
        patch_view=patch_view,
    )

    scores_path = out_dir / "scores.jsonl"
    errors_path = out_dir / "errors.jsonl"

    done_keys = read_done_keys(scores_path) if args.resume else set()

    written = 0
    skipped = 0
    errors = 0

    print(
        f"[Desc2Fix][LLM-Judge] rq={rq}, dataset={dataset}, "
        f"scenario={scenario}, model={model_key}, patch_view={patch_view}, "
        f"instances={len(triplets)}"
    )

    for triplet in tqdm(
        triplets,
        desc=f"judge/{rq}/{dataset}/{scenario}/{model_key}/{patch_view or 'none'}",
        ncols=100,
    ):
        instance_id = str(triplet.get("instance_id"))
        done_key = make_done_key(
            instance_id=instance_id,
            scenario=scenario,
            patch_view=patch_view,
        )

        if args.resume and done_key in done_keys:
            skipped += 1
            continue

        try:
            signals = None
            if scenario == "structured":
                signals = load_signals(
                    dataset=dataset,
                    model=model_key,
                    triplet=triplet,
                    cfg=cfg,
                )
                if signals is None:
                    raise FileNotFoundError(
                        f"Missing structured signals for {dataset}/{model_key}/{instance_id}"
                    )

            variables, artifact_pair = artifact_pair_for_judge(
                rq=rq,
                triplet=triplet,
                scenario=scenario,
                patch_view=patch_view or "full",
                signals=signals,
            )

            if should_skip_empty_patch(rq, patch_view, variables):
                skipped += 1
                continue

            if should_skip_empty_artifacts(rq, scenario, variables):
                skipped += 1
                continue

            prompt = render_legacy_prompt(template, variables)
            prompt = trim_prompt(
                prompt,
                model_key=model_key,
                model_name=spec.model_name,
                max_total_tokens=spec.max_total_tokens,
            )

            raw_response = client.generate(
                prompt=prompt,
                max_tokens=JUDGE_RESERVED_RESPONSE_TOKENS,
                temperature=spec.temperature,
                top_p=spec.top_p,
            )

            parsed = parse_llm_json(raw_response)
            scores = normalize_judge_output(parsed)

            if rq == "rq3" and patch_view:
                legacy_scenario = f"{scenario.upper()}_{patch_view}"
            elif rq == "rq4_internal" and patch_view:
                legacy_scenario = f"{scenario.upper()}_{patch_view}"
            else:
                legacy_scenario = scenario.upper()

            payload = {
                "rq": rq,
                "dataset": dataset,
                "artifact_pair": artifact_pair,
                "instance_id": instance_id,
                "project": triplet.get("project"),
                "repo": triplet.get("repo"),
                "scenario": legacy_scenario,
                "scenario_base": scenario,
                "model": model_key,
                "model_name": spec.model_name,
                "patch_view": patch_view,
                "alignment_score": scores["alignment_score"],
                "coverage": scores["coverage"],
                "correctness": scores["correctness"],
                "specificity": scores["specificity"],
                "justification": scores["justification"],
                "raw_response": raw_response,
                "prompt_path": prompt_path,
            }

            append_jsonl(scores_path, payload)
            written += 1

        except Exception as exc:
            errors += 1
            append_jsonl(
                errors_path,
                {
                    "rq": rq,
                    "dataset": dataset,
                    "instance_id": instance_id,
                    "scenario": scenario,
                    "model": model_key,
                    "patch_view": patch_view,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"[ERR] {rq}/{dataset}/{model_key}/{instance_id}: {exc}")

    summary = {
        "rq": rq,
        "dataset": dataset,
        "scenario": scenario,
        "model": model_key,
        "patch_view": patch_view,
        "instances": len(triplets),
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "scores_path": str(scores_path),
        "errors_path": str(errors_path),
        "prompt_path": prompt_path,
        "reserved_response_tokens": JUDGE_RESERVED_RESPONSE_TOKENS,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
    }

    write_json(out_dir / "summary.json", summary)

    print(
        f"[OK][LLM-Judge] {rq}/{dataset}/{scenario}/{model_key}/{patch_view}: "
        f"written={written}, skipped={skipped}, errors={errors}"
    )


def main() -> None:
    args = parse_args()

    if args.all:
        args.rq = "all"
        args.dataset = "all"
        args.model = "all"
        args.scenario = "both"
        args.patch_view = "all"

    load_env()
    cfg = load_configs()

    rqs = RQS if args.rq == "all" else [args.rq]
    datasets = ["defects4j", "swtbench"] if args.dataset == "all" else [args.dataset]

    registry = ModelRegistry(cfg["models"])
    models = registry.enabled_keys() if args.model == "all" else [args.model]

    for rq in rqs:
        for dataset in datasets:
            for model_key in models:
                scenarios = ["raw", "structured"] if args.scenario == "both" else [args.scenario]

                if rq in {"rq3", "rq4_internal"}:
                    patch_views = PATCH_VIEWS if args.patch_view == "all" else [args.patch_view]
                else:
                    patch_views = [None]

                for scenario in scenarios:
                    for patch_view in patch_views:
                        run_one_configuration(
                            rq=rq,
                            dataset=dataset,
                            scenario=scenario,
                            model_key=model_key,
                            patch_view=patch_view,
                            cfg=cfg,
                            args=args,
                        )


if __name__ == "__main__":
    main()
