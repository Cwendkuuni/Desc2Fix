#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter
from pathlib import Path

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    path = Path("data/processed/defects4j/triplets.json")
    rows = load(path)

    print("Total:", len(rows))

    by_project = Counter(t.get("project") for t in rows)
    print("\nBy project:")
    for k, v in sorted(by_project.items()):
        print(f"  {k}: {v}")

    missing_report = []
    missing_src_patch = []
    missing_test_signal = []
    missing_test_patch = []

    for t in rows:
        iid = t.get("instance_id")
        if not t.get("bug_report", {}).get("raw_text", "").strip():
            missing_report.append(iid)

        if not t.get("patch", {}).get("src", {}).get("raw", "").strip():
            missing_src_patch.append(iid)

        tests = t.get("tests", {})
        test_patch = t.get("patch", {}).get("test")
        has_tests = bool(tests.get("trigger") or tests.get("relevant"))
        has_test_patch = bool(test_patch and test_patch.get("raw", "").strip())

        if not has_tests:
            missing_test_signal.append(iid)

        if not has_test_patch:
            missing_test_patch.append(iid)

    print("\nMissing report:", len(missing_report))
    print("Missing src patch:", len(missing_src_patch))
    print("Missing trigger/relevant tests:", len(missing_test_signal))
    print("Missing test patch:", len(missing_test_patch))

    print("\nFirst missing src patch:", missing_src_patch[:20])
    print("First missing tests:", missing_test_signal[:20])
    print("First missing test patch:", missing_test_patch[:20])

if __name__ == "__main__":
    main()
