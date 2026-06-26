# Results Guide

This document explains where to find the main Desc2Fix human-evaluation results and how to interpret the generated tables.

It complements:

```text
docs/HUMAN_EVALUATION.md
docs/REPRODUCTION_WORKFLOW.md
```

---

## 1. Main result directory

The final consolidated human-evaluation tables are stored in:

```text
results/tables/human_evaluation/
```

The most important file is:

```text
results/tables/human_evaluation/global_human_evaluation_summary.csv
```

This file combines the human-evaluation outputs for:

```text
RQ1 = Structured bug signal extraction
RQ2 = Bug report ↔ test alignment
RQ3 = Bug report ↔ patch alignment
RQ4 internal = Test ↔ patch alignment
```

---

## 2. Global summary table

### File

```text
results/tables/human_evaluation/global_human_evaluation_summary.csv
```

### Purpose

This is the main table to use for reporting, README summaries, dashboards, and paper-level result extraction.

It contains one normalized schema for all RQs.

### Expected row counts

```text
RQ1:          12 rows
RQ2:          96 rows
RQ3:         288 rows
RQ4 internal:144 rows
Total:       540 rows
```

### Validation command

```bash
python - <<'PY'
import pandas as pd

base = "results/tables/human_evaluation"
g = pd.read_csv(f"{base}/global_human_evaluation_summary.csv")

print(g.groupby(["rq", "rq_label"]).size())
print(g.head(20))
PY
```

Expected output:

```text
rq            rq_label               
rq1           RQ1_Extraction              12
rq2           RQ2_Report_Test             96
rq3           RQ3_Report_Patch           288
rq4_internal  RQ4_Test_Patch_Internal    144
dtype: int64
```

---

## 3. Global summary columns

The global table uses the following columns:

```text
rq
rq_label
dataset
representation
patch_view
model
human
metric
n
primary_value
secondary_value
primary_value_name
secondary_value_name
```

### Column meaning

| Column | Meaning |
|---|---|
| `rq` | Internal RQ key, e.g., `rq1`, `rq2`, `rq3`, `rq4_internal` |
| `rq_label` | Human-readable RQ label |
| `dataset` | Dataset name, e.g., `Defects4J`, `SWTBench` |
| `representation` | `RAW`, `STRUCTURED`, or `STRUCTURED_EXTRACTION` |
| `patch_view` | `none`, `full`, `add`, or `remove` |
| `model` | LLM model, e.g., `gpt4o`, `deepseek` |
| `human` | Human reference, e.g., `human1`, `human2`, `HumanAvg` |
| `metric` | Compared metric or score |
| `n` | Number of evaluated instances |
| `primary_value` | Main reported value |
| `secondary_value` | Secondary reported value |
| `primary_value_name` | Name of the primary value |
| `secondary_value_name` | Name of the secondary value |

---

## 4. Interpreting values by RQ

### RQ1

For RQ1, the global table stores:

```text
primary_value_name   = similarity_mean
secondary_value_name = presence_agreement_mean
```

So:

```text
primary_value   = macro signal similarity
secondary_value = macro presence agreement
```

RQ1 rows compare extracted structured bug signals against human annotations.

Example interpretation:

```text
model = gpt4o
human = HumanAvg
metric = macro_signal_similarity
primary_value = similarity_mean
secondary_value = presence_agreement_mean
```

This means: GPT-4o is compared with the average human reference on structured signal extraction.

### RQ2/RQ3/RQ4 internal

For RQ2, RQ3, and RQ4 internal, the global table stores:

```text
primary_value_name   = spearman
secondary_value_name = mae
```

So:

```text
primary_value   = Spearman correlation between LLM scores and human scores
secondary_value = mean absolute error between LLM scores and human scores
```

These rows compare LLM judge scores against human annotations for:

```text
alignment_score
coverage
correctness
specificity
```

---

## 5. RQ-specific final tables

### RQ1

Main RQ1 files:

```text
results/tables/human_evaluation/rq1_human_pairwise_summary.csv
results/tables/human_evaluation/rq1_llm_vs_human_summary.csv
results/tables/human_evaluation/rq1_llm_pairwise_summary.csv
```

Use:

```text
rq1_llm_vs_human_summary.csv
```

for the main LLM-vs-human RQ1 results.

Use:

```text
rq1_human_pairwise_summary.csv
```

to report human-human agreement.

Use:

```text
rq1_llm_pairwise_summary.csv
```

to compare GPT-4o and DeepSeek extraction behavior.

### RQ2

Main RQ2 file:

```text
results/tables/human_evaluation/RQ2_Report_Test_human_llm_agreement.csv
```

This table contains LLM-vs-human agreement for:

```text
bug report ↔ test
```

It includes:

```text
datasets: Defects4J, SWTBench
representations: RAW, STRUCTURED
models: gpt4o, deepseek
human references: human1, human2, HumanAvg
metrics: alignment_score, coverage, correctness, specificity
```

Expected rows:

```text
96
```

### RQ3

Main RQ3 file:

```text
results/tables/human_evaluation/RQ3_Report_Patch_human_llm_agreement.csv
```

This table contains LLM-vs-human agreement for:

```text
bug report ↔ patch
```

It includes:

```text
datasets: Defects4J, SWTBench
representations: RAW, STRUCTURED
patch views: full, add, remove
models: gpt4o, deepseek
human references: human1, human2, HumanAvg
metrics: alignment_score, coverage, correctness, specificity
```

Expected rows:

```text
288
```

### RQ4 internal

Main RQ4 internal file:

```text
results/tables/human_evaluation/RQ4_Test_Patch_Internal_human_llm_agreement.csv
```

This table contains LLM-vs-human agreement for:

```text
test ↔ patch
```

It includes:

```text
datasets: Defects4J, SWTBench
representation: RAW
patch views: full, add, remove
models: gpt4o, deepseek
human references: human1, human2, HumanAvg
metrics: alignment_score, coverage, correctness, specificity
```

Expected rows:

```text
144
```

---

## 6. Debug summary files

Debug summary files are stored in:

```text
results/tables/human_evaluation/
```

Important files:

```text
rq1_debug_summary.csv
RQ2_Report_Test_debug_summary.csv
RQ3_Report_Patch_debug_summary.csv
RQ4_Test_Patch_Internal_debug_summary.csv
```

### Meaning of `human_without_llm`

This indicates that a human annotation exists but no matching LLM output was found.

Known expected cases are associated with a few Defects4J add/remove patch views:

```text
Codec__4 remove
Cli__17 remove
Closure__114 add
Math__20 add
```

### Meaning of `llm_without_human`

This indicates that an LLM output exists outside the human annotation sample.

These are usually old smoke-test outputs. They are kept for traceability but excluded from final agreement computations.

---

## 7. Detailed experiment outputs

Detailed, non-consolidated outputs are in:

```text
results/experiments/
```

### RQ1 extraction agreement

```text
results/experiments/rq1_human_extraction/
```

Important files:

```text
human_extractions_flat.csv
llm_extractions_flat.csv
human_pairwise_field_agreement.csv
human_pairwise_summary.csv
llm_vs_human_field_agreement.csv
llm_vs_human_summary.csv
llm_pairwise_field_agreement.csv
llm_pairwise_summary.csv
debug_human_without_llm.csv
debug_llm_without_human.csv
```

### RQ2/RQ3/RQ4 agreement

```text
results/experiments/human_agreement/
```

Important note:

```text
results/experiments/human_agreement/
```

is overwritten each time `scripts/run_human_agreement.py` is executed for a new RQ.

Use `results/tables/human_evaluation/` for persistent consolidated outputs.

### LLM judge outputs

```text
results/experiments/llm_judges/
```

Typical structure:

```text
results/experiments/llm_judges/
├── rq2/
├── rq3/
└── rq4_internal/
```

Each run stores JSONL score files by:

```text
rq
dataset
scenario
model
patch_view
```

---

## 8. Recommended reporting strategy

For paper-level reporting:

1. Use `global_human_evaluation_summary.csv` for the global overview.
2. Use the RQ-specific human-LLM agreement tables for detailed results.
3. Use debug summary files to document unmatched cases.
4. Use RQ1-specific summaries for extraction agreement rather than treating RQ1 as a 1-5 scoring task.

Recommended main table:

```text
results/tables/human_evaluation/global_human_evaluation_summary.csv
```

Recommended detailed tables:

```text
results/tables/human_evaluation/rq1_llm_vs_human_summary.csv
results/tables/human_evaluation/RQ2_Report_Test_human_llm_agreement.csv
results/tables/human_evaluation/RQ3_Report_Patch_human_llm_agreement.csv
results/tables/human_evaluation/RQ4_Test_Patch_Internal_human_llm_agreement.csv
```

Recommended debug evidence:

```text
results/tables/human_evaluation/rq1_debug_summary.csv
results/tables/human_evaluation/RQ2_Report_Test_debug_summary.csv
results/tables/human_evaluation/RQ3_Report_Patch_debug_summary.csv
results/tables/human_evaluation/RQ4_Test_Patch_Internal_debug_summary.csv
```

---

## 9. Quick checks

### Check global row counts

```bash
python - <<'PY'
import pandas as pd

base = "results/tables/human_evaluation"
g = pd.read_csv(f"{base}/global_human_evaluation_summary.csv")

print(g.groupby(["rq", "rq_label"]).size())
PY
```

### Check RQ-specific rows

```bash
python - <<'PY'
import pandas as pd

base = "results/tables/human_evaluation"

files = [
    "rq1_llm_vs_human_summary.csv",
    "RQ2_Report_Test_human_llm_agreement.csv",
    "RQ3_Report_Patch_human_llm_agreement.csv",
    "RQ4_Test_Patch_Internal_human_llm_agreement.csv",
]

for f in files:
    df = pd.read_csv(f"{base}/{f}")
    print(f, len(df))
PY
```

Expected:

```text
rq1_llm_vs_human_summary.csv 84
RQ2_Report_Test_human_llm_agreement.csv 96
RQ3_Report_Patch_human_llm_agreement.csv 288
RQ4_Test_Patch_Internal_human_llm_agreement.csv 144
```

Note: `rq1_llm_vs_human_summary.csv` includes both field-level and macro-level rows. The global table uses only the macro-level RQ1 rows.

---

## 10. Final validated state

The current validated final summary is:

```text
RQ1 macro rows: 12
RQ2 rows:       96
RQ3 rows:       288
RQ4 rows:       144
Total rows:     540
```

This means that the human-evaluation pipeline is complete and ready for documentation, README integration, and dashboard/report generation.
