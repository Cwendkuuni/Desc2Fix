# Human Evaluation Guide

This document explains the human-evaluation component of the Desc2Fix replication package.

It covers the RQ mapping, human annotation structure, validation scripts, generated outputs, and interpretation of debug files.

---

## 1. Purpose

The human-evaluation package supports the validation of Desc2Fix results across four evaluation tracks:

```text
RQ1 = Structured bug signal extraction
RQ2 = Bug report ↔ test alignment
RQ3 = Bug report ↔ patch alignment
RQ4 internal = Test ↔ patch alignment
```

RQ1 evaluates whether LLMs extract structured bug-report signals consistently with human annotators.

RQ2, RQ3, and RQ4 internal evaluate whether LLM judges agree with human annotators when assessing semantic alignment between artifacts.

---

## 2. Human annotation directory

Human annotations are stored in:

```text
Human_Eval/
├── RQ1/
├── RQ2/
├── RQ3/
└── RQ4/
```

This is the canonical human-annotation directory used by the scripts.

---

## 3. RQ mapping

The current package uses the following mapping:

| Package RQ | Meaning | Artifact pair |
|---|---|---|
| RQ1 | Structured bug signal extraction | Bug report → structured signals |
| RQ2 | Report-test alignment | Bug report ↔ test |
| RQ3 | Report-patch alignment | Bug report ↔ patch |
| RQ4 internal | Test-patch alignment | Test ↔ patch |

---

## 4. RQ1: structured bug signal extraction

### 4.1 Goal

RQ1 evaluates the extraction of six structured signals from bug reports:

```text
exception
api_involved
input_conditions
expected
actual
reproduction_steps
```

The comparison is performed between:

```text
human1 ↔ human2
GPT-4o ↔ human1
GPT-4o ↔ human2
GPT-4o ↔ HumanAvg
DeepSeek ↔ human1
DeepSeek ↔ human2
DeepSeek ↔ HumanAvg
GPT-4o ↔ DeepSeek
```

### 4.2 Expected human files

```text
Human_Eval/RQ1/
├── Defects4J_human1_extracted_signals.jsonl
├── Defects4J_human2_extracted_signals.jsonl
├── SWTBench_human1_extracted_signals.jsonl
└── SWTBench_human2_extracted_signals.jsonl
```

### 4.3 Metrics

RQ1 uses extraction-specific metrics:

```text
presence_agreement
similarity
signal_count_difference
macro_signal_similarity
```

The similarity is computed at field level and then summarized at macro level.

### 4.4 Command

```bash
python scripts/run_rq1_human_extraction_agreement.py \
  --human-root Human_Eval \
  --structured-root data/structured
```

### 4.5 Outputs

```text
results/experiments/rq1_human_extraction/
├── human_extractions_flat.csv
├── llm_extractions_flat.csv
├── human_pairwise_field_agreement.csv
├── human_pairwise_summary.csv
├── llm_vs_human_field_agreement.csv
├── llm_vs_human_summary.csv
├── llm_pairwise_field_agreement.csv
├── llm_pairwise_summary.csv
├── debug_human_without_llm.csv
└── debug_llm_without_human.csv
```

### 4.6 Validated current status

```text
Human annotations: 400
Human annotations without LLM output: 0
LLM rows used for agreement: 399
Extra LLM rows outside human sample: 18
```

---

## 5. RQ2: bug report ↔ test alignment

### 5.1 Goal

RQ2 evaluates whether generated or selected tests align with the original bug report.

Artifact pair:

```text
bug report ↔ test
```

### 5.2 Representations

RQ2 includes:

```text
RAW
STRUCTURED
```

### 5.3 Human score schema

Each pair is scored using:

```text
alignment_score
coverage
correctness
specificity
```

### 5.4 Command

```bash
python scripts/run_human_agreement.py \
  --rq rq2 \
  --human-root Human_Eval
```

### 5.5 Validated current status

```text
human_flat: 800
human_agg: 400
human_llm_agreement: 96
```

---

## 6. RQ3: bug report ↔ patch alignment

### 6.1 Goal

RQ3 evaluates whether a patch aligns with the original bug report.

Artifact pair:

```text
bug report ↔ patch
```

### 6.2 Representations and patch views

RQ3 includes:

```text
representations:
  RAW
  STRUCTURED

patch views:
  full
  add
  remove
```

### 6.3 Human score schema

Each pair is scored using:

```text
alignment_score
coverage
correctness
specificity
```

### 6.4 Command

```bash
python scripts/run_human_agreement.py \
  --rq rq3 \
  --human-root Human_Eval
```

### 6.5 Validated current status

```text
human_flat: 2236
human_agg: 1118
human_llm_agreement: 288
```

### 6.6 Known missing add/remove cases

A few Defects4J add/remove patch views have human annotations but no matching LLM judge output:

```text
Codec__4 remove
Cli__17 remove
Closure__114 add
Math__20 add
```

These cases are reported in debug files and are not silently ignored.

---

## 7. RQ4 internal: test ↔ patch alignment

### 7.1 Goal

RQ4 internal evaluates whether the test artifact aligns with the patch.

Artifact pair:

```text
test ↔ patch
```

This analysis is kept as an internal/off-paper track.

### 7.2 Representation and patch views

RQ4 internal uses:

```text
representation:
  RAW

patch views:
  full
  add
  remove
```

### 7.3 Human score schema

Each pair is scored using:

```text
alignment_score
coverage
correctness
specificity
```

### 7.4 Command

```bash
python scripts/run_human_agreement.py \
  --rq rq4_internal \
  --human-root Human_Eval
```

### 7.5 Validated current status

```text
human_flat: 1118
human_agg: 559
human_llm_agreement: 144
```

### 7.6 Known missing add/remove cases

The same Defects4J add/remove limitations affect RQ4 internal:

```text
Codec__4 remove
Cli__17 remove
Closure__114 add
Math__20 add
```

---

## 8. Standard human-agreement outputs

For RQ2, RQ3, and RQ4 internal, the standard script writes to:

```text
results/experiments/human_agreement/
```

Expected files:

```text
human_flat.csv
human_agg.csv
inter_human_agreement.csv
human_llm_agreement.csv
debug_human_without_llm.csv
debug_llm_without_human.csv
```

Important note:

```text
results/experiments/human_agreement/
```

is overwritten each time `scripts/run_human_agreement.py` is run for a different RQ.

For persistent consolidated tables, use the summary-generation script described below.

---

## 9. Consolidated human-evaluation tables

To regenerate all final human-evaluation tables:

```bash
python scripts/generate_human_summary_tables.py --human-root Human_Eval
```

This writes persistent per-RQ and global tables under:

```text
results/tables/human_evaluation/
```

Important files:

```text
global_human_evaluation_summary.csv
rq1_human_pairwise_summary.csv
rq1_llm_vs_human_summary.csv
rq1_llm_pairwise_summary.csv
RQ2_Report_Test_human_llm_agreement.csv
RQ3_Report_Patch_human_llm_agreement.csv
RQ4_Test_Patch_Internal_human_llm_agreement.csv
rq1_debug_summary.csv
RQ2_Report_Test_debug_summary.csv
RQ3_Report_Patch_debug_summary.csv
RQ4_Test_Patch_Internal_debug_summary.csv
```

---

## 10. Global summary table

The main final table is:

```text
results/tables/human_evaluation/global_human_evaluation_summary.csv
```

It combines:

```text
RQ1 macro signal similarity and presence agreement
RQ2 human ↔ LLM alignment agreement
RQ3 human ↔ LLM patch-alignment agreement
RQ4 internal human ↔ LLM test-patch agreement
```

Validated row counts:

```text
RQ1: 12
RQ2: 96
RQ3: 288
RQ4 internal: 144
Total: 540
```

Expected check:

```bash
python - <<'PY'
import pandas as pd

base = "results/tables/human_evaluation"
g = pd.read_csv(f"{base}/global_human_evaluation_summary.csv")

print(g.groupby(["rq", "rq_label"]).size())
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

## 11. Debug-file interpretation

### 11.1 `debug_human_without_llm.csv`

This file records human annotations for which no matching LLM output was found.

A non-empty file is not automatically a pipeline failure. It means the missing cases must be inspected and documented.

Known expected cases are related to missing add/remove patch views in Defects4J.

### 11.2 `debug_llm_without_human.csv`

This file records LLM outputs outside the human annotation sample.

These are usually old smoke-test outputs. They are kept for traceability but excluded from final agreement computations.

---

## 12. Minimal validation checklist

Run:

```bash
python scripts/run_rq1_human_extraction_agreement.py \
  --human-root Human_Eval \
  --structured-root data/structured

python scripts/run_human_agreement.py --rq rq2 --human-root Human_Eval
python scripts/run_human_agreement.py --rq rq3 --human-root Human_Eval
python scripts/run_human_agreement.py --rq rq4_internal --human-root Human_Eval

python scripts/generate_human_summary_tables.py --human-root Human_Eval
```

Then check:

```bash
python - <<'PY'
import pandas as pd

base = "results/tables/human_evaluation"

g = pd.read_csv(f"{base}/global_human_evaluation_summary.csv")
print(g.groupby(["rq", "rq_label"]).size())

for f in [
    "rq1_debug_summary.csv",
    "RQ2_Report_Test_debug_summary.csv",
    "RQ3_Report_Patch_debug_summary.csv",
    "RQ4_Test_Patch_Internal_debug_summary.csv",
]:
    print("\n---", f, "---")
    print(pd.read_csv(f"{base}/{f}"))
PY
```

---

## 13. Final validated state

The current human-evaluation package is validated with:

```text
RQ1 extraction human agreement: complete
RQ2 report-test human agreement: complete
RQ3 report-patch human agreement: complete
RQ4 test-patch internal agreement: complete
Consolidated global table: complete
```

The only known remaining unmatched cases are documented in debug outputs and correspond to a small number of Defects4J add/remove patch views or old smoke-test outputs.
