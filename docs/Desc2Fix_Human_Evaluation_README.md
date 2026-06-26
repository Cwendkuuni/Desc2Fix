# Desc2Fix Human Evaluation Package

This README documents the human-evaluation part of the Desc2Fix replication package.

The package uses `Human_Eval/` as the canonical directory for human annotations.
---

## 1. Repository layout

```text
Desc2Fix/
├── Human_Eval/
│   ├── RQ1/
│   ├── RQ2/
│   ├── RQ3/
│   └── RQ4/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── structured/
│   └── cache/
├── results/
│   ├── experiments/
│   └── tables/
├── scripts/
├── src/
└── docs/
```

Main folders:

```text
Human_Eval/                  Human annotations
data/raw/                    Raw or downloaded benchmark artifacts
data/processed/              Reconstructed report-test-patch triplets
data/structured/             LLM-extracted structured bug signals
results/experiments/         Detailed experiment outputs
results/tables/              Consolidated final tables
scripts/                     Reproduction scripts
src/desc2fix/                Python package source code
```

---

## 2. Research question mapping

```text
RQ1 = Structured bug signal extraction
RQ2 = Bug report ↔ test alignment
RQ3 = Bug report ↔ patch alignment
RQ4 internal = Test ↔ patch alignment
```

Mapping from older local scripts:

```text
Old RQ1.1 = bug report ↔ triggering/relevant test  => package RQ2
Old RQ2.1 = bug report ↔ patch                    => package RQ3
Old RQ3.1 = test ↔ patch                          => package RQ4 internal
```

---

## 3. Human annotation folder

Human annotations are stored in:

```text
Human_Eval/
├── RQ1/
├── RQ2/
├── RQ3/
└── RQ4/
```

### RQ1

`Human_Eval/RQ1/` contains human annotations for extracted bug-report signals.

Expected files:

```text
Human_Eval/RQ1/
├── Defects4J_human1_extracted_signals.jsonl
├── Defects4J_human2_extracted_signals.jsonl
├── SWTBench_human1_extracted_signals.jsonl
└── SWTBench_human2_extracted_signals.jsonl
```

RQ1 compares six extracted fields:

```text
exception
api_involved
input_conditions
expected
actual
reproduction_steps
```

### RQ2

`Human_Eval/RQ2/` contains human alignment scores for:

```text
bug report ↔ test
```

It includes both RAW and STRUCTURED representations.

### RQ3

`Human_Eval/RQ3/` contains human alignment scores for:

```text
bug report ↔ patch
```

It includes:

```text
representations: RAW, STRUCTURED
patch views:     full, add, remove
```

### RQ4 internal

`Human_Eval/RQ4/` contains human alignment scores for the internal/off-paper analysis:

```text
test ↔ patch
```

It uses:

```text
representation: RAW
patch views:    full, add, remove
```

---

## 4. Human score schema for RQ2/RQ3/RQ4

For RQ2, RQ3, and RQ4 internal, annotations use four scores:

```text
alignment_score
coverage
correctness
specificity
```

The package compares:

```text
human1 ↔ human2
GPT-4o ↔ human1
GPT-4o ↔ human2
GPT-4o ↔ HumanAvg
DeepSeek ↔ human1
DeepSeek ↔ human2
DeepSeek ↔ HumanAvg
```

The main statistics are:

```text
Spearman correlation
Kendall correlation
MAE
RMSE
Mean bias
```

---

## 5. RQ1 extraction-agreement metrics

RQ1 does not use 1-5 alignment scores. It compares structured extracted signals.

The package computes:

```text
field-level presence agreement
field-level similarity
macro signal similarity
signal count difference
```

Outputs are produced for:

```text
human1 ↔ human2
GPT-4o ↔ human1/human2/HumanAvg
DeepSeek ↔ human1/human2/HumanAvg
GPT-4o ↔ DeepSeek
```

---

## 6. Core commands

Before running scripts, install the local package:

```bash
python -m pip install -e .
```

If needed for the current shell session:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

### RQ1 human extraction agreement

```bash
python scripts/run_rq1_human_extraction_agreement.py \
  --human-root Human_Eval \
  --structured-root data/structured
```

Outputs:

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

### RQ2 human agreement

```bash
python scripts/run_human_agreement.py --rq rq2 --human-root Human_Eval
```

### RQ3 human agreement

```bash
python scripts/run_human_agreement.py --rq rq3 --human-root Human_Eval
```

### RQ4 internal human agreement

```bash
python scripts/run_human_agreement.py --rq rq4_internal --human-root Human_Eval
```

### Consolidated human-evaluation tables

```bash
python scripts/generate_human_summary_tables.py --human-root Human_Eval
```

Outputs:

```text
results/tables/human_evaluation/
├── global_human_evaluation_summary.csv
├── rq1_human_pairwise_summary.csv
├── rq1_llm_vs_human_summary.csv
├── rq1_llm_pairwise_summary.csv
├── RQ2_Report_Test_human_llm_agreement.csv
├── RQ3_Report_Patch_human_llm_agreement.csv
├── RQ4_Test_Patch_Internal_human_llm_agreement.csv
└── *_debug_summary.csv
```

---

## 7. Current validated output sizes

The current validated human-evaluation pipeline produces:

```text
RQ1:
  human annotations: 400
  human_without_llm: 0
  LLM rows used for agreement: 399
  extra LLM rows outside human sample: 18

RQ2:
  human_flat: 800
  human_agg: 400
  human_llm_agreement: 96

RQ3:
  human_flat: 2236
  human_agg: 1118
  human_llm_agreement: 288

RQ4 internal:
  human_flat: 1118
  human_agg: 559
  human_llm_agreement: 144

Global summary:
  RQ1: 12 rows
  RQ2: 96 rows
  RQ3: 288 rows
  RQ4: 144 rows
  Total: 540 rows
```

The consolidated global table is:

```text
results/tables/human_evaluation/global_human_evaluation_summary.csv
```

Expected groups:

```text
rq1           RQ1_Extraction              12
rq2           RQ2_Report_Test             96
rq3           RQ3_Report_Patch           288
rq4_internal  RQ4_Test_Patch_Internal    144
```

---

## 8. Debug-file interpretation

Debug files are not necessarily failures. They document unmatched cases.

### `debug_human_without_llm.csv`

A row means that a human annotation exists, but no matching LLM output was found.

Known remaining cases:

```text
Codec__4 remove
Cli__17 remove
Closure__114 add
Math__20 add
```

These occur in Defects4J add/remove patch views and are likely due to missing or non-exploitable add/remove patch views in the reconstructed corpus.

### `debug_llm_without_human.csv`

A row means that an LLM output exists outside the human annotation sample.

These are usually old smoke-test outputs kept for traceability. They are excluded from agreement computations when using the final summary scripts.

---

## 9. Current known limitations

1. A few Defects4J add/remove patch views have human annotations but no matching LLM judge output.
2. Some old smoke-test LLM outputs remain in `results/experiments/llm_judges/` and `data/structured/`.
3. Some DeepSeek outputs occasionally fail JSON parsing. These failures affect only a small number of instances and are recorded in debug outputs.
4. RQ4 internal is not part of the main paper RQ mapping; it is kept as an internal/off-paper test ↔ patch analysis.

---

## 10. Recommended final table for reporting

Use:

```text
results/tables/human_evaluation/global_human_evaluation_summary.csv
```

This table unifies:

```text
RQ1 macro signal similarity and presence agreement
RQ2 human ↔ LLM alignment agreement
RQ3 human ↔ LLM patch-alignment agreement
RQ4 internal human ↔ LLM test-patch agreement
```

For detailed per-RQ tables, use:

```text
results/tables/human_evaluation/rq1_llm_vs_human_summary.csv
results/tables/human_evaluation/RQ2_Report_Test_human_llm_agreement.csv
results/tables/human_evaluation/RQ3_Report_Patch_human_llm_agreement.csv
results/tables/human_evaluation/RQ4_Test_Patch_Internal_human_llm_agreement.csv
```

---

## 11. Minimal reproduction checklist

```bash
python -m pip install -e .

python scripts/run_rq1_human_extraction_agreement.py \
  --human-root Human_Eval \
  --structured-root data/structured

python scripts/run_human_agreement.py --rq rq2 --human-root Human_Eval
python scripts/run_human_agreement.py --rq rq3 --human-root Human_Eval
python scripts/run_human_agreement.py --rq rq4_internal --human-root Human_Eval

python scripts/generate_human_summary_tables.py --human-root Human_Eval
```

Then inspect:

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


