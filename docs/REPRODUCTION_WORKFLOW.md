# Reproduction Workflow

This document describes the recommended end-to-end workflow for reproducing the Desc2Fix package outputs.

It focuses on the practical execution order: environment setup, corpus preparation, LLM extraction, RQ-specific analyses, LLM judges, human agreement, and final consolidated tables.

---

## 1. Environment setup

From the repository root:

```bash
cd /home/e113/Documents/FROM-DESCRIPTION-TO-FIX/For_GitHub/Desc2Fix/Desc2Fix
```

Activate the environment if needed:

```bash
source des2fix_env/bin/activate
```

Install the package in editable mode:

```bash
python -m pip install -e .
```

If the current shell does not find the local package, export the source path:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

Quick import check:

```bash
python - <<'PY'
import desc2fix
print("desc2fix import OK")
PY
```

---

## 2. Configuration files

The main configuration files are:

```text
config/datasets.yaml
config/models.yaml
config/paths.yaml
```

### Models

The package currently uses:

```text
gpt4o     -> OpenAI GPT-4o
deepseek  -> DeepSeek Chat through OpenAI-compatible API
```

Model credentials should be defined in the local environment or `.env` file.

---

## 3. Benchmark setup

Benchmark setup prepares the local raw materials needed for corpus construction.

```bash
python scripts/setup_benchmarks.py
```

This step may download or verify benchmark resources depending on the local configuration.

---

## 4. Corpus preparation

Prepare the report-test-patch triplets:

```bash
python scripts/prepare_corpus.py --dataset defects4j
python scripts/prepare_corpus.py --dataset swtbench
```

Expected processed outputs:

```text
data/processed/defects4j/
data/processed/swtbench/
```

The processed triplets are the main input for extraction, metric computation, and LLM judging.

---

## 5. Structured signal extraction

Structured bug-report signals are extracted with:

```bash
python scripts/extract_signals.py --dataset defects4j --model gpt4o --resume
python scripts/extract_signals.py --dataset defects4j --model deepseek --resume
python scripts/extract_signals.py --dataset swtbench --model gpt4o --resume
python scripts/extract_signals.py --dataset swtbench --model deepseek --resume
```

Expected output layout:

```text
data/structured/
├── defects4j/
│   ├── gpt4o/
│   └── deepseek/
└── swtbench/
    ├── gpt4o/
    └── deepseek/
```

For extracting only the human-evaluation sample:

```bash
python scripts/extract_signals.py   --dataset defects4j   --model gpt4o   --sample-from-human Human_Eval   --resume
```

Repeat for all dataset/model combinations.

---

## 6. RQ1: structured signal extraction analysis

Run the standard RQ1 analysis:

```bash
python scripts/run_rq1.py
```

Run RQ1 human extraction agreement:

```bash
python scripts/run_rq1_human_extraction_agreement.py   --human-root Human_Eval   --structured-root data/structured
```

Expected outputs:

```text
results/experiments/rq1_human_extraction/
```

Key files:

```text
human_pairwise_summary.csv
llm_vs_human_summary.csv
llm_pairwise_summary.csv
debug_human_without_llm.csv
debug_llm_without_human.csv
```

---

## 7. RQ2: bug report ↔ test alignment

### 7.1 Metric-based analysis

```bash
python scripts/run_rq2.py
```

### 7.2 LLM judge analysis

Run LLM judges for both datasets, both models, and both scenarios:

```bash
python scripts/run_llm_judges.py   --rq rq2   --dataset defects4j   --model gpt4o   --scenario both   --sample-from-human Human_Eval   --resume

python scripts/run_llm_judges.py   --rq rq2   --dataset defects4j   --model deepseek   --scenario both   --sample-from-human Human_Eval   --resume

python scripts/run_llm_judges.py   --rq rq2   --dataset swtbench   --model gpt4o   --scenario both   --sample-from-human Human_Eval   --resume

python scripts/run_llm_judges.py   --rq rq2   --dataset swtbench   --model deepseek   --scenario both   --sample-from-human Human_Eval   --resume
```

### 7.3 Human agreement

```bash
python scripts/run_human_agreement.py   --rq rq2   --human-root Human_Eval
```

---

## 8. RQ3: bug report ↔ patch alignment

### 8.1 Metric-based analysis

```bash
python scripts/run_rq3.py
```

### 8.2 LLM judge analysis

RQ3 uses three patch views:

```text
full
add
remove
```

Run for each patch view, dataset, model, and scenario.

Example for Defects4J/GPT-4o/full:

```bash
python scripts/run_llm_judges.py   --rq rq3   --dataset defects4j   --model gpt4o   --scenario both   --patch-view full   --sample-from-human Human_Eval   --resume
```

Recommended loop:

```bash
for dataset in defects4j swtbench; do
  for model in gpt4o deepseek; do
    for patch_view in full add remove; do
      python scripts/run_llm_judges.py         --rq rq3         --dataset "$dataset"         --model "$model"         --scenario both         --patch-view "$patch_view"         --sample-from-human Human_Eval         --resume
    done
  done
done
```

### 8.3 Human agreement

```bash
python scripts/run_human_agreement.py   --rq rq3   --human-root Human_Eval
```

---

## 9. RQ4 internal: test ↔ patch alignment

RQ4 internal is an internal/off-paper analysis.

### 9.1 Metric-based analysis

```bash
python scripts/run_rq4_internal.py
```

### 9.2 LLM judge analysis

RQ4 internal uses RAW representation and three patch views:

```text
full
add
remove
```

Recommended loop:

```bash
for dataset in defects4j swtbench; do
  for model in gpt4o deepseek; do
    for patch_view in full add remove; do
      python scripts/run_llm_judges.py         --rq rq4_internal         --dataset "$dataset"         --model "$model"         --scenario raw         --patch-view "$patch_view"         --sample-from-human Human_Eval         --resume
    done
  done
done
```

### 9.3 Human agreement

```bash
python scripts/run_human_agreement.py   --rq rq4_internal   --human-root Human_Eval
```

---

## 10. Consolidated human-evaluation tables

After RQ1/RQ2/RQ3/RQ4 human outputs are available, generate the final tables:

```bash
python scripts/generate_human_summary_tables.py   --human-root Human_Eval
```

Final output directory:

```text
results/tables/human_evaluation/
```

Main final table:

```text
results/tables/human_evaluation/global_human_evaluation_summary.csv
```

Expected row counts:

```text
RQ1: 12
RQ2: 96
RQ3: 288
RQ4 internal: 144
Total: 540
```

Validation command:

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

## 11. Recommended quick reproduction path

If processed data, structured signals, LLM judge outputs, and human annotations are already available, the fastest validation path is:

```bash
python -m pip install -e .

python scripts/run_rq1_human_extraction_agreement.py   --human-root Human_Eval   --structured-root data/structured

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
PY
```

---

## 12. Debug outputs

The pipeline writes debug files to document unmatched cases.

### Human without LLM

```text
debug_human_without_llm.csv
```

A row means that a human annotation exists but the corresponding LLM output was not found.

Known expected Defects4J add/remove cases include:

```text
Codec__4 remove
Cli__17 remove
Closure__114 add
Math__20 add
```

### LLM without human

```text
debug_llm_without_human.csv
```

A row means that an LLM output exists outside the human sample. These are usually old smoke-test outputs. They are excluded from final agreement computations.

---

## 13. Notes on reproducibility

Use `--resume` for LLM-based scripts to avoid recomputing completed outputs.

Use `--sample-from-human Human_Eval` when reproducing only the human-evaluation subset.

For full-dataset experiments, omit `--sample-from-human`.

For RQ3 and RQ4 internal, always check patch views separately:

```text
full
add
remove
```

---

## 14. Final validated state

The current human-evaluation reproduction state is:

```text
RQ1 extraction agreement: complete
RQ2 report-test agreement: complete
RQ3 report-patch agreement: complete
RQ4 internal test-patch agreement: complete
Global human-evaluation table: complete
```

The final consolidated table contains 540 rows.
