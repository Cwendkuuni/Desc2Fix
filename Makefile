.PHONY: install setup corpus validate extract rq1 rq2 rq3 rq4 tables figures dashboard full no-llm

install:
	python -m pip install -r requirements.txt
	python -m spacy download en_core_web_sm

setup:
	python scripts/setup_benchmarks.py --all

corpus:
	python scripts/prepare_corpus.py --all --mode raw

validate:
	python scripts/validate_corpus.py --all

extract:
	python scripts/extract_signals.py --all --resume

rq1:
	python scripts/run_rq1.py --all

rq2:
	python scripts/run_rq2.py --all

rq3:
	python scripts/run_rq3.py --all

rq4:
	python scripts/run_rq4_internal.py --all

tables:
	python scripts/generate_tables.py --all

figures:
	python scripts/generate_figures.py --all

dashboard:
	streamlit run dashboard/app.py

full:
	bash reproduce.sh --full

no-llm:
	bash reproduce.sh --no-llm
