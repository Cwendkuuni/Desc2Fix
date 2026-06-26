# Desc2Fix Human Evaluation Plotly Figures

Generated with:

```bash
python scripts/generate_plotly_figures.py
```

This directory contains Plotly HD figures exported as HTML and static image files.

Main figures:

- `fig00_human_eval_row_counts`: global row counts by RQ.
- `fig01_rq1_llm_vs_human_similarity_presence`: RQ1 LLM-vs-HumanAvg extraction agreement.
- `fig02_rq1_pairwise_human_and_llm`: RQ1 human-human and model-model agreement.
- `fig03_rq2_spearman_heatmap_humanavg`: RQ2 HumanAvg Spearman heatmap.
- `fig04_rq2_mae_grid_humanavg`: RQ2 HumanAvg MAE grid.
- `fig05_rq3_spearman_heatmap_<dataset>`: RQ3 HumanAvg Spearman heatmaps.
- `fig06_rq3_mae_grid_<dataset>`: RQ3 HumanAvg MAE grids.
- `fig07_rq4_spearman_heatmap_<dataset>`: RQ4 internal HumanAvg Spearman heatmaps.
- `fig08_rq4_mae_grid_<dataset>`: RQ4 internal HumanAvg MAE grids.
- `fig09_global_spearman_mae_by_rq`: global mean Spearman and MAE by RQ.
- `fig10_debug_unmatched_cases`: unmatched debug cases.

Notes:

- Static export requires Kaleido.
- HTML export does not require Kaleido.
- Figures use large fonts and white backgrounds for LaTeX and paper readability.