# EcoliAMR-Bench

## Project Title
Benchmarking Machine Learning Models for Antimicrobial Resistance Prediction in *Escherichia coli* from Whole-Genome k-mer Profiles.

## Objective
Predict Resistant/Susceptible phenotypes for *E. coli* against three antibiotics from whole-genome k-mer profiles, and compare model performance under random versus lineage-aware (MLST-group) evaluation:

- Ciprofloxacin
- Ampicillin
- Gentamicin

## Status
**Primary benchmark complete.** The full pipeline - cohort selection, assembly retrieval, feature generation, model benchmarking, and scientific interpretation - has run end to end on a frozen 956-genome cohort. Remaining work is report/figure preparation for submission, plus optional future extensions (statistical significance testing, repeated-seed evaluation, hyperparameter tuning); see `reports/course_project_final_handoff.md` for the authoritative summary and checklist.

## Data Source
[BV-BRC](https://www.bv-brc.org/) - 956 *E. coli* genome assemblies retrieved and QC-validated (300 from earlier pilot/expansion rounds, 656 of 700 additionally selected genomes; 44 failed retrieval/identity validation and are permanently excluded, with root causes documented in `data/processed/course_project_additional_700_execution_summary.json`).

## Final Dataset

| Antibiotic | Usable genomes | Resistant | Susceptible |
|---|---|---|---|
| Ciprofloxacin | 836 | 190 | 646 |
| Ampicillin | 734 | 393 | 341 |
| Gentamicin | 827 | 93 | 734 |

Cohort selection, Policy A phenotype eligibility, and the sampling/retrieval history are documented in `reports/course_project_cohort_design.md` and `reports/course_project_final_cohort.md`.

## Feature Representation
**Primary representation**: direct k=15 feature hashing, 262,144 dimensions, built out-of-core (one genome's sequence hashed and discarded at a time via a SQLite-backed row store - never a fully in-memory raw k-mer matrix). Selected after comparing it against raw canonical k=10 k-mers and canonical k=31 minimizers on resource grounds (the only candidate that fits this project's 11GB memory-safety threshold at full cohort scale); see `reports/course_project_feature_generation.md`. This representation has no per-feature biological interpretability.

Earlier, smaller-scale explorations of raw k-mers (k=10-31), minimizers, and feature hashing at various dimensions are preserved in `reports/intermediate_k_comparison.md`, `reports/minimizer_resource_scaling_assessment.md`, and `reports/feature_hashing_comparison.md`.

## Models
- Logistic Regression
- Random Forest
- XGBoost

All three evaluated on both a stratified random 80/20 split and an MLST-group-aware split, for each antibiotic (18 evaluations total). Model configurations, seeds, and full results are in `reports/course_project_model_benchmark.md`.

## Evaluation Metrics
- ROC-AUC
- Average precision
- Balanced accuracy
- Macro-F1
- Per-class (Resistant/Susceptible) precision and recall
- Confusion matrix

Raw accuracy alone is deliberately not used to compare random vs. lineage-aware evaluation, since the two strategies' test folds differ in size and class prevalence for every antibiotic - see `reports/course_project_results_interpretation.md`.

## Key Finding
Lineage-aware evaluation scored lower than random evaluation in all 9 model x antibiotic combinations, on both ROC-AUC and balanced accuracy - a consistent direction, but not one this study's single-split design can attribute to a specific cause (clonal leakage vs. test-fold composition) or treat as statistically significant. Full interpretation, including the gentamicin small-test-set caveat (only 6 Resistant genomes in its lineage-aware test fold), is in `reports/course_project_results_interpretation.md`.

## Biological Interpretation
[CARD](https://card.mcmaster.ca/) (Comprehensive Antibiotic Resistance Database) was used in one **pilot-only, exploratory** side-study: top-ranked k-mers from a 70-genome, ciprofloxacin-only, raw k=10 refit were matched against CARD reference sequences (`reports/pilot_ciprofloxacin_biological_interpretation.md`). That exercise found no unambiguous match to a known fluoroquinolone-resistance gene and is explicitly not biological support for the main (hashed-feature) benchmark, which cannot be mapped back to individual k-mers or genes at all. See `reports/course_project_final_handoff.md` Section 4 for the full audit.

## Repository Structure
- `src/bvbrc_recon/` - reusable library code (retrieval, QC, feature extraction, splitting, modeling, resource estimation)
- `src/run_*.py` - one-off task scripts that produce the artifacts under `data/processed/` and `reports/`
- `data/raw/`, `data/processed/`, `data/external/` - retrieved assemblies, derived artifacts/manifests, and external reference data (e.g. CARD)
- `reports/` - one Markdown report per task; `course_project_final_handoff.md` is the entry point for the final study
- `tests/` - unit/integration tests (549 passing)

## Reproducing
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```
Individual pipeline stages are re-run via the scripts in `src/run_*.py`; each script's docstring documents what it reads and writes.
