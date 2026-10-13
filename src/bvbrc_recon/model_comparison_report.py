"""Renders reports/pilot_model_comparison.md."""

from __future__ import annotations

MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost"]


def _result_row(model_name: str, result) -> str:
    # LogisticRegression's result (PilotModelResult) has no `error`/`fit_seconds`/
    # `predict_seconds` attributes at all (not just None) - accessed defensively
    # so one shared renderer works for both result types.
    error = getattr(result, "error", None)
    if error:
        return f"| {model_name} | ERROR | {error} | - | - | - | - | - | - | - |"

    fit_seconds = getattr(result, "fit_seconds", None)
    predict_seconds = getattr(result, "predict_seconds", None)
    timing = f"{fit_seconds:.3f}s / {predict_seconds:.3f}s" if fit_seconds is not None else "n/a (not tracked for LogisticRegression)"

    return (
        f"| {model_name} | {result.n_train}/{result.n_test} | {result.train_class_counts} / "
        f"{result.test_class_counts} | {result.accuracy} | {result.roc_auc} | "
        f"{result.average_precision} | {result.recall_per_class.get('Resistant')} | "
        f"{result.recall_per_class.get('Susceptible')} | {result.confusion_matrix} | {timing} |"
    )


def _split_table(split_results: dict) -> str:
    header = (
        "| Model | Train/Test n | Train / Test class counts | Accuracy | ROC-AUC | "
        "Avg. precision | Recall (R) | Recall (S) | Confusion matrix [[TN,FP],[FN,TP]] | Fit/Predict time |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = "\n".join(_result_row(name, split_results[name]) for name in MODEL_ORDER)
    return header + rows


def _params_section(split_results: dict) -> str:
    # LogisticRegression's result (PilotModelResult) has no `.params` field -
    # its fixed configuration is documented inline, matching
    # reports/pilot_ciprofloxacin_report.md and reports/pilot_lineage_evaluation.md.
    lines = [
        "- **LogisticRegression**: `{'max_iter': 1000, 'random_state': seed}` "
        "(lbfgs solver, scikit-learn's default - unchanged from the original pilot)"
    ]
    for name in ("RandomForest", "XGBoost"):
        result = split_results[name]
        lines.append(f"- **{name}**: `{result.params}`")
    return "\n".join(lines)


def _warnings_section(all_results: dict) -> str:
    lines = []
    for split_name, split_results in all_results.items():
        for model_name in MODEL_ORDER:
            result = split_results[model_name]
            if result.warnings:
                lines.append(f"- [{split_name} split, {model_name}]: {'; '.join(result.warnings)}")
    return "\n".join(lines) if lines else "None."


def render_model_comparison_report(
    antibiotic: str,
    seed: int,
    analysis_date: str,
    n_genomes: int,
    matrix_shape: tuple,
    matrix_nnz: int,
    results: dict,
) -> str:
    density = matrix_nnz / (matrix_shape[0] * matrix_shape[1])
    return f"""# Pilot Model Comparison: Logistic Regression vs. Random Forest vs. XGBoost ({antibiotic.capitalize()})

- Analysis date (UTC): {analysis_date}
- Antibiotic: {antibiotic}
- Random seed (all models, both splits): {seed}

**PRELIMINARY PILOT RESULTS from a single ~100-genome cohort for one antibiotic - not the final
three-antibiotic benchmark.** Reuses the EXISTING k-mer feature matrix and the EXISTING saved
lineage-aware split; no genome was downloaded, no k-mer was recomputed, and the original pilot
cohort/matrix/splits were not modified.

## Dataset and feature matrix

- Genomes: {n_genomes} (25 Resistant / 75 Susceptible, per the original pilot cohort)
- Feature matrix: {matrix_shape[0]} x {matrix_shape[1]} canonical k=10 k-mers, {matrix_nnz:,} nonzero
  entries ({density:.1%} dense) - loaded from the existing
  `data/processed/pilot_{antibiotic}_kmer_matrix.npz`, never recomputed or densified by this task.

## Split methodology and saved assignments used

- **Random split**: the same stratified random split Logistic Regression's original pilot report
  used (same seed, same `test_size=0.3`). Row indices were regenerated via
  `compute_random_split_indices`, which is proven (by a dedicated unit test) to reproduce the
  *exact* same partition as the original pilot's `train_and_evaluate_pilot_model` call - not a new
  split. Saved for the first time as `data/processed/pilot_{antibiotic}_random_split.csv` for
  auditability.
- **Lineage-aware split**: loaded directly from the EXISTING
  `data/processed/pilot_{antibiotic}_lineage_split.csv` (MLST-group-held-out, produced by the prior
  task) - **not regenerated**. Genome-id-to-row alignment and the absence of any row overlap
  between partitions were verified before use.

## Model configurations

{_params_section(results["random"])}

All three models: no learned preprocessing (scaler, feature selection, etc.) was applied - raw
k-mer counts are fed directly to each model, exactly as for the original Logistic Regression
pilot, so "fit preprocessing on training data only" is trivially satisfied (there is none to leak).
No hyperparameter search was performed for any model; Random Forest and XGBoost use their
library's own defaults plus only `random_state` (reproducibility) and `n_jobs=-1` (parallelism,
does not change results). Random Forest's default `max_features="sqrt"` for classifiers was left
as-is deliberately: with {matrix_shape[1]:,} features and only {n_genomes} genomes (p >> n), this
default is the standard mitigation against each tree considering all features at every split, not
a tuning choice made here.

## Results - random split

{_split_table(results["random"])}

## Results - lineage-aware split

{_split_table(results["lineage"])}

## Runtime observations

Fit/predict times are wall-clock, single run, on this machine - indicative only, not benchmarked
across repetitions. At {matrix_shape[1]:,} features for only {n_genomes} genomes, Random Forest and
XGBoost both completed without exceeding available memory; neither required densifying the sparse
matrix (verified by tests that the sparse matrix's `.toarray()` is never invoked during
evaluation).

**XGBoost sparse-input caveat (documented, not resolved here)**: XGBoost's sparsity-aware split
finding treats an implicit zero in a sparse matrix as a "missing value" signal during tree
building, not necessarily as the literal count zero. For k-mer **count** data, a zero entry means
"this k-mer was not observed in this genome" - a meaningful value, not missing data. This is a
known characteristic of how XGBoost consumes sparse input in general, not a bug introduced here,
but it means XGBoost's internal treatment of the feature matrix is subtly different from Random
Forest's and Logistic Regression's (which treat the stored zero as an ordinary feature value) -
worth keeping in mind when comparing feature importances or fine-tuning later, though it is not
expected to invalidate the pilot comparison here.

## Comparison and interpretation

- **Random split vs. lineage-aware split is not a paired comparison**: the two splits use
  different genomes in the test set (different composition, different class balance, different
  group structure) - see `reports/pilot_lineage_evaluation.md` for the full discussion already
  established for Logistic Regression, which applies identically here for Random Forest and
  XGBoost.
- Any difference between a model's random-split and lineage-aware metrics is preliminary evidence
  of sensitivity to test-set composition, not proof that controlling for lineage alone caused the
  difference, and not evidence that one model is "better" in general from a single ~100-genome
  pilot.
- No model was tuned against either test set's results; the configurations above were fixed before
  any evaluation was run.

## Warnings raised during evaluation

{_warnings_section(results)}

## Computational limitations and caveats

- A single train/test split per model per evaluation strategy - no cross-validation, so variance
  across seeds/splits is not characterized here (noted as a next step in the lineage evaluation
  report too).
- {matrix_shape[1]:,} raw k=10 k-mer features for {n_genomes} genomes is an extreme p >> n regime;
  all three models are documented as prone to overfitting noise in this regime (Logistic
  Regression's L2 regularization, Random Forest's `max_features="sqrt"`, and XGBoost's shallow
  default tree depth all provide some mitigation, but none of this was tuned or verified beyond
  what each library does by default).
- This pilot's k=10 choice (vs. the separately unresolved proposal-vs-instructor k=10-vs-31
  question) was not revisited here - see `reports/pilot_{antibiotic}_report.md`.

## Pilot results vs. final benchmark - explicit distinction

These are preliminary results from ONE ~100-genome pilot cohort for ONE antibiotic
(ciprofloxacin), using ONE k-mer configuration (k=10) and ONE random seed per split. They do not
establish which model, if any, will perform best on the full three-antibiotic benchmark, do not
use lineage-aware splitting as the default evaluation method going forward (that decision is still
open), and are not a substitute for evaluating Ampicillin and Gentamicin or for a larger cohort.
"""
