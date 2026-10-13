"""Renders reports/intermediate_k_comparison.md."""

from __future__ import annotations


def _resource_table(k_configs: dict[int, dict]) -> str:
    lines = [
        "| k | Genomes | n_features | nnz | Density | Extraction time | Peak RSS | npz storage | In-memory CSR |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(k_configs):
        cfg = k_configs[k]
        if cfg.get("failure"):
            lines.append(f"| {k} | FAILED | - | - | - | - | - | - | {cfg['failure']} |")
            continue
        status = "complete" if cfg["complete"] else "INCOMPLETE/ABORTED"
        lines.append(
            f"| {k} | {cfg['n_genomes_processed']}/{cfg['n_genomes_requested']} ({status}) | "
            f"{cfg['n_features']:,} | {cfg['nnz']:,} | {cfg['density']:.2%} | "
            f"{cfg['extraction_seconds']:.1f}s | {cfg['peak_rss_mb']:.0f} MB | "
            f"{cfg['npz_storage_bytes']/1e6:.1f} MB | {cfg['in_memory_csr_bytes']/1e6:.1f} MB |"
        )
    return "\n".join(lines)


def _model_table(lr_results: dict[int, dict]) -> str:
    lines = [
        "| k | Split | Accuracy | ROC-AUC | Avg. precision | Recall (R) | Recall (S) | Confusion matrix |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k in sorted(lr_results):
        for split_name in ("random", "lineage"):
            r = lr_results[k].get(split_name)
            if r is None:
                lines.append(f"| {k} | {split_name} | n/a (matrix incomplete/failed) | - | - | - | - | - |")
                continue
            lines.append(
                f"| {k} | {split_name} | {r['accuracy']} | {r['roc_auc']} | {r['average_precision']} | "
                f"{r['recall_per_class'].get('Resistant')} | {r['recall_per_class'].get('Susceptible')} | "
                f"{r['confusion_matrix']} |"
            )
    return "\n".join(lines)


def _scaling_table(scaling_rows: list[dict]) -> str:
    lines = [
        "| k | Target cohort | Regime (pilot-measured) | Est. n_features | Est. CSR memory |",
        "|---|---|---|---|---|",
    ]
    for row in scaling_rows:
        lines.append(
            f"| {row['k']} | {row['label']} | {row['regime']} | "
            f"{row['n_features_low']:,}-{row['n_features_high']:,} | {row['csr_gb']:.1f} GB |"
        )
    return "\n".join(lines)


def render_intermediate_k_report(
    analysis_date: str,
    k_configs: dict[int, dict],
    lr_results: dict[int, dict],
    scaling_rows: list[dict],
    narrative_findings: str,
    recommendation: str,
) -> str:
    return f"""# Intermediate k-mer Configuration Comparison (k=10, 15, 17, 19, 21, 31)

- Analysis date (UTC): {analysis_date}
- Scope: uses ONLY the existing 100-genome ciprofloxacin pilot assemblies already on disk. No
  additional genome was downloaded. No full-scale feature matrix was built. k=10 and k=31 results
  are the EXISTING artifacts from the prior feasibility task, reused here unmodified - not rerun.
- **No final k-mer configuration has been approved for the full three-antibiotic benchmark as a
  result of this task.** This remains an open decision.

## Methodology and configuration

- k=15, 17, 19, 21 were extracted using the existing memory-safe streaming builder
  (`bvbrc_recon.streaming_kmer_build.build_matrix_streaming`), run as a SEPARATE OS process per k
  value (`src/run_intermediate_k_experiment.py <k>`) so that memory is fully released between k
  values rather than accumulating within one long-lived process.
- Canonicalization, ambiguous-base skipping, and per-contig (never cross-contig) window counting
  are identical to every other k already computed in this project - same function, not a
  reimplementation.
- Logistic Regression only (`bvbrc_recon.pilot_model.evaluate_with_precomputed_split`), same fixed
  configuration already used for k=10/k=31 (`max_iter=1000, random_state=20261009`), evaluated on
  the EXISTING saved random and lineage splits - neither split was regenerated, and no model
  hyperparameter was chosen or adjusted based on any test-set result.
- Random Forest and XGBoost were intentionally NOT run for every intermediate k (by task
  instruction) - the existing k=10/k=31 RF/XGBoost results from the prior task are not reproduced
  for k=15-21.

## Resource and feature-space comparison (measured, all k values)

{_resource_table(k_configs)}

**Observed pattern (measured, not assumed)**: feature count grows sharply from k=10 to k=15 (524K
to 23.2M) but far more gradually from k=15 through k=31 (23.2M to 32.6M) - consistent with k=10
being governed by the small, quickly-saturated theoretical k-mer space, while k>=15 is instead
governed by the actual sequence diversity across these 100 *E. coli* genomes, which is a much
larger but still bounded quantity. Peak memory and total nonzero count are similarly flat across
k=15-31 (~496-507M nonzeros, ~7.0-7.6GB peak RSS) - **the pilot-scale memory cost of k=15 is
already nearly as high as k=31's**, despite the large difference in nominal k.

No k value failed or triggered the memory-safety abort at this 100-genome pilot scale - all six
configurations (including the pre-existing k=10/k=31) completed with `complete: true`.

## Logistic Regression results on existing saved splits

{_model_table(lr_results)}

**These are exploratory results from the SAME 100-genome pilot reused across every k value - they
do not establish which k generalizes better to the full dataset, and no k here is declared "best"
on the strength of these numbers alone.**

{narrative_findings}

## Full-dataset scaling risk (projected from pilot measurements - NOT known exactly)

{_scaling_table(scaling_rows)}

**Full-scale memory usage is NOT known exactly from a 100-genome pilot** - every number above is a
projection using `resource_estimation.py`'s explicit, documented extrapolation rules (measured
nnz-per-genome scaled linearly; column count bounded between the pilot's own count and a capped
naive linear extrapolation). The practically important, somewhat counterintuitive finding: because
final sparse-matrix memory is dominated by nonzero COUNT (which is nearly flat across k=15-31 at
pilot scale) rather than by column count, **k=15 through k=31 are all projected to require a
similarly large (~430-530GB) full-scale matrix** - there is no k in this tested range that is
clearly "cheaper" than another once k reaches approximately 15. Only k=10 is projected
meaningfully cheaper in absolute memory (~44-52GB), at the cost of being nearly fully dense and
(per the earlier CARD-interpretation task) offering weak per-feature sequence specificity.

## Failures or memory-safety aborts

None occurred for any of the six k values at this 100-genome pilot scale. The memory-safety abort
mechanism (`build_matrix_streaming`'s per-genome RSS check) was exercised and verified by unit
tests (see `tests/test_streaming_kmer_build.py`, `tests/test_run_intermediate_k_experiment.py`),
but was never triggered by a real run in this task - this pilot size did not approach the 11GB
safety threshold for any tested k (the closest, k=31, peaked at ~7.6GB).

## Recommendation for the next validation stage

{recommendation}

## Limitations

- 100 genomes, one antibiotic, one train/test split per evaluation strategy: none of the
  comparisons above establish generalization, and the same small-pilot overfitting risk already
  documented for k=10/k=31 applies identically to every intermediate k tested here.
- Full-scale memory projections assume nnz-per-genome and column-count growth rates measured from
  THIS 100-genome cohort continue to hold at ~70-90x the genome count - a real but untested
  assumption, explicitly flagged (not verified) in `resource_estimation.py`'s own rationale text.
- No feature-selection, hashing, or out-of-core strategy was evaluated in this task - only raw
  canonical k-mer counts at each k, exactly as already used for k=10/k=31.
"""
