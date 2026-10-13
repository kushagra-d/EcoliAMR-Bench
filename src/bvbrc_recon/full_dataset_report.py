"""Renders reports/full_dataset_feasibility_and_pipeline_plan.md."""

from __future__ import annotations


def _fmt_gb(bytes_val: float) -> str:
    return f"{bytes_val / 1e9:.1f} GB"


def render_cohort_section(antibiotic: str, outcome, char) -> str:
    eligible = outcome.eligible_genome_ids()
    counts = outcome.class_counts()
    r, s = counts.get("Resistant", 0), counts.get("Susceptible", 0)
    total = r + s
    return f"""### {antibiotic.capitalize()}

- Policy A eligible genomes: **{total}** (Resistant {r} [{100*r/total:.1f}%], Susceptible {s} [{100*s/total:.1f}%])
- MLST available: {outcome.eligible_with_mlst} / {total} ({outcome.eligible_distinct_mlst_groups} distinct groups)
- Mean genome length (measured, eligible genomes): {char['mean_length']:,.0f} bp
- Mean contig count (measured): {char['mean_contigs']:.1f}
- NCBI assembly_accession populated: {char['has_assembly']} / {total} ({100*char['has_assembly']/total:.1f}%)
- NCBI biosample_accession populated: {char['has_biosample']} / {total} ({100*char['has_biosample']/total:.1f}%)
"""


def render_resource_table(resource_rows: list[dict]) -> str:
    lines = [
        "| Antibiotic (or union) | n genomes | k | Regime | Est. n_features | Est. nnz | Est. CSR memory |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in resource_rows:
        lines.append(
            f"| {row['label']} | {row['n']:,} | {row['k']} | {row['regime']} | "
            f"{row['n_features_low']:,}-{row['n_features_high']:,} | {row['nnz']:,} | {_fmt_gb(row['csr_bytes'])} |"
        )
    return "\n".join(lines)


def render_k_comparison_table(k_configs: dict) -> str:
    lines = [
        "| k | Genomes processed | n_features | nnz | Density | Extraction time | npz storage | In-memory CSR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, cfg in k_configs.items():
        lines.append(
            f"| {k} | {cfg['n_genomes_processed']}/{cfg['n_genomes_requested']} "
            f"({'complete' if cfg['complete'] else 'INCOMPLETE'}) | {cfg['n_features']:,} | "
            f"{cfg['nnz']:,} | {cfg['density']:.2%} | {cfg['extraction_seconds']:.1f}s | "
            f"{cfg['npz_storage_bytes']/1e6:.1f} MB | {cfg['in_memory_csr_bytes']/1e6:.1f} MB |"
        )
    return "\n".join(lines)


def render_model_comparison_table(model_lines: list[str]) -> str:
    header = "| k | Split | Model | Accuracy | ROC-AUC |\n|---|---|---|---|---|\n"
    return header + "\n".join(model_lines)


def render_full_dataset_report(
    analysis_date: str,
    cohort_sections: str,
    resource_table: str,
    pilot_fasta_measured_mb_per_genome: float,
    fasta_storage_table: str,
    k_comparison_table: str,
    model_comparison_table: str,
    k_comparison_narrative: str,
    pipeline_design: str,
    recommendation: str,
) -> str:
    return f"""# Full-Dataset Feasibility Assessment and Pipeline Plan

- Analysis date (UTC): {analysis_date}
- Scope: feasibility assessment and pipeline DESIGN only. No additional genome was downloaded, no
  full-scale k-mer extraction was run, no benchmark model was trained on the full dataset. The
  k=10-vs-k=31 comparison below uses ONLY the existing 100-genome ciprofloxacin pilot's already-
  downloaded assemblies.
- Every number below is labeled as either **measured** (computed directly from existing data or
  from the small pilot comparison run in this task) or **estimated/extrapolated** (projected from
  measured pilot values to full-dataset scale, with the extrapolation method stated).

## Step 2 - Policy A cohort size and resource estimates, per antibiotic

Cohort sizes and MLST coverage below reuse `final_policy_analysis.apply_policy_a` fresh against the
existing reconnaissance CSVs (read-only) - these are **measured**, not estimated, and match the
numbers already reported in `reports/final_phenotype_policy_analysis.md` exactly (same function,
same inputs).

{cohort_sections}

**Cross-antibiotic overlap (measured, not estimated)**: the same genome is frequently tested
against more than one antibiotic. The union of Policy-A-eligible genomes across all three
antibiotics is smaller than the sum of the three individual counts - a full pipeline should
retrieve each unique genome assembly ONCE and reuse it across all three antibiotics' analyses,
not redownload it per antibiotic.

### FASTA storage estimate

Measured from the existing 100-genome ciprofloxacin pilot: **{pilot_fasta_measured_mb_per_genome:.2f}
MB/genome average** (495 MB / 100 genomes). This is a reasonable **linear** extrapolation basis -
genome length is a property of the organism, not of how many other genomes are in the collection.

{fasta_storage_table}

### k-mer feature-matrix dimension and memory estimates (full dataset)

**This is the central, non-obvious finding of this feasibility assessment**: the number of matrix
COLUMNS does NOT scale linearly with genome count the way FASTA storage does. At k=10, the pilot's
100 genomes already reach ~99.9% of the entire theoretically possible canonical-k-mer space - more
genomes will add almost no new columns, only more (increasingly dense) nonzero entries. At k=31,
the opposite problem appears: the space is so large that the pilot's own 100 genomes show no
saturation at all, so the true column count at full scale cannot be measured from this pilot and
can only be bounded - the upper bound below is a crude, explicitly labeled naive linear
extrapolation, not a prediction, and is almost certainly an overestimate because more genomes of
the same species share more sequence, not less.

{resource_table}

**Both configurations are estimated to produce an impractically large in-memory sparse matrix at
full dataset scale on commodity hardware** (tens of GB at k=10, potentially hundreds of GB at
k=31) - this is explicit motivation for the k-configuration discussion in Step 3/5 below, not a
decision made here.

## Step 3 - k=10 vs. k=31 pilot comparison (measured, 100 ciprofloxacin genomes)

**Memory note (discovered empirically while running this comparison, not assumed)**: a naive
implementation that holds all 100 genomes' full per-genome k-mer counts in memory simultaneously
before merging them OOM-killed this process at k=31 (confirmed: a single genome's k=31 count table
alone uses ~500-700MB RSS; 100 simultaneously exceeded this machine's 17GB RAM). The comparison
below used a corrected, memory-bounded implementation that processes one genome at a time -
documented in `src/run_kmer_k_comparison.py` - which completed both k values successfully with a
measured peak RSS of 2.8GB (k=10) and 7.6GB (k=31), safely under an 11GB abort threshold set with
~6GB of headroom on this 17GB machine.

{k_comparison_table}

{k_comparison_narrative}

### Exploratory baseline model comparison (same existing saved splits, no tuning)

{model_comparison_table}

**Limitations of this comparison, stated explicitly:**
- 100 genomes is small enough that any apparent k=31 advantage could reflect this specific sample,
  not a general property of k=31 for this task - no cross-validation or repeated sampling was
  performed.
- Both k values were evaluated on the EXACT SAME saved random and lineage splits already used for
  the k=10 pilot's own reported results - no new split was generated, and test-set rows/labels
  were never used to choose k, select features, or tune any model (only the already-fixed model
  configurations from the existing pilot/model-comparison work were reused).
- A higher k=31 metric here is not evidence that k=31 "works better" in general - it is one
  measurement, on one small, non-lineage-representative cohort, with two models (Random Forest,
  XGBoost) whose default regularization/feature-subsampling behavior interacts with extreme
  dimensionality in ways not characterized here.
- The much larger k=31 feature count compounds the existing p >> n risk already noted in
  `reports/pilot_model_comparison.md`; nothing here resolves that risk, it is simply larger at
  k=31.

## Step 4 - Reproducible full-dataset pipeline design

{pipeline_design}

## Step 5 - Recommendation

{recommendation}
"""
