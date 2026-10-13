"""Renders reports/pilot_lineage_evaluation.md."""

from __future__ import annotations


def _model_section(title: str, result) -> str:
    if result is None:
        return f"### {title}\n\nNot computed (see blocker above)."
    return f"""### {title}

- Train: {result.n_train} genomes {result.train_class_counts}
- Test: {result.n_test} genomes {result.test_class_counts}
- Accuracy: {result.accuracy}
- ROC-AUC: {result.roc_auc}
- Average precision: {result.average_precision}
- Recall per class: {result.recall_per_class}
- Confusion matrix `[[TN, FP], [FN, TP]]` (0=Susceptible, 1=Resistant): {result.confusion_matrix}
- Warnings: {result.warnings or "none"}
"""


def render_lineage_report(
    antibiotic: str,
    seed: int,
    test_size: float,
    analysis_date: str,
    n_genomes: int,
    assignment,
    lineage_split,
    random_result,
    lineage_result,
) -> str:
    n_with_mlst = n_genomes - len(assignment.missing_mlst_genome_ids)
    anomaly_note = (
        f"**{len(assignment.anomalous_scheme_genome_ids)} genome(s) have an MLST scheme "
        f"different from the majority scheme ({assignment.majority_scheme}): "
        f"{assignment.anomalous_scheme_genome_ids}** - flagged, not corrected or excluded."
        if assignment.anomalous_scheme_genome_ids
        else f"No scheme anomaly detected: every genome with an MLST value uses the same scheme "
        f"({assignment.majority_scheme})."
    )

    group_size_table = "\n".join(
        f"| {size} | {list(assignment.group_sizes.values()).count(size)} |"
        for size in sorted(set(assignment.group_sizes.values()), reverse=True)
    )

    if lineage_split.feasible:
        lineage_split_section = f"""
- Train: {len(lineage_split.train_genome_ids)} genomes across {lineage_split.train_group_count} MLST groups
- Test: {len(lineage_split.test_genome_ids)} genomes across {lineage_split.test_group_count} MLST groups
- Requested test fraction: {test_size:.0%}; actual test fraction: {lineage_split.actual_test_fraction:.1%}
- Train class counts: {lineage_split.train_class_counts}
- Test class counts: {lineage_split.test_class_counts}
- Warnings: {lineage_split.warnings or "none"}
"""
    else:
        lineage_split_section = f"\n**BLOCKED: {lineage_split.blocker}**\n\nNo lineage-aware model was trained - a random split was NOT substituted in its place.\n"

    comparison_section = ""
    if lineage_result is not None:
        comparison_section = f"""
## Comparison and interpretation

| Metric | Random split | Lineage-aware split |
|---|---|---|
| Train / Test size | {random_result.n_train} / {random_result.n_test} | {lineage_result.n_train} / {lineage_result.n_test} |
| Accuracy | {random_result.accuracy} | {lineage_result.accuracy} |
| ROC-AUC | {random_result.roc_auc} | {lineage_result.roc_auc} |
| Average precision | {random_result.average_precision} | {lineage_result.average_precision} |
| Recall (Resistant) | {random_result.recall_per_class.get('Resistant')} | {lineage_result.recall_per_class.get('Resistant')} |
| Recall (Susceptible) | {random_result.recall_per_class.get('Susceptible')} | {lineage_result.recall_per_class.get('Susceptible')} |

**These two evaluations do not use the same test set composition** (different genomes, different
group structure, different class balance) - this is **not a perfectly paired comparison**, and any
difference in the numbers above is preliminary evidence that the model's apparent performance is
sensitive to how the test set is constructed, not proof that genetic lineage/relatedness alone
caused the difference. A random split can place near-identical or closely related genomes (e.g.
same MLST group, or even near-duplicate submissions) on both sides of the split, which can inflate
apparent performance relative to a split that keeps such genomes together; the lineage-aware split
is designed to remove that specific source of optimism, but it is evaluated on a different,
smaller, and differently-composed test set, so the two numbers are informative together, not as a
strict before/after measurement of one isolated variable.
"""

    return f"""# Pilot Lineage-Aware Evaluation ({antibiotic.capitalize()})

- Analysis date (UTC): {analysis_date}
- Antibiotic: {antibiotic}
- Random seed (grouping + model fitting): {seed}
- Reuses the EXISTING 100-genome pilot cohort and k-mer feature matrix - no genome was downloaded
  and no k-mer was recomputed for this task.

## Split methodology and grouping field

Grouping field: `genome_mlst` (from the existing reconnaissance metadata, joined to the pilot's
100 genome_ids - never re-downloaded). A genome's group key is its MLST value; a genome with a
**missing** MLST value is given its **own singleton group** (its genome_id) rather than being
merged into one shared "unknown" lineage - see Caution below for why this matters.

Split mechanism: `sklearn.model_selection.GroupShuffleSplit` (a single train/test split that keeps
every group entirely on one side), with group-integrity verified explicitly after the split (not
only trusted from the library's documented behavior).

**MLST grouping is a practical proxy for relatedness, not a guarantee of complete genomic
independence** between partitions: a 7-gene MLST scheme can place genomes with different sequence
types in the same partition even if they share recent horizontal transfers, plasmids, or other
close relationships that MLST does not capture. This split reduces one specific, well-understood
leakage pathway (near-identical genomes of the same sequence type appearing on both sides); it does
not certify general genomic independence.

## MLST coverage and limitations

- {n_with_mlst} / {n_genomes} pilot genomes have a non-missing `genome_mlst` value.
- {len(assignment.missing_mlst_genome_ids)} genome(s) have a missing MLST value: {assignment.missing_mlst_genome_ids or "none"}.
  Each was assigned its own singleton group - **not** pooled into a shared "missing" lineage.
- {assignment.n_groups} distinct groups total among the 100 pilot genomes (including the
  singleton group(s) for any missing-MLST genome).
- {anomaly_note}

**Group size distribution:**

| Group size (genomes) | Number of groups of that size |
|---|---|
{group_size_table}

## Group and class distributions (lineage-aware split)
{lineage_split_section}

## Random-split baseline (reproduced)

Reused the exact existing Logistic Regression configuration, k-mer matrix, labels, and seed from
`reports/pilot_{antibiotic}_report.md` - these numbers should match that report exactly; any
difference would indicate a non-determinism bug.

{_model_section("Random split (ordinary stratified train/test split - NOT lineage-independent)", random_result)}

## Lineage-aware evaluation

{_model_section("Lineage-aware split (MLST-group-held-out)", lineage_result)}
{comparison_section}
## Limitations and next steps

- This remains a single ~100-genome pilot for one antibiotic; neither evaluation here should be
  read as the three-antibiotic benchmark's expected performance.
- The lineage-aware test set ({len(lineage_split.test_genome_ids) if lineage_split.feasible else 0}
  genomes) is smaller and differently composed than the random-split test set
  ({random_result.n_test} genomes) - metric differences reflect both the lineage-awareness itself
  and this composition difference, which cannot be fully disentangled with a single split of this
  dataset size.
- MLST-based grouping does not certify complete genomic independence (see above) - a stricter
  grouping (e.g. cgMLST or phylogenetic distance) is a reasonable next step before the full
  three-antibiotic benchmark, not implemented here.
- Next steps naturally suggested by this task but explicitly out of scope here: extending this
  comparison to Ampicillin and Gentamicin, repeating the lineage-aware split across multiple seeds
  to characterize variance (a single split, as required here, cannot distinguish seed-sensitivity
  from a genuine effect), and revisiting genome-level QC thresholds before any larger retrieval.
"""
