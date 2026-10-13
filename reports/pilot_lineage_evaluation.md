# Pilot Lineage-Aware Evaluation (Ciprofloxacin)

- Analysis date (UTC): 2026-10-09T18:50:18.991501+00:00
- Antibiotic: ciprofloxacin
- Random seed (grouping + model fitting): 20261009
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

- 99 / 100 pilot genomes have a non-missing `genome_mlst` value.
- 1 genome(s) have a missing MLST value: ['562.58626'].
  Each was assigned its own singleton group - **not** pooled into a shared "missing" lineage.
- 44 distinct groups total among the 100 pilot genomes (including the
  singleton group(s) for any missing-MLST genome).
- No scheme anomaly detected: every genome with an MLST value uses the same scheme (ecoli_achtman_4).

**Group size distribution:**

| Group size (genomes) | Number of groups of that size |
|---|---|
| 16 | 1 |
| 10 | 1 |
| 9 | 1 |
| 8 | 1 |
| 7 | 1 |
| 3 | 1 |
| 2 | 9 |
| 1 | 29 |

## Group and class distributions (lineage-aware split)

- Train: 71 genomes across 30 MLST groups
- Test: 29 genomes across 14 MLST groups
- Requested test fraction: 30%; actual test fraction: 29.0%
- Train class counts: {'Resistant': 18, 'Susceptible': 53}
- Test class counts: {'Resistant': 7, 'Susceptible': 22}
- Warnings: none


## Random-split baseline (reproduced)

Reused the exact existing Logistic Regression configuration, k-mer matrix, labels, and seed from
`reports/pilot_ciprofloxacin_report.md` - these numbers should match that report exactly; any
difference would indicate a non-determinism bug.

### Random split (ordinary stratified train/test split - NOT lineage-independent)

- Train: 70 genomes {'Susceptible': 52, 'Resistant': 18}
- Test: 30 genomes {'Susceptible': 23, 'Resistant': 7}
- Accuracy: 0.9
- ROC-AUC: 0.9627329192546584
- Average precision: 0.9200680272108843
- Recall per class: {'Susceptible': 1.0, 'Resistant': 0.5714285714285714}
- Confusion matrix `[[TN, FP], [FN, TP]]` (0=Susceptible, 1=Resistant): [[23, 0], [3, 4]]
- Warnings: none


## Lineage-aware evaluation

### Lineage-aware split (MLST-group-held-out)

- Train: 71 genomes {'Susceptible': 53, 'Resistant': 18}
- Test: 29 genomes {'Susceptible': 22, 'Resistant': 7}
- Accuracy: 0.7931034482758621
- ROC-AUC: 0.8896103896103896
- Average precision: 0.7398268398268397
- Recall per class: {'Susceptible': 0.7727272727272727, 'Resistant': 0.8571428571428571}
- Confusion matrix `[[TN, FP], [FN, TP]]` (0=Susceptible, 1=Resistant): [[17, 5], [1, 6]]
- Warnings: none


## Comparison and interpretation

| Metric | Random split | Lineage-aware split |
|---|---|---|
| Train / Test size | 70 / 30 | 71 / 29 |
| Accuracy | 0.9 | 0.7931034482758621 |
| ROC-AUC | 0.9627329192546584 | 0.8896103896103896 |
| Average precision | 0.9200680272108843 | 0.7398268398268397 |
| Recall (Resistant) | 0.5714285714285714 | 0.8571428571428571 |
| Recall (Susceptible) | 1.0 | 0.7727272727272727 |

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

## Limitations and next steps

- This remains a single ~100-genome pilot for one antibiotic; neither evaluation here should be
  read as the three-antibiotic benchmark's expected performance.
- The lineage-aware test set (29
  genomes) is smaller and differently composed than the random-split test set
  (30 genomes) - metric differences reflect both the lineage-awareness itself
  and this composition difference, which cannot be fully disentangled with a single split of this
  dataset size.
- MLST-based grouping does not certify complete genomic independence (see above) - a stricter
  grouping (e.g. cgMLST or phylogenetic distance) is a reasonable next step before the full
  three-antibiotic benchmark, not implemented here.
- Next steps naturally suggested by this task but explicitly out of scope here: extending this
  comparison to Ampicillin and Gentamicin, repeating the lineage-aware split across multiple seeds
  to characterize variance (a single split, as required here, cannot distinguish seed-sensitivity
  from a genuine effect), and revisiting genome-level QC thresholds before any larger retrieval.
