# Phenotype Conflict Analysis and Curation-Policy Comparison

- Analysis date (UTC): 2026-10-09T17:43:09.241949+00:00
- Source CSVs: `data/processed/{antibiotic}_amr_metadata.csv` (read-only input; not modified by this analysis)

This report performs no label assignment and selects no modelling cohort. Every count below is
computed directly from the existing reconnaissance CSVs under `data/processed/` (read-only; those
files and the original `reports/bvbrc_feasibility_report.md` were not modified). Genomes are
grouped by `genome_id` loaded explicitly as a string (see Task C for why this matters).

## Per-antibiotic results

### Ciprofloxacin

**Records:** 15777 | **Unique genomes:** 10904

#### Task A - Genome-level pattern audit

| Genome pattern (recognized categories only) | Genomes | % of unique genomes |
|---|---|---|
| `no_recognized_interpretation` | 3469 | 31.8% |
| `only_resistant` | 1840 | 16.9% |
| `only_susceptible` | 5457 | 50.0% |
| `only_intermediate` | 110 | 1.0% |
| `resistant_and_susceptible` | 18 | 0.2% |
| `resistant_and_intermediate_no_susceptible` | 6 | 0.1% |
| `susceptible_and_intermediate_no_resistant` | 4 | 0.0% |
| `all_three` | 0 | 0.0% |
| **Total (partition check)** | **10904** | should equal 10904 |

**Record-vs-genome multiplicity:**

| Record-multiplicity class | Genomes |
|---|---|
| `single` | 6341 |
| `exact_duplicate` | 28 |
| `repeated_concordant` | 4507 |
| `conflicting` | 28 |
| **Total (partition check)** | **10904** (should equal 10904) |

**Conflict-count reconciliation with the prior reconnaissance report:**

The prior reconnaissance report counted **3241** genomes as "conflicting" using a broader definition (any two records disagreeing on categorized phenotype, where "missing"/"other" values count as distinct categories). Re-deriving that same broad count from the current CSV confirms it was computed correctly as *defined* - it was not an error, but a different and broader question than this audit asks. Restricting to genuine Resistant/Susceptible/Intermediate disagreement (ignoring missing/other) gives **28** genomes - the number used throughout the rest of this report as "conflicting".

**Metadata heterogeneity among conflicting genomes:**

Among the **28** genomes with a genuine categorical conflict (Resistant+Susceptible, Resistant+Intermediate-no-Susceptible, Susceptible+Intermediate-no-Resistant, or all three):

| Metadata field | Differs across the genome's records | Same across the genome's records |
|---|---|---|
| `testing_standard` | 7 (25.0%) | 21 |
| `testing_standard_year` | 7 (25.0%) | 21 |
| `laboratory_typing_method` | 28 (100.0%) | 0 |
| `laboratory_typing_platform` | 1 (3.6%) | 27 |
| `measurement_unit` | 28 (100.0%) | 0 |

*Co-occurrence only, not causation: a differing field alongside a conflicting call does not establish that the field difference caused the disagreement - it may equally reflect genuine biological variability (e.g. a true phenotype change between sampling events) or an unrelated data-entry difference.*

#### Task B - Policy 1: strict categorical agreement

Resistant+Intermediate (and Susceptible+Intermediate) genomes ARE eligible under this policy -
Intermediate does not block inclusion and is never mapped into the binary class; it is simply
not counted against the genome when only one of Resistant/Susceptible is present.

| Outcome | Genomes | % of unique genomes |
|---|---|---|
| eligible_resistant | 1846 | 16.9% |
| eligible_susceptible | 5461 | 50.1% |
| excluded_no_recognized_binary_interpretation | 3579 | 32.8% |
| excluded_categorical_disagreement | 18 | 0.2% |
| **Total (partition check)** | **10904** | should equal 10904 |

Class proportions among eligible genomes: Resistant 1846/7307 (25.3%), Susceptible 5461/7307 (74.7%).

- Eligible genomes with more than one AST record for this antibiotic: 4430
  - of those, differing `testing_standard` despite categorical agreement: 1897
  - of those, differing `testing_standard_year` despite categorical agreement: 34
  - of those, differing `laboratory_typing_method` despite categorical agreement: 1575
  - of those, differing `laboratory_typing_platform` despite categorical agreement: 458
  - of those, differing `measurement_unit` despite categorical agreement: 4430
- Eligible genomes that also carry an Intermediate record alongside their binary call: 10

#### Task B - Policy 2: require a single binary interpretation

Any Intermediate co-occurrence, or any Resistant+Susceptible disagreement, excludes the genome
entirely under this stricter policy.

| Outcome | Genomes | % of unique genomes |
|---|---|---|
| eligible_resistant | 1840 | 16.9% |
| eligible_susceptible | 5457 | 50.0% |
| excluded_intermediate_present_with_binary | 120 | 1.1% |
| excluded_both_classes_present | 18 | 0.2% |
| excluded_no_binary_interpretation | 3469 | 31.8% |
| **Total (partition check)** | **10904** | should equal 10904 |

Class proportions among eligible genomes: Resistant 1840/7297 (25.2%), Susceptible 5457/7297 (74.8%).

- Eligible genomes with more than one AST record for this antibiotic: 4420
  - of those, differing `testing_standard` despite categorical agreement: 1893
  - of those, differing `testing_standard_year` despite categorical agreement: 30
  - of those, differing `laboratory_typing_method` despite categorical agreement: 1565
  - of those, differing `laboratory_typing_platform` despite categorical agreement: 458
  - of those, differing `measurement_unit` despite categorical agreement: 4420
- Eligible genomes that also carry an Intermediate record alongside their binary call: 0

#### Task B - Policy 3: standard-stratified descriptive analysis (no cohort implied)

| Raw `testing_standard` value | Records | Resistant | Susceptible | Intermediate | Missing | Other | Genomes represented | ... globally conflicting |
|---|---|---|---|---|---|---|---|---|
| `<missing/blank>` | 7248 | 599 | 3313 | 23 | 3313 | 0 | 4881 | 21 |
| `EUCAST` | 4649 | 934 | 3014 | 79 | 622 | 0 | 4466 | 1 |
| `CLSI` | 3710 | 438 | 181 | 10 | 3081 | 0 | 3359 | 1 |
| `clsi` | 48 | 25 | 22 | 1 | 0 | 0 | 36 | 6 |
| `EUCAST, CLSI` | 43 | 30 | 9 | 4 | 0 | 0 | 43 | 0 |
| `CLSI, EUCAST` | 35 | 35 | 0 | 0 | 0 | 0 | 35 | 0 |
| `eucast` | 24 | 3 | 18 | 3 | 0 | 0 | 24 | 6 |
| `EUCAST,CLSI` | 12 | 0 | 0 | 0 | 12 | 0 | 12 | 0 |
| `EUCAST and CLSI` | 8 | 2 | 2 | 0 | 4 | 0 | 4 | 0 |

*"...globally conflicting" counts, for genomes represented in this testing-standard stratum, how many are globally conflicting across ALL of their records for this antibiotic (not just the records in this stratum). A genome with records in more than one stratum is counted in each. This is a cross-tabulation for descriptive purposes only - it does not imply any stratum is or is not suitable for a primary modelling cohort.*

### Ampicillin

**Records:** 11896 | **Unique genomes:** 7383

#### Task A - Genome-level pattern audit

| Genome pattern (recognized categories only) | Genomes | % of unique genomes |
|---|---|---|
| `no_recognized_interpretation` | 878 | 11.9% |
| `only_resistant` | 3658 | 49.5% |
| `only_susceptible` | 2827 | 38.3% |
| `only_intermediate` | 8 | 0.1% |
| `resistant_and_susceptible` | 7 | 0.1% |
| `resistant_and_intermediate_no_susceptible` | 1 | 0.0% |
| `susceptible_and_intermediate_no_resistant` | 4 | 0.1% |
| `all_three` | 0 | 0.0% |
| **Total (partition check)** | **7383** | should equal 7383 |

**Record-vs-genome multiplicity:**

| Record-multiplicity class | Genomes |
|---|---|
| `single` | 2993 |
| `exact_duplicate` | 16 |
| `repeated_concordant` | 4362 |
| `conflicting` | 12 |
| **Total (partition check)** | **7383** (should equal 7383) |

**Conflict-count reconciliation with the prior reconnaissance report:**

The prior reconnaissance report counted **4149** genomes as "conflicting" using a broader definition (any two records disagreeing on categorized phenotype, where "missing"/"other" values count as distinct categories). Re-deriving that same broad count from the current CSV confirms it was computed correctly as *defined* - it was not an error, but a different and broader question than this audit asks. Restricting to genuine Resistant/Susceptible/Intermediate disagreement (ignoring missing/other) gives **12** genomes - the number used throughout the rest of this report as "conflicting".

**Metadata heterogeneity among conflicting genomes:**

Among the **12** genomes with a genuine categorical conflict (Resistant+Susceptible, Resistant+Intermediate-no-Susceptible, Susceptible+Intermediate-no-Resistant, or all three):

| Metadata field | Differs across the genome's records | Same across the genome's records |
|---|---|---|
| `testing_standard` | 11 (91.7%) | 1 |
| `testing_standard_year` | 11 (91.7%) | 1 |
| `laboratory_typing_method` | 12 (100.0%) | 0 |
| `laboratory_typing_platform` | 1 (8.3%) | 11 |
| `measurement_unit` | 12 (100.0%) | 0 |

*Co-occurrence only, not causation: a differing field alongside a conflicting call does not establish that the field difference caused the disagreement - it may equally reflect genuine biological variability (e.g. a true phenotype change between sampling events) or an unrelated data-entry difference.*

#### Task B - Policy 1: strict categorical agreement

Resistant+Intermediate (and Susceptible+Intermediate) genomes ARE eligible under this policy -
Intermediate does not block inclusion and is never mapped into the binary class; it is simply
not counted against the genome when only one of Resistant/Susceptible is present.

| Outcome | Genomes | % of unique genomes |
|---|---|---|
| eligible_resistant | 3659 | 49.6% |
| eligible_susceptible | 2831 | 38.3% |
| excluded_categorical_disagreement | 7 | 0.1% |
| excluded_no_recognized_binary_interpretation | 886 | 12.0% |
| **Total (partition check)** | **7383** | should equal 7383 |

Class proportions among eligible genomes: Resistant 3659/6490 (56.4%), Susceptible 2831/6490 (43.6%).

- Eligible genomes with more than one AST record for this antibiotic: 4337
  - of those, differing `testing_standard` despite categorical agreement: 2878
  - of those, differing `testing_standard_year` despite categorical agreement: 959
  - of those, differing `laboratory_typing_method` despite categorical agreement: 1423
  - of those, differing `laboratory_typing_platform` despite categorical agreement: 461
  - of those, differing `measurement_unit` despite categorical agreement: 4336
- Eligible genomes that also carry an Intermediate record alongside their binary call: 5

#### Task B - Policy 2: require a single binary interpretation

Any Intermediate co-occurrence, or any Resistant+Susceptible disagreement, excludes the genome
entirely under this stricter policy.

| Outcome | Genomes | % of unique genomes |
|---|---|---|
| eligible_resistant | 3658 | 49.5% |
| eligible_susceptible | 2827 | 38.3% |
| excluded_both_classes_present | 7 | 0.1% |
| excluded_intermediate_present_with_binary | 13 | 0.2% |
| excluded_no_binary_interpretation | 878 | 11.9% |
| **Total (partition check)** | **7383** | should equal 7383 |

Class proportions among eligible genomes: Resistant 3658/6485 (56.4%), Susceptible 2827/6485 (43.6%).

- Eligible genomes with more than one AST record for this antibiotic: 4332
  - of those, differing `testing_standard` despite categorical agreement: 2873
  - of those, differing `testing_standard_year` despite categorical agreement: 954
  - of those, differing `laboratory_typing_method` despite categorical agreement: 1418
  - of those, differing `laboratory_typing_platform` despite categorical agreement: 461
  - of those, differing `measurement_unit` despite categorical agreement: 4331
- Eligible genomes that also carry an Intermediate record alongside their binary call: 0

#### Task B - Policy 3: standard-stratified descriptive analysis (no cohort implied)

| Raw `testing_standard` value | Records | Resistant | Susceptible | Intermediate | Missing | Other | Genomes represented | ... globally conflicting |
|---|---|---|---|---|---|---|---|---|
| `<missing/blank>` | 5887 | 731 | 1019 | 3 | 4134 | 0 | 4613 | 0 |
| `EUCAST` | 3864 | 1859 | 1386 | 0 | 619 | 0 | 3680 | 0 |
| `CLSI` | 1059 | 594 | 48 | 5 | 412 | 0 | 892 | 1 |
| `eucast` | 952 | 520 | 432 | 0 | 0 | 0 | 952 | 11 |
| `EUCAST, CLSI` | 43 | 26 | 17 | 0 | 0 | 0 | 43 | 0 |
| `clsi` | 36 | 27 | 4 | 5 | 0 | 0 | 36 | 11 |
| `CLSI, EUCAST` | 35 | 35 | 0 | 0 | 0 | 0 | 35 | 0 |
| `EUCAST,CLSI` | 12 | 0 | 0 | 0 | 12 | 0 | 12 | 0 |
| `EUCAST and CLSI` | 8 | 4 | 0 | 0 | 4 | 0 | 4 | 0 |

*"...globally conflicting" counts, for genomes represented in this testing-standard stratum, how many are globally conflicting across ALL of their records for this antibiotic (not just the records in this stratum). A genome with records in more than one stratum is counted in each. This is a cross-tabulation for descriptive purposes only - it does not imply any stratum is or is not suitable for a primary modelling cohort.*

### Gentamicin

**Records:** 13317 | **Unique genomes:** 11175

#### Task A - Genome-level pattern audit

| Genome pattern (recognized categories only) | Genomes | % of unique genomes |
|---|---|---|
| `no_recognized_interpretation` | 3669 | 32.8% |
| `only_resistant` | 860 | 7.7% |
| `only_susceptible` | 6478 | 58.0% |
| `only_intermediate` | 166 | 1.5% |
| `resistant_and_susceptible` | 0 | 0.0% |
| `resistant_and_intermediate_no_susceptible` | 2 | 0.0% |
| `susceptible_and_intermediate_no_resistant` | 0 | 0.0% |
| `all_three` | 0 | 0.0% |
| **Total (partition check)** | **11175** | should equal 11175 |

**Record-vs-genome multiplicity:**

| Record-multiplicity class | Genomes |
|---|---|
| `single` | 9343 |
| `exact_duplicate` | 19 |
| `repeated_concordant` | 1811 |
| `conflicting` | 2 |
| **Total (partition check)** | **11175** (should equal 11175) |

**Conflict-count reconciliation with the prior reconnaissance report:**

The prior reconnaissance report counted **1570** genomes as "conflicting" using a broader definition (any two records disagreeing on categorized phenotype, where "missing"/"other" values count as distinct categories). Re-deriving that same broad count from the current CSV confirms it was computed correctly as *defined* - it was not an error, but a different and broader question than this audit asks. Restricting to genuine Resistant/Susceptible/Intermediate disagreement (ignoring missing/other) gives **2** genomes - the number used throughout the rest of this report as "conflicting".

**Metadata heterogeneity among conflicting genomes:**

Among the **2** genomes with a genuine categorical conflict (Resistant+Susceptible, Resistant+Intermediate-no-Susceptible, Susceptible+Intermediate-no-Resistant, or all three):

| Metadata field | Differs across the genome's records | Same across the genome's records |
|---|---|---|
| `testing_standard` | 2 (100.0%) | 0 |
| `testing_standard_year` | 2 (100.0%) | 0 |
| `laboratory_typing_method` | 2 (100.0%) | 0 |
| `laboratory_typing_platform` | 0 (0.0%) | 2 |
| `measurement_unit` | 2 (100.0%) | 0 |

*Co-occurrence only, not causation: a differing field alongside a conflicting call does not establish that the field difference caused the disagreement - it may equally reflect genuine biological variability (e.g. a true phenotype change between sampling events) or an unrelated data-entry difference.*

#### Task B - Policy 1: strict categorical agreement

Resistant+Intermediate (and Susceptible+Intermediate) genomes ARE eligible under this policy -
Intermediate does not block inclusion and is never mapped into the binary class; it is simply
not counted against the genome when only one of Resistant/Susceptible is present.

| Outcome | Genomes | % of unique genomes |
|---|---|---|
| eligible_resistant | 862 | 7.7% |
| eligible_susceptible | 6478 | 58.0% |
| excluded_no_recognized_binary_interpretation | 3835 | 34.3% |
| **Total (partition check)** | **11175** | should equal 11175 |

Class proportions among eligible genomes: Resistant 862/7340 (11.7%), Susceptible 6478/7340 (88.3%).

- Eligible genomes with more than one AST record for this antibiotic: 1761
  - of those, differing `testing_standard` despite categorical agreement: 37
  - of those, differing `testing_standard_year` despite categorical agreement: 37
  - of those, differing `laboratory_typing_method` despite categorical agreement: 1578
  - of those, differing `laboratory_typing_platform` despite categorical agreement: 461
  - of those, differing `measurement_unit` despite categorical agreement: 1760
- Eligible genomes that also carry an Intermediate record alongside their binary call: 2

#### Task B - Policy 2: require a single binary interpretation

Any Intermediate co-occurrence, or any Resistant+Susceptible disagreement, excludes the genome
entirely under this stricter policy.

| Outcome | Genomes | % of unique genomes |
|---|---|---|
| eligible_resistant | 860 | 7.7% |
| eligible_susceptible | 6478 | 58.0% |
| excluded_intermediate_present_with_binary | 168 | 1.5% |
| excluded_no_binary_interpretation | 3669 | 32.8% |
| **Total (partition check)** | **11175** | should equal 11175 |

Class proportions among eligible genomes: Resistant 860/7338 (11.7%), Susceptible 6478/7338 (88.3%).

- Eligible genomes with more than one AST record for this antibiotic: 1759
  - of those, differing `testing_standard` despite categorical agreement: 35
  - of those, differing `testing_standard_year` despite categorical agreement: 35
  - of those, differing `laboratory_typing_method` despite categorical agreement: 1576
  - of those, differing `laboratory_typing_platform` despite categorical agreement: 461
  - of those, differing `measurement_unit` despite categorical agreement: 1758
- Eligible genomes that also carry an Intermediate record alongside their binary call: 0

#### Task B - Policy 3: standard-stratified descriptive analysis (no cohort implied)

| Raw `testing_standard` value | Records | Resistant | Susceptible | Intermediate | Missing | Other | Genomes represented | ... globally conflicting |
|---|---|---|---|---|---|---|---|---|
| `EUCAST` | 4979 | 489 | 3590 | 96 | 804 | 0 | 4795 | 0 |
| `<missing/blank>` | 4467 | 154 | 2631 | 27 | 1655 | 0 | 2913 | 2 |
| `CLSI` | 3779 | 215 | 406 | 44 | 3114 | 0 | 3428 | 0 |
| `clsi` | 48 | 13 | 33 | 2 | 0 | 0 | 36 | 2 |
| `eucast` | 24 | 1 | 23 | 0 | 0 | 0 | 24 | 0 |
| `EUCAST,CLSI` | 12 | 0 | 0 | 0 | 12 | 0 | 12 | 0 |
| `EUCAST and CLSI` | 8 | 0 | 4 | 0 | 4 | 0 | 4 | 0 |

*"...globally conflicting" counts, for genomes represented in this testing-standard stratum, how many are globally conflicting across ALL of their records for this antibiotic (not just the records in this stratum). A genome with records in more than one stratum is counted in each. This is a cross-tabulation for descriptive purposes only - it does not imply any stratum is or is not suitable for a primary modelling cohort.*


## Task C - Decisions that still require human judgment

- **Combined/mixed testing_standard values** (e.g. `"EUCAST, CLSI"`, `"CLSI, EUCAST"`,
  `"EUCAST and CLSI"`) were observed as distinct raw strings. Whether these represent a single
  concordant cross-standard confirmation, an ambiguous/unreliable entry, or something else is not
  resolved here - no rule has been invented to merge, split, or discard them.
- **Whether testing_standard_year must match** between records used to confirm a phenotype call
  is unresolved; breakpoints are revised over time and a multi-year gap between concordant calls
  may or may not be acceptable depending on the antibiotic and standard.
- **Whether laboratory_typing_method/platform need to be restricted** (e.g. requiring broth
  dilution with a reported MIC over disk diffusion) has not been decided; this report only
  quantifies how often these fields differ among agreeing or conflicting records.
- **How Intermediate results should be handled** beyond "never silently mapped to Susceptible" -
  whether Intermediate genomes should form a third model class, be excluded, or be revisited with
  updated breakpoints is left open; Policies 1 and 2 above show two different consequences of two
  different stances, not a recommendation.
- **How to treat genomes that agree categorically but differ in MIC/disk measurement or metadata**
  (the `repeated_concordant` bucket) - whether to keep one representative record, average/aggregate
  measurements, or keep all records as-is for provenance is unresolved.
- **How to define an isolate** when multiple genome_ids or AST records may refer to related or
  repeated biological material (e.g. resequencing, repeat submissions) is unresolved. This audit
  also surfaced a concrete, separate tooling risk worth carrying into that decision: naively
  reading these CSVs with pandas auto-inferring `genome_id` as a float silently merges distinct
  genome_id strings in this dataset (confirmed: fewer unique float-coerced ids than unique string
  ids for all three antibiotics) - any future isolate-definition work must load `genome_id` as a
  string, as this module does, not rely on default CSV type inference.

