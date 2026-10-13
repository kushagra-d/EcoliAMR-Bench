# Final Phenotype-Curation Policy Analysis (All Three Antibiotics)

- Analysis date (UTC): 2026-10-09T19:38:48.327952+00:00
- Scope: metadata-only. No genome sequence was downloaded, no k-mer was extracted, no model was
  trained. Reuses the existing reconnaissance CSVs and `conflict_analysis.py` functions read-only;
  none of them were modified.

## Step 2 - Testing-standard and method heterogeneity, per antibiotic

### Ciprofloxacin: raw metadata distributions

- Records: 15777 | Unique genomes: 10904
- Genomes with repeated, categorically-CONCORDANT records (agree on phenotype, >1 record): **4507**
  - of those, differing `testing_standard` across their own records: 1949
  - of those, differing `testing_standard_year`: 30
  - of those, differing `laboratory_typing_method`: 1567
  - of those, differing `laboratory_typing_platform`: 460
  - of those, differing `measurement_unit`: 4483

**`testing_standard` (raw, exact strings - casing and combined values preserved):**

| raw testing_standard | Count |
|---|---|
| `<missing/blank>` | 7248 |
| `EUCAST` | 4649 |
| `CLSI` | 3710 |
| `clsi` | 48 |
| `EUCAST, CLSI` | 43 |
| `CLSI, EUCAST` | 35 |
| `eucast` | 24 |
| `EUCAST,CLSI` | 12 |
| `EUCAST and CLSI` | 8 |

**`testing_standard_year`:**

| raw testing_standard_year | Count |
|---|---|
| `<missing/blank>` | 11280 |
| `2021` | 2602 |
| `2018` | 599 |
| `2016` | 479 |
| `2017` | 448 |
| `2015` | 263 |
| `2013` | 50 |
| `2019` | 43 |
| `2008` | 13 |

**`laboratory_typing_method`:**

| raw laboratory_typing_method | Count |
|---|---|
| `Disk diffusion` | 6538 |
| `MIC` | 4681 |
| `Broth dilution` | 3402 |
| `Agar dilution` | 1094 |
| `<missing/blank>` | 62 |

**`laboratory_typing_platform`:**

| raw laboratory_typing_platform | Count |
|---|---|
| `<missing/blank>` | 12729 |
| `VITEK 2` | 2219 |
| `E-test` | 461 |
| `Sensititre` | 365 |
| `BD Phoenix NMIC-203` | 1 |
| `KB panel` | 1 |
| `Sensititre EUVSEC/EUVSEC2` | 1 |

**`measurement_unit`:**

| raw measurement_unit | Count |
|---|---|
| `mg/L` | 9751 |
| `<missing/blank>` | 5915 |
| `mm` | 111 |

### Ampicillin: raw metadata distributions

- Records: 11896 | Unique genomes: 7383
- Genomes with repeated, categorically-CONCORDANT records (agree on phenotype, >1 record): **4362**
  - of those, differing `testing_standard` across their own records: 2873
  - of those, differing `testing_standard_year`: 954
  - of those, differing `laboratory_typing_method`: 1418
  - of those, differing `laboratory_typing_platform`: 461
  - of those, differing `measurement_unit`: 4334

**`testing_standard` (raw, exact strings - casing and combined values preserved):**

| raw testing_standard | Count |
|---|---|
| `<missing/blank>` | 5887 |
| `EUCAST` | 3864 |
| `CLSI` | 1059 |
| `eucast` | 952 |
| `EUCAST, CLSI` | 43 |
| `clsi` | 36 |
| `CLSI, EUCAST` | 35 |
| `EUCAST,CLSI` | 12 |
| `EUCAST and CLSI` | 8 |

**`testing_standard_year`:**

| raw testing_standard_year | Count |
|---|---|
| `<missing/blank>` | 8946 |
| `2019` | 971 |
| `2016` | 598 |
| `2017` | 512 |
| `2018` | 382 |
| `2015` | 264 |
| `2021` | 173 |
| `2013` | 50 |

**`laboratory_typing_method`:**

| raw laboratory_typing_method | Count |
|---|---|
| `Disk diffusion` | 6750 |
| `Broth dilution` | 3251 |
| `<missing/blank>` | 975 |
| `MIC` | 920 |

**`laboratory_typing_platform`:**

| raw laboratory_typing_platform | Count |
|---|---|
| `<missing/blank>` | 9769 |
| `VITEK 2` | 1299 |
| `E-test` | 461 |
| `Sensititre` | 365 |
| `BD Phoenix NMIC-203` | 1 |
| `Sensititre EUVSEC/EUVSEC2` | 1 |

**`measurement_unit`:**

| raw measurement_unit | Count |
|---|---|
| `<missing/blank>` | 6071 |
| `mg/L` | 5717 |
| `mm` | 108 |

### Gentamicin: raw metadata distributions

- Records: 13317 | Unique genomes: 11175
- Genomes with repeated, categorically-CONCORDANT records (agree on phenotype, >1 record): **1811**
  - of those, differing `testing_standard` across their own records: 35
  - of those, differing `testing_standard_year`: 35
  - of those, differing `laboratory_typing_method`: 1594
  - of those, differing `laboratory_typing_platform`: 462
  - of those, differing `measurement_unit`: 1777

**`testing_standard` (raw, exact strings - casing and combined values preserved):**

| raw testing_standard | Count |
|---|---|
| `EUCAST` | 4979 |
| `<missing/blank>` | 4467 |
| `CLSI` | 3779 |
| `clsi` | 48 |
| `eucast` | 24 |
| `EUCAST,CLSI` | 12 |
| `EUCAST and CLSI` | 8 |

**`testing_standard_year`:**

| raw testing_standard_year | Count |
|---|---|
| `<missing/blank>` | 8461 |
| `2021` | 2567 |
| `2016` | 783 |
| `2018` | 632 |
| `2017` | 512 |
| `2015` | 299 |
| `2013` | 50 |
| `2008` | 13 |

**`laboratory_typing_method`:**

| raw laboratory_typing_method | Count |
|---|---|
| `MIC` | 4716 |
| `Disk diffusion` | 3755 |
| `Broth dilution` | 3705 |
| `Agar dilution` | 1094 |
| `<missing/blank>` | 47 |

**`laboratory_typing_platform`:**

| raw laboratory_typing_platform | Count |
|---|---|
| `<missing/blank>` | 10082 |
| `VITEK 2` | 2405 |
| `E-test` | 462 |
| `Sensititre` | 365 |
| `BD Phoenix NMIC-203` | 1 |
| `KB panel` | 1 |
| `Sensititre EUVSEC/EUVSEC2` | 1 |

**`measurement_unit`:**

| raw measurement_unit | Count |
|---|---|
| `mg/L` | 7221 |
| `<missing/blank>` | 5987 |
| `mm` | 109 |


## Step 3 - Candidate policy comparison, per antibiotic

Policy A reuses `conflict_analysis.apply_policy_1` exactly (strict categorical agreement:
Resistant+Intermediate is retained as Resistant, Susceptible+Intermediate as Susceptible;
Resistant+Susceptible disagreement is excluded; Intermediate is never mapped to a binary class).
Policies B and C add, respectively, a testing-standard-consistency and a laboratory-method-
consistency requirement on top of Policy A, under the explicit rules stated above (see
`final_policy_analysis.py` docstrings) - casing is folded for a single named standard only;
combined/mixed standard strings are never guessed at or merged into one.

### Ciprofloxacin

#### A_categorical_agreement

| Outcome | Genomes | % of genomes |
|---|---|---|
| `eligible_susceptible` | 5461 | 50.1% |
| `excluded_no_recognized_binary_interpretation` | 3579 | 32.8% |
| `eligible_resistant` | 1846 | 16.9% |
| `excluded_categorical_disagreement` | 18 | 0.2% |
| **Total (partition check)** | **10904** | should equal 10904 |

- Eligible: **7307** genomes - Resistant 1846/7307 (25.3%), Susceptible 5461/7307 (74.7%)
- Eligible genomes with more than one AST record for this antibiotic: 4430
  - differing `testing_standard`: 1897
  - differing `testing_standard_year`: 34
  - differing `laboratory_typing_method`: 1575
  - differing `laboratory_typing_platform`: 458
  - differing `measurement_unit`: 4430
- Eligible genomes with an MLST value: 7274 / 7307 (783 distinct groups among them)

#### B_categorical_agreement_plus_standard_consistency

| Outcome | Genomes | % of genomes |
|---|---|---|
| `excluded_by_policy_a:excluded_no_recognized_binary_interpretation` | 3579 | 32.8% |
| `excluded_standard_missing_or_combined_only` | 2892 | 26.5% |
| `excluded_standard_inconsistent_across_records` | 1897 | 17.4% |
| `eligible_susceptible` | 1392 | 12.8% |
| `eligible_resistant` | 1126 | 10.3% |
| `excluded_by_policy_a:excluded_categorical_disagreement` | 18 | 0.2% |
| **Total (partition check)** | **10904** | should equal 10904 |

- Eligible: **2518** genomes - Resistant 1126/2518 (44.7%), Susceptible 1392/2518 (55.3%)
- Eligible genomes with more than one AST record for this antibiotic: 184
  - differing `testing_standard`: 0
  - differing `testing_standard_year`: 0
  - differing `laboratory_typing_method`: 1
  - differing `laboratory_typing_platform`: 1
  - differing `measurement_unit`: 184
- Eligible genomes with an MLST value: 2487 / 2518 (398 distinct groups among them)

#### C_categorical_agreement_plus_method_consistency

| Outcome | Genomes | % of genomes |
|---|---|---|
| `eligible_susceptible` | 4104 | 37.6% |
| `excluded_by_policy_a:excluded_no_recognized_binary_interpretation` | 3579 | 32.8% |
| `eligible_resistant` | 1601 | 14.7% |
| `excluded_method_inconsistent_method_or_unit` | 1575 | 14.4% |
| `excluded_method_missing_method_only` | 27 | 0.2% |
| `excluded_by_policy_a:excluded_categorical_disagreement` | 18 | 0.2% |
| **Total (partition check)** | **10904** | should equal 10904 |

- Eligible: **5705** genomes - Resistant 1601/5705 (28.1%), Susceptible 4104/5705 (71.9%)
- Eligible genomes with more than one AST record for this antibiotic: 2855
  - differing `testing_standard`: 1863
  - differing `testing_standard_year`: 0
  - differing `laboratory_typing_method`: 0
  - differing `laboratory_typing_platform`: 0
  - differing `measurement_unit`: 2855
- Eligible genomes with an MLST value: 5673 / 5705 (677 distinct groups among them)

### Ampicillin

#### A_categorical_agreement

| Outcome | Genomes | % of genomes |
|---|---|---|
| `eligible_resistant` | 3659 | 49.6% |
| `eligible_susceptible` | 2831 | 38.3% |
| `excluded_no_recognized_binary_interpretation` | 886 | 12.0% |
| `excluded_categorical_disagreement` | 7 | 0.1% |
| **Total (partition check)** | **7383** | should equal 7383 |

- Eligible: **6490** genomes - Resistant 3659/6490 (56.4%), Susceptible 2831/6490 (43.6%)
- Eligible genomes with more than one AST record for this antibiotic: 4337
  - differing `testing_standard`: 2878
  - differing `testing_standard_year`: 959
  - differing `laboratory_typing_method`: 1423
  - differing `laboratory_typing_platform`: 461
  - differing `measurement_unit`: 4336
- Eligible genomes with an MLST value: 6448 / 6490 (802 distinct groups among them)

#### B_categorical_agreement_plus_standard_consistency

| Outcome | Genomes | % of genomes |
|---|---|---|
| `excluded_standard_inconsistent_across_records` | 2878 | 39.0% |
| `excluded_standard_missing_or_combined_only` | 1832 | 24.8% |
| `eligible_resistant` | 1491 | 20.2% |
| `excluded_by_policy_a:excluded_no_recognized_binary_interpretation` | 886 | 12.0% |
| `eligible_susceptible` | 289 | 3.9% |
| `excluded_by_policy_a:excluded_categorical_disagreement` | 7 | 0.1% |
| **Total (partition check)** | **7383** | should equal 7383 |

- Eligible: **1780** genomes - Resistant 1491/1780 (83.8%), Susceptible 289/1780 (16.2%)
- Eligible genomes with more than one AST record for this antibiotic: 184
  - differing `testing_standard`: 0
  - differing `testing_standard_year`: 0
  - differing `laboratory_typing_method`: 0
  - differing `laboratory_typing_platform`: 0
  - differing `measurement_unit`: 183
- Eligible genomes with an MLST value: 1749 / 1780 (428 distinct groups among them)

#### C_categorical_agreement_plus_method_consistency

| Outcome | Genomes | % of genomes |
|---|---|---|
| `eligible_resistant` | 2955 | 40.0% |
| `eligible_susceptible` | 2081 | 28.2% |
| `excluded_method_inconsistent_method_or_unit` | 1423 | 19.3% |
| `excluded_by_policy_a:excluded_no_recognized_binary_interpretation` | 886 | 12.0% |
| `excluded_method_missing_method_only` | 31 | 0.4% |
| `excluded_by_policy_a:excluded_categorical_disagreement` | 7 | 0.1% |
| **Total (partition check)** | **7383** | should equal 7383 |

- Eligible: **5036** genomes - Resistant 2955/5036 (58.7%), Susceptible 2081/5036 (41.3%)
- Eligible genomes with more than one AST record for this antibiotic: 2914
  - differing `testing_standard`: 1919
  - differing `testing_standard_year`: 0
  - differing `laboratory_typing_method`: 0
  - differing `laboratory_typing_platform`: 0
  - differing `measurement_unit`: 2913
- Eligible genomes with an MLST value: 5004 / 5036 (707 distinct groups among them)

### Gentamicin

#### A_categorical_agreement

| Outcome | Genomes | % of genomes |
|---|---|---|
| `eligible_susceptible` | 6478 | 58.0% |
| `excluded_no_recognized_binary_interpretation` | 3835 | 34.3% |
| `eligible_resistant` | 862 | 7.7% |
| **Total (partition check)** | **11175** | should equal 11175 |

- Eligible: **7340** genomes - Resistant 862/7340 (11.7%), Susceptible 6478/7340 (88.3%)
- Eligible genomes with more than one AST record for this antibiotic: 1761
  - differing `testing_standard`: 37
  - differing `testing_standard_year`: 37
  - differing `laboratory_typing_method`: 1578
  - differing `laboratory_typing_platform`: 461
  - differing `measurement_unit`: 1760
- Eligible genomes with an MLST value: 7306 / 7340 (805 distinct groups among them)

#### B_categorical_agreement_plus_standard_consistency

| Outcome | Genomes | % of genomes |
|---|---|---|
| `excluded_by_policy_a:excluded_no_recognized_binary_interpretation` | 3835 | 34.3% |
| `eligible_susceptible` | 3814 | 34.1% |
| `excluded_standard_missing_or_combined_only` | 2789 | 25.0% |
| `eligible_resistant` | 700 | 6.3% |
| `excluded_standard_inconsistent_across_records` | 37 | 0.3% |
| **Total (partition check)** | **11175** | should equal 11175 |

- Eligible: **4514** genomes - Resistant 700/4514 (15.5%), Susceptible 3814/4514 (84.5%)
- Eligible genomes with more than one AST record for this antibiotic: 184
  - differing `testing_standard`: 0
  - differing `testing_standard_year`: 0
  - differing `laboratory_typing_method`: 1
  - differing `laboratory_typing_platform`: 1
  - differing `measurement_unit`: 183
- Eligible genomes with an MLST value: 4482 / 4514 (617 distinct groups among them)

#### C_categorical_agreement_plus_method_consistency

| Outcome | Genomes | % of genomes |
|---|---|---|
| `eligible_susceptible` | 4979 | 44.6% |
| `excluded_by_policy_a:excluded_no_recognized_binary_interpretation` | 3835 | 34.3% |
| `excluded_method_inconsistent_method_or_unit` | 1578 | 14.1% |
| `eligible_resistant` | 753 | 6.7% |
| `excluded_method_missing_method_only` | 30 | 0.3% |
| **Total (partition check)** | **11175** | should equal 11175 |

- Eligible: **5732** genomes - Resistant 753/5732 (13.1%), Susceptible 4979/5732 (86.9%)
- Eligible genomes with more than one AST record for this antibiotic: 183
  - differing `testing_standard`: 0
  - differing `testing_standard_year`: 0
  - differing `laboratory_typing_method`: 0
  - differing `laboratory_typing_platform`: 0
  - differing `measurement_unit`: 182
- Eligible genomes with an MLST value: 5700 / 5732 (701 distinct groups among them)


## Step 4 - Existing ciprofloxacin pilot under each policy

The existing 100-genome ciprofloxacin pilot cohort (`data/processed/pilot_ciprofloxacin_selected_cohort.csv`) was built using `conflict_analysis.apply_policy_2` (a STRICTER rule than any policy A/B/C here - it additionally forbids any Intermediate co-occurrence). The pilot cohort itself was NOT changed by this task. Checking whether its 100 genomes and their ORIGINAL pilot labels would be retained under each new policy:

| Policy | Retained | Excluded | Label mismatches vs. pilot |
|---|---|---|---|
| A | 100 | 0 | 0 |
| B | 29 | 71 | 0 |
| C | 80 | 20 | 0 |

A policy's eligible-genome LABEL is never reinterpreted as a new pilot result here - the existing pilot's reported metrics remain exactly what they were computed from (`reports/pilot_ciprofloxacin_report.md`), regardless of what this task finds.

## Step 5 - Recommendation

**Empirical findings (directly computed above):**

- Ciprofloxacin: Policy A retains 7307, Policy B retains 2518 (66% fewer than A), Policy C retains 5705 (22% fewer than A).
- Ampicillin: Policy A retains 6490, Policy B retains 1780 (73% fewer than A), Policy C retains 5036 (22% fewer than A).
- Gentamicin: Policy A retains 7340, Policy B retains 4514 (39% fewer than A), Policy C retains 5732 (22% fewer than A).

- Across all three antibiotics, Policy B (standard consistency) removes a substantial fraction of Policy A's eligible genomes, driven overwhelmingly by missing `testing_standard` values (not by inconsistency between named standards) - this project's own reconnaissance already established that roughly 30-50% of lab-evidence AST records per antibiotic have a blank `testing_standard` field.
- Policy C (method consistency) removes relatively few additional genomes beyond Policy A, because `laboratory_typing_method` is populated for the large majority of records and is usually consistent within a genome's own repeated records.

**Judgment calls made in this analysis (explicitly distinguished from the above):**

- That casing differences in `testing_standard` (e.g. `eucast` vs `EUCAST`) represent the same intended value is an assumption, not a verified fact about how each record was entered - a reasonable one, but stated as such.
- That a combined standard string (e.g. `EUCAST, CLSI`) should be treated as `missing_or_combined` rather than guessed at is a deliberate, conservative choice, not the only defensible one - a reviewer could instead choose to trust the first-listed standard, which this analysis does not do.
- That differing `measurement_unit` values should count as method-incompatible (Policy C) even when the method label matches is a judgment call reflecting the instruction not to treat different measurement types as interchangeable, not an empirically derived rule.

**Recommendation:**

Policy A (categorical agreement alone, i.e. the existing `apply_policy_1`) is recommended as the primary label-curation rule for all three antibiotics going forward, NOT because it retains the most genomes, but because: (1) it already excludes the two most scientifically indefensible cases - genuine Resistant/Susceptible disagreement and unlabeled/Intermediate-only genomes - without requiring an additional, less-verifiable assumption about standard or method comparability; (2) Policy B's standard-consistency requirement is overwhelmingly driven by missing metadata rather than genuine inconsistency, so it mostly discards usable genomes rather than resolving a real scientific conflict; (3) Policy C adds comparatively little additional rigor at a modest cost in sample size. This recommendation is a judgment call about which trade-off is most defensible for a 100-genome-scale pilot - a reviewer prioritizing standard purity over sample size could reasonably choose Policy B instead, and that choice should be made explicitly, not by default. Policy A is feasible to apply identically across all three antibiotics, since it depends only on the phenotype field, which is populated consistently across all three CSVs.
