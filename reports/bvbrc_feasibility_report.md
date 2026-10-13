# BV-BRC AMR Metadata Feasibility Report

- Retrieval date (UTC): 2026-10-09T17:28:16.502667+00:00
- Query filters applied: taxon_id=562 (Escherichia coli) AND antibiotic=<name, lowercase> AND evidence="Laboratory Method"
- Source: BV-BRC Data API (`https://www.bv-brc.org/api/genome_amr/`, `.../genome/`)
- Scope: metadata only - no genome sequence data was downloaded.

## Per-antibiotic results

### Ciprofloxacin

- Records retrieved (unique by `id`): **15777**
- Reported total (`Content-Range`, first page): 15777
- Reported total (`Content-Range`, last page): 15777
- Reported total was stable across all pages.
- Duplicate record ids seen across pages (same `id`, counted once, overlap logged): 0
- Pagination errors: none
- Retrieval appears COMPLETE: unique record count matches the last reported total.
- Unique genome IDs referenced: **10904**
- Genome metadata found for: 10904 / 10904 genome IDs
- Genome IDs with NO matching genome record (metadata missing): 0
- Genome-metadata batch errors: none
- MLST available for: 10838 / 10904 joined genomes (1061 distinct MLST values)
- `genome_status` distribution among joined genomes: <missing/blank>=2, Complete=49, Deprecated=1, WGS=10852
- NCBI `assembly_accession` populated for: 667 / 10904 joined genomes (the rest have a BV-BRC assembly but no recorded NCBI cross-reference in this field)
- Genomes with exactly one AST record for this antibiotic: 6341
- Genomes with multiple AST records that AGREE on phenotype (replicates, not collapsed): 1322
- Genomes with multiple AST records that DISAGREE on phenotype (conflicting calls, all rows preserved in CSV): 3241

**Raw `resistant_phenotype` value distribution** (every observed value, nothing grouped):

| Raw `resistant_phenotype` value | Count |
|---|---|
| `<missing/blank>` | 7032 |
| `Susceptible` | 6559 |
| `Resistant` | 2066 |
| `Intermediate` | 120 |

**`testing_standard` x `testing_standard_year` distribution:**

| testing_standard | testing_standard_year | Count |
|---|---|---|
| <missing/blank> | <missing/blank> | 7248 |
| EUCAST | <missing/blank> | 3607 |
| CLSI | 2021 | 2429 |
| CLSI | 2018 | 575 |
| EUCAST | 2016 | 479 |
| CLSI | <missing/blank> | 425 |
| EUCAST | 2017 | 425 |
| CLSI | 2015 | 231 |
| EUCAST | 2021 | 138 |
| EUCAST, CLSI | 2019 | 43 |
| CLSI, EUCAST | 2021 | 35 |
| CLSI | 2013 | 26 |
| clsi | 2013 | 24 |
| eucast | 2015 | 24 |
| clsi | 2018 | 24 |
| CLSI | 2008 | 13 |
| EUCAST,CLSI | 2017 | 12 |
| CLSI | 2017 | 11 |
| EUCAST and CLSI | 2015 | 8 |

### Ampicillin

- Records retrieved (unique by `id`): **11896**
- Reported total (`Content-Range`, first page): 11896
- Reported total (`Content-Range`, last page): 11896
- Reported total was stable across all pages.
- Duplicate record ids seen across pages (same `id`, counted once, overlap logged): 0
- Pagination errors: none
- Retrieval appears COMPLETE: unique record count matches the last reported total.
- Unique genome IDs referenced: **7383**
- Genome metadata found for: 7383 / 7383 genome IDs
- Genome IDs with NO matching genome record (metadata missing): 0
- Genome-metadata batch errors: none
- MLST available for: 7338 / 7383 joined genomes (946 distinct MLST values)
- `genome_status` distribution among joined genomes: <missing/blank>=2, Complete=48, Deprecated=2, WGS=7331
- NCBI `assembly_accession` populated for: 548 / 7383 joined genomes (the rest have a BV-BRC assembly but no recorded NCBI cross-reference in this field)
- Genomes with exactly one AST record for this antibiotic: 2993
- Genomes with multiple AST records that AGREE on phenotype (replicates, not collapsed): 241
- Genomes with multiple AST records that DISAGREE on phenotype (conflicting calls, all rows preserved in CSV): 4149

**Raw `resistant_phenotype` value distribution** (every observed value, nothing grouped):

| Raw `resistant_phenotype` value | Count |
|---|---|
| `<missing/blank>` | 5181 |
| `Resistant` | 3796 |
| `Susceptible` | 2906 |
| `Intermediate` | 13 |

**`testing_standard` x `testing_standard_year` distribution:**

| testing_standard | testing_standard_year | Count |
|---|---|---|
| <missing/blank> | <missing/blank> | 5887 |
| EUCAST | <missing/blank> | 2639 |
| eucast | 2019 | 928 |
| EUCAST | 2016 | 598 |
| EUCAST | 2017 | 489 |
| CLSI | <missing/blank> | 420 |
| CLSI | 2018 | 370 |
| CLSI | 2015 | 232 |
| EUCAST | 2021 | 138 |
| EUCAST, CLSI | 2019 | 43 |
| CLSI, EUCAST | 2021 | 35 |
| CLSI | 2013 | 26 |
| eucast | 2015 | 24 |
| clsi | 2013 | 24 |
| clsi | 2018 | 12 |
| EUCAST,CLSI | 2017 | 12 |
| CLSI | 2017 | 11 |
| EUCAST and CLSI | 2015 | 8 |

### Gentamicin

- Records retrieved (unique by `id`): **13317**
- Reported total (`Content-Range`, first page): 13317
- Reported total (`Content-Range`, last page): 13317
- Reported total was stable across all pages.
- Duplicate record ids seen across pages (same `id`, counted once, overlap logged): 0
- Pagination errors: none
- Retrieval appears COMPLETE: unique record count matches the last reported total.
- Unique genome IDs referenced: **11175**
- Genome metadata found for: 11175 / 11175 genome IDs
- Genome IDs with NO matching genome record (metadata missing): 0
- Genome-metadata batch errors: none
- MLST available for: 11107 / 11175 joined genomes (1100 distinct MLST values)
- `genome_status` distribution among joined genomes: <missing/blank>=2, Complete=49, Deprecated=1, WGS=11123
- NCBI `assembly_accession` populated for: 645 / 11175 joined genomes (the rest have a BV-BRC assembly but no recorded NCBI cross-reference in this field)
- Genomes with exactly one AST record for this antibiotic: 9343
- Genomes with multiple AST records that AGREE on phenotype (replicates, not collapsed): 262
- Genomes with multiple AST records that DISAGREE on phenotype (conflicting calls, all rows preserved in CSV): 1570

**Raw `resistant_phenotype` value distribution** (every observed value, nothing grouped):

| Raw `resistant_phenotype` value | Count |
|---|---|
| `Susceptible` | 6687 |
| `<missing/blank>` | 5589 |
| `Resistant` | 872 |
| `Intermediate` | 169 |

**`testing_standard` x `testing_standard_year` distribution:**

| testing_standard | testing_standard_year | Count |
|---|---|---|
| <missing/blank> | <missing/blank> | 4467 |
| EUCAST | <missing/blank> | 3569 |
| CLSI | 2021 | 2429 |
| EUCAST | 2016 | 783 |
| CLSI | 2018 | 608 |
| EUCAST | 2017 | 489 |
| CLSI | <missing/blank> | 425 |
| CLSI | 2015 | 267 |
| EUCAST | 2021 | 138 |
| CLSI | 2013 | 26 |
| eucast | 2015 | 24 |
| clsi | 2013 | 24 |
| clsi | 2018 | 24 |
| CLSI | 2008 | 13 |
| EUCAST,CLSI | 2017 | 12 |
| CLSI | 2017 | 11 |
| EUCAST and CLSI | 2015 | 8 |


## Rules applied (recorded explicitly, per task constraints)

- Intermediate was NOT treated as Susceptible anywhere in this pipeline.
- Missing/blank/non-standard `resistant_phenotype` values were NOT treated as a valid class; they
  are counted separately under the raw-value distribution above.
- No cross-standard (e.g. EUCAST vs CLSI) combination or normalization was performed; the
  `testing_standard` x `testing_standard_year` table above is reported as-is for a later, explicit
  decision.
- Exact duplicate records (same `id` returned on more than one page, a pagination artifact) were
  deduplicated; this is the ONLY collapsing rule applied, and the count of such duplicates is
  reported per antibiotic above.
- Multiple independent AST records for the same genome_id (replicates or conflicts) were NOT
  collapsed into one row; both the CSV and the counts above preserve every record.
- No genome-quality thresholds (contig count, CheckM completeness/contamination, etc.) were
  applied to filter records; those fields are included in the CSV for a later, explicit decision.

## Limitations and unresolved questions

- "Completeness" above is evidence-based: it only reports COMPLETE when every page succeeded and
  the final unique-record count matches the most recently reported total. Any error, or any report
  marked not-complete, means the corresponding CSV may be a partial snapshot - treat it as such
  rather than as the final cohort.
- BV-BRC is a live, continuously-updated database; "reported total" drift during retrieval reflects
  genuine concurrent updates, not a client bug, but it does mean these counts are a snapshot, not a
  fixed ground truth.
- The MLST scheme is not validated against the recorded species for each genome in this script;
  an earlier manual spot-check found at least one genome labelled *Escherichia coli* with an MLST
  value from a different species' scheme, which this automated pass does not detect or correct.
- No genome-level QC filtering (assembly completeness, contamination, contig count) has been
  applied; the CSVs intentionally include genomes of unknown or poor quality for now.
- Final cohort size, lineage-split strategy, and label-standardization choices are explicitly out
  of scope for this task and are left for a dedicated follow-up decision informed by this report.
