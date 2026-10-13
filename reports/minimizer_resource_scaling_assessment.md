# Minimizer Resource-Scaling Assessment (brief, machine-generated)

See `data/processed/minimizer_resource_scaling_assessment.json` for full detail; this file is a brief, unpolished summary only - not a final narrative report.

## Cross-checks (measured config vs. saved artifacts)

- w=5: shape_matches=True, nnz_matches=True, vocab_rows_match=True, feature_dist={'min': 1499566, 'mean': 1874846.07, 'median': 1880285.5, 'max': 2135951, 'std': 74467.51545758123}
- w=10: shape_matches=True, nnz_matches=True, vocab_rows_match=True, feature_dist={'min': 841548, 'mean': 1054110.92, 'median': 1057665.0, 'max': 1201638, 'std': 42105.86865929262}
- w=15: shape_matches=True, nnz_matches=True, vocab_rows_match=True, feature_dist={'min': 586704, 'mean': 734693.51, 'median': 736894.5, 'max': 839001, 'std': 29431.371134724595}

## Union-cohort (8,710 genomes) full-scale estimates, all representations

| Representation | Config | Vocab [low,base,high] | Est. nnz | Est. total memory | Fits <16GB? | Est. time |
|---|---|---|---|---|---|---|
| minimizer | k=31,w=5 | [12,177,139, 113,646,048, 1,060,628,806] | 16,329,909,269 | 389.3 GB | False | 688.1 min |
| minimizer | k=31,w=10 | [6,884,601, 64,252,177, 599,648,746] | 9,181,306,113 | 531.9 GB | False | 644.3 min |
| minimizer | k=31,w=15 | [4,815,833, 44,944,907, 419,459,054] | 6,399,180,472 | 459.9 GB | False | 569.9 min |
| raw k-mer | k=10 | [524,346, 524,572, 524,800] | 4,326,152,567 | 247.0 GB | False | 396.9 min |
| raw k-mer | k=15 | [23,171,787, 111,535,906, 536,870,912] | 43,256,815,999 | 623.5 GB | False | 709.7 min |
| raw k-mer | k=17 | [25,436,868, 237,395,626, 2,215,551,202] | 43,957,629,829 | 652.2 GB | False | 663.3 min |
| raw k-mer | k=19 | [26,626,636, 248,499,419, 2,319,179,995] | 44,055,749,111 | 612.7 GB | False | 654.6 min |
| raw k-mer | k=21 | [27,690,097, 258,424,422, 2,411,807,448] | 44,092,835,855 | 655.3 GB | False | 698.4 min |
| raw k-mer | k=31 | [32,648,927, 304,703,884, 2,843,721,541] | 44,186,647,433 | 658.9 GB | False | 809.2 min |
| feature hashing | d=65536 | [65,536, 65,536, 65,536] | 570,818,560 | 623.5 GB | False | 64.3 min |
| feature hashing | d=262144 | [262,144, 262,144, 262,144] | 2,283,274,240 | 623.5 GB | False | 75.1 min |
| feature hashing | d=1048576 | [1,048,576, 1,048,576, 1,048,576] | 9,051,571,098 | 623.5 GB | False | 82.0 min |

Notes: vocabulary [low,base,high] for raw k-mers uses the saturation-aware method already established in `resource_estimation.py`; for minimizers, [low,base,high] is derived by scaling raw k=31's own projection by the pilot-measured minimizer/raw-k31 vocabulary ratio (assumption: this ratio holds at full scale - unverified beyond the single 100-genome pilot); for feature hashing, vocabulary is fixed by construction (no uncertainty). nnz and processing time are extrapolated linearly from the pilot's own measured per-genome rate for every representation. Total memory for raw k-mers and minimizers = estimated CSR matrix size + a construction-overhead term scaled from that representation's OWN pilot-measured (peak_rss - final_CSR_size) ratio. **Feature hashing is different**: its pilot `peak_rss` was measured in a process that also loaded the pre-hashing raw matrix first (same finding already documented in the feature-hashing task's own report), so that ratio was NOT extrapolated for hashing; instead hashing's total memory is the max of its own tiny final CSR size and its prerequisite raw representation's (raw k=15's) own full-scale estimate - which is why all three hashing dimensions show the SAME ~623GB figure here, driven entirely by the prerequisite, not by the hashed matrix itself. This is a real, if non-obvious, finding: hashing AS CURRENTLY IMPLEMENTED (a post-hoc transform of an already-built raw matrix) does not reduce full-scale memory below its prerequisite's own requirement - only a streaming hash-during-extraction implementation (not built here) could. None of this has been measured at full scale; see the JSON's `rationale`/`memory_note` fields for the exact assumption behind every number.