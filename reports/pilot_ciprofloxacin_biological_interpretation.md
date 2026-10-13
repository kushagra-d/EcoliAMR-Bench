# Ciprofloxacin Pilot - Biological Interpretation of Predictive k-mers (PRELIMINARY, EXPLORATORY)

- Analysis date (UTC): 2026-10-09T19:25:29.314122+00:00
- Antibiotic: ciprofloxacin
- Random seed: 20261009

**This is exploratory biological interpretation of a 70-genome TRAINING-partition
refit, not a validated biomarker discovery, and not proof that any matched gene causes the
observed phenotype. Read the Limitations and Caveats sections before drawing any conclusion.**

## CARD resource and version

- Source: official CARD data download endpoint, `https://card.mcmaster.ca/latest/data`
  (confirmed live, reproducible, unauthenticated GET request returning `card-data.tar.bz2`).
- CARD version: **4.0.2** (from `card.json`'s `_version` field).
- Retrieval date (this session): 2026-10-09.
- License: per `CARD-Download-README.txt` bundled in the download - free for personal, academic,
  government, and non-profit research/educational use; commercial use requires a separate license
  from McMaster University. Citation: Alcock et al. 2023, "CARD 2023: expanded curation, support
  for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance
  Database", Nucleic Acids Research 51:D690-D699.
- Files used: all five nucleotide reference FASTA files shipped in the download
  (`nucleotide_fasta_protein_homolog_model.fasta`, `..._protein_variant_model.fasta`,
  `..._protein_knockout_model.fasta`, `..._protein_overexpression_model.fasta`,
  `..._rRNA_gene_variant_model.fasta`) - **not** pre-filtered to fluoroquinolone-specific genes, so
  matches to unrelated AMR genes are reported too (see Limitations for why that matters here).
- 6415 reference sequences loaded, 0 with an unparsed header (every
  header matched the documented `>gb|accession|strand|start-end|ARO:id|short_name [organism]`
  format), 799 windows skipped for containing an ambiguous
  base.

## Candidate-selection methodology

No fitted model was persisted by the earlier pilot/model-comparison work - this task refit the
**exact same Logistic Regression and Random Forest configurations** already used and reported
elsewhere in this project, on the **training rows of the existing saved random split only**
(`data/processed/pilot_ciprofloxacin_random_split.csv`, 70 genomes). **No test-set
label or row was read by the selection step** - verified by a dedicated unit test
(`test_select_candidate_kmers_never_selects_a_test_only_signal`) using a synthetic dataset where a
test-only signal must NOT be selected.

- **LogisticRegression** candidates: top 20 k-mers by absolute
  coefficient magnitude (`params={'max_iter': 1000, 'random_state': 20261009}`).
- **RandomForest** candidates: top 20 k-mers by Gini feature importance
  (`params={'n_estimators': 100, 'n_jobs': -1, 'random_state': 20261009}`).
- Combined, de-duplicated candidate set: **39 unique
  k-mers** (some overlap between the two models' top lists is expected and not forced).
- Rankings are **model-specific** and reported separately per candidate below - a high
  Random Forest importance does not imply a high Logistic Regression coefficient, and vice versa.
- Treated as exploratory throughout: these rankings come from fitting ~70 genomes
  with ~524,000 raw features - far too few samples relative to features for the ranking itself to
  be considered stable or validated.

## CARD k-mer index and k-mer-space-saturation caveat (read before interpreting any match)

The CARD reference sequences alone contain **457,502 distinct canonical 10-mers**
out of the ~524,800 theoretically possible canonical 10-mers - **87.2% of the
entire possible k=10 sequence space**. This means that, independent of any biological
relationship, a large fraction of ALL possible 10-mers - including ones with no connection to
antibiotic resistance - will find an "exact match" somewhere in CARD purely because CARD's
reference sequences are large enough to nearly saturate the k=10 feature space. **An exact match
at k=10 is therefore weak evidence on its own** and must not be read as gene identification
without much stronger corroboration (e.g. a longer contiguous alignment, which this pilot's k=10
representation cannot provide). This caveat applies to every "exact" match reported below.

## Matching results

- Candidates evaluated: **39**
- Exact matches: **37**
- Approximate (single-nucleotide-substitution) matches: **2**
- Unmapped (no match within 1 substitution): **0**
- Ambiguous (matched more than one distinct CARD gene/short_name): **34**

"Approximate" here means a precise, literal definition: at least one single-nucleotide-substitution
neighbor of the candidate's canonical 10-mer exactly matches an indexed CARD reference k-mer. This
is NOT a BLAST-style alignment and does NOT by itself establish sequence homology beyond that
10-bp window - approximate matches are reported as a weaker category than exact matches and are
never described as a gene identification.

## Notable observations (stated plainly, including the negative result)

- **5 of 39** candidates matched exactly one distinct CARD
  gene/short_name (the least-ambiguous, relatively "cleanest" matches available from this
  analysis) - the remaining matched candidates were ambiguous across anywhere from 2 to several
  hundred distinct genes (see the table below), which is itself evidence of how little a single
  10-mer match constrains gene identity.
- Checking every match (ambiguous or not) against a short, manually curated, deliberately
  non-exhaustive list of CARD short_name patterns associated with KNOWN fluoroquinolone-resistance
  mechanisms (`gyrA`, `gyrB`, `parC`, `parE`, `qnr`, `oqxA`, `oqxB`, `qepA`,
  `aac(6')-Ib-cr`): **25** candidate/gene-hit pairs matched one of these
  patterns, but **0 of those were unambiguous** (i.e. in
  every case where a candidate's match happened to include a known fluoroquinolone-mechanism gene
  name, that candidate ALSO matched other, unrelated genes in the same breath - never a clean,
  isolated match to a known mechanism).
- **None of the 5 cleanest (unambiguous) matches correspond to a known
  fluoroquinolone-resistance mechanism** - stated here explicitly as a negative result, not
  omitted. This pilot's k=10 exploratory approach did not surface a clean link to the established
  ciprofloxacin-resistance mechanism in *E. coli*; that absence is itself informative (consistent
  with the k-mer-saturation caveat above, and/or with this pilot's small size and binary-labeling
  limitations), not evidence that no such link exists.

## Candidate annotations

| Candidate k-mer | Selected by (model#rank, score) | Match type | Ambiguous? | CARD gene hit(s): short_name (ARO id, supporting sequences, organism count) |
|---|---|---|---|---|
| `GTCGGTCTGA` | LogisticRegression#1 (score=0.001519) | exact | no | AAC(6')-IIa (ARO:3002594, 1 seq, 1 organism(s)) |
| `ATAATCAGAC` | LogisticRegression#2 (score=0.001354) | exact | YES | ugd (ARO:3003577, 1 seq, 1 organism(s)); adeK (ARO:3000782, 1 seq, 1 organism(s)); GOB-33 (ARO:3005682, 1 seq, 1 organism(s)); GOB-34 (ARO:3005683, 1 seq, 1 organism(s)); GOB-35 (ARO:3005684, 1 seq, 1 organism(s)); ... +4 more distinct gene(s) |
| `ATCAGACCGA` | LogisticRegression#3 (score=0.001332) | exact | YES | catII (ARO:3002684, 1 seq, 1 organism(s)); tet(41) (ARO:3000569, 1 seq, 1 organism(s)); TolC (ARO:3000237, 1 seq, 1 organism(s)); Bado_rpoB_RIF (ARO:3004480, 1 seq, 1 organism(s)); Ecol_rpoB_RIF (ARO:3003288, 1 seq, 1 organism(s)); ... +3 more distinct gene(s) |
| `AATCAGACCG` | LogisticRegression#4 (score=0.001315) | exact | YES | PER-4 (ARO:3002366, 1 seq, 1 organism(s)); mdtB (ARO:3000793, 1 seq, 1 organism(s)); PER-7 (ARO:3002369, 1 seq, 1 organism(s)); PER-1 (ARO:3002363, 1 seq, 1 organism(s)); BEL-1 (ARO:3002385, 1 seq, 1 organism(s)); ... +18 more distinct gene(s) |
| `CAGACCGACG` | LogisticRegression#5 (score=0.001239) | exact | YES | mexX (ARO:3003034, 1 seq, 1 organism(s)); APH(3')-IIc (ARO:3002646, 1 seq, 1 organism(s)); APH(6)-Id (ARO:3002660, 1 seq, 1 organism(s)); Bpse_Omp38 (ARO:3004123, 1 seq, 1 organism(s)); Scin_EFTu_ELF (ARO:3003359, 1 seq, 1 organism(s)); ... +1 more distinct gene(s) |
| `ATAAAAATAT` | LogisticRegression#6 (score=0.001183) | exact | YES | OXA-347 (ARO:3001777, 2 seq, 1 organism(s)); mecA (ARO:3000617, 2 seq, 1 organism(s)); vgaALC (ARO:3002830, 1 seq, 1 organism(s)); APH(3')-VIIa (ARO:3002654, 1 seq, 1 organism(s)); APH(9)-Ia (ARO:3002662, 1 seq, 1 organism(s)); ... +47 more distinct gene(s) |
| `CCGACGATAC` | LogisticRegression#7 (score=0.001138) | exact | YES | SAT-3 (ARO:3002898, 1 seq, 1 organism(s)); opmE (ARO:3003700, 1 seq, 1 organism(s)); cmlA9 (ARO:3005043, 1 seq, 1 organism(s)); Mtub_Rv2731_PZA (ARO:3004987, 1 seq, 1 organism(s)); Bdol_gyrA_FLO (ARO:3004860, 1 seq, 1 organism(s)) |
| `AGACCGACGA` | LogisticRegression#8 (score=0.001126) | exact | YES | tet(G) (ARO:3000174, 1 seq, 1 organism(s)); Bpse_Omp38 (ARO:3004123, 1 seq, 1 organism(s)); Kpne_KpnG (ARO:3004588, 1 seq, 1 organism(s)) |
| `GGTCTGATTA` | LogisticRegression#9 (score=0.001121) | exact | no | Bhyo_23S_TYL (ARO:3004133, 1 seq, 1 organism(s)) |
| `ATCGTCGGTC` | LogisticRegression#10 (score=0.001093); RandomForest#15 (score=0.007484) | exact | YES | AAC(3)-IIb (ARO:3002534, 1 seq, 1 organism(s)); tet(G) (ARO:3000174, 1 seq, 1 organism(s)); emrB (ARO:3000074, 1 seq, 1 organism(s)); Kpne_KpnG (ARO:3004588, 1 seq, 1 organism(s)); Kpne_KpnH (ARO:3004597, 1 seq, 1 organism(s)) |
| `ATGCCGGATG` | LogisticRegression#11 (score=-0.001088) | exact | YES | SCO-1 (ARO:3004856, 1 seq, 1 organism(s)); Ecol_murA_FOF (ARO:3003775, 1 seq, 1 organism(s)) |
| `CATCACCGCC` | LogisticRegression#12 (score=0.0009921) | exact | YES | Eclo_acrA (ARO:3004042, 2 seq, 1 organism(s)); OXA-60 (ARO:3001808, 1 seq, 1 organism(s)); aadA4 (ARO:3002604, 1 seq, 1 organism(s)); aadA5 (ARO:3002605, 1 seq, 1 organism(s)); MexD (ARO:3000801, 1 seq, 1 organism(s)); ... +19 more distinct gene(s) |
| `ACCGACGATA` | LogisticRegression#13 (score=0.0009783) | exact | YES | SAT-3 (ARO:3002898, 1 seq, 1 organism(s)); emrB (ARO:3000074, 1 seq, 1 organism(s)); opmE (ARO:3003700, 1 seq, 1 organism(s)); Kpne_KpnH (ARO:3004597, 1 seq, 1 organism(s)); FosG (ARO:3007368, 1 seq, 1 organism(s)); ... +1 more distinct gene(s) |
| `CACCGCCAGC` | LogisticRegression#14 (score=0.0009626) | exact | YES | AxyY (ARO:3004144, 3 seq, 1 organism(s)); TriA (ARO:3003679, 2 seq, 1 organism(s)); MuxB (ARO:3004074, 2 seq, 1 organism(s)); Mtub_ppsD_PZA (ARO:3004976, 2 seq, 1 organism(s)); NDM-6 (ARO:3002356, 1 seq, 1 organism(s)); ... +268 more distinct gene(s) |
| `AATAATCAGA` | LogisticRegression#15 (score=0.0009537) | exact | YES | EreD (ARO:3004608, 1 seq, 1 organism(s)); mecB (ARO:3003440, 1 seq, 1 organism(s)); vanS_in_vanD_cl (ARO:3002934, 1 seq, 1 organism(s)); vanT_in_vanE_cl (ARO:3002971, 1 seq, 1 organism(s)); tet(47) (ARO:3004613, 1 seq, 1 organism(s)) |
| `ACTGTTGCAA` | LogisticRegression#16 (score=0.0009446) | exact | YES | cmlA5 (ARO:3002695, 2 seq, 1 organism(s)); cmlA4 (ARO:3002694, 2 seq, 1 organism(s)); cmlA1 (ARO:3002693, 2 seq, 1 organism(s)); cmlA6 (ARO:3002696, 2 seq, 1 organism(s)); cmlA8 (ARO:3004665, 2 seq, 1 organism(s)); ... +16 more distinct gene(s) |
| `ACGAGTGGGA` | LogisticRegression#17 (score=0.0009392) | exact | YES | MexE (ARO:3000803, 1 seq, 1 organism(s)); EAM-1 (ARO:3006873, 1 seq, 1 organism(s)); B3SU1-1 (ARO:3008078, 1 seq, 1 organism(s)); Mtub_katG_INH (ARO:3003392, 1 seq, 1 organism(s)); Mtub_katG_PTO (ARO:3005102, 1 seq, 1 organism(s)) |
| `CTGATTATTA` | LogisticRegression#18 (score=0.0009322) | exact | YES | mdtA (ARO:3000792, 1 seq, 1 organism(s)); mecB (ARO:3003440, 1 seq, 1 organism(s)); Ecol_mdfA (ARO:3001328, 1 seq, 1 organism(s)); vanT_in_vanE_cl (ARO:3002971, 1 seq, 1 organism(s)); EstT (ARO:3007459, 1 seq, 1 organism(s)) |
| `CTTATCAGGC` | LogisticRegression#19 (score=-0.0009321) | exact | YES | TUS-1 (ARO:3000844, 1 seq, 1 organism(s)); patB (ARO:3000025, 1 seq, 1 organism(s)) |
| `ATCTGAACAT` | LogisticRegression#20 (score=-0.0009216) | exact | YES | kdpD (ARO:3004047, 1 seq, 1 organism(s)); Mtub_ponA1_RFB (ARO:3004960, 1 seq, 1 organism(s)) |
| `ACGGTGTAGC` | RandomForest#1 (score=0.009011) | approximate | YES | Ecol_rpoB_RIF (ARO:3003288, 2 seq, 1 organism(s)); MCR-3.3 (ARO:3004662, 2 seq, 1 organism(s)); MCR-3.12 (ARO:3004693, 2 seq, 1 organism(s)); MCR-3.4 (ARO:3004691, 2 seq, 1 organism(s)); MCR-3.11 (ARO:3004500, 2 seq, 1 organism(s)); ... +226 more distinct gene(s) |
| `ATGCTGTCAG` | RandomForest#2 (score=0.008445) | exact | YES | OXA-134 (ARO:3001695, 1 seq, 1 organism(s)); OXA-335 (ARO:3001523, 1 seq, 1 organism(s)); Ecol_ampC_BLA (ARO:3004290, 1 seq, 1 organism(s)); EC-13 (ARO:3006874, 1 seq, 1 organism(s)); EC-14 (ARO:3006875, 1 seq, 1 organism(s)); ... +6 more distinct gene(s) |
| `CAAGCGAACG` | RandomForest#3 (score=0.008375) | exact | YES | vanS_in_vanC_cl (ARO:3002933, 1 seq, 1 organism(s)); CTX-M-78 (ARO:3001939, 1 seq, 1 organism(s)); Bbur_16S_GEN (ARO:3003504, 1 seq, 1 organism(s)); Bbur_16S_SPT (ARO:3003502, 1 seq, 1 organism(s)) |
| `CCAGCGGGAC` | RandomForest#4 (score=0.008363) | exact | YES | APH(9)-Ib (ARO:3002663, 1 seq, 1 organism(s)); mexY (ARO:3003033, 1 seq, 1 organism(s)); MexB (ARO:3000378, 1 seq, 1 organism(s)); MexF (ARO:3000804, 1 seq, 1 organism(s)); Mtub_rpoC_RIF (ARO:3004994, 1 seq, 1 organism(s)) |
| `GGAGAGACTA` | RandomForest#5 (score=0.008175) | exact | no | Hpyl_16S_TET (ARO:3003510, 1 seq, 1 organism(s)) |
| `ACTACGATGA` | RandomForest#6 (score=0.007953) | exact | YES | aadA27 (ARO:3004682, 1 seq, 1 organism(s)); ANT(3'')-IIc (ARO:3004091, 1 seq, 1 organism(s)) |
| `CACTCGTATC` | RandomForest#7 (score=0.007879) | exact | no | mepR (ARO:3000746, 1 seq, 1 organism(s)) |
| `ACTTCTCTTA` | RandomForest#8 (score=0.007803) | exact | YES | Saga_mprF (ARO:3003774, 1 seq, 1 organism(s)); Hpyl_pbp2_AMX (ARO:3007058, 1 seq, 1 organism(s)) |
| `AAGCGTTGAG` | RandomForest#9 (score=0.007794) | exact | YES | adeI (ARO:3000780, 1 seq, 1 organism(s)); PFM-1 (ARO:3006965, 1 seq, 1 organism(s)); vanS_in_vanF_cl (ARO:3002936, 1 seq, 1 organism(s)); B3SU2-1 (ARO:3008079, 1 seq, 1 organism(s)); OXA-1091 (ARO:3008509, 1 seq, 1 organism(s)); ... +1 more distinct gene(s) |
| `GCGGGAGAGC` | RandomForest#10 (score=0.007775) | exact | YES | dfrA22 (ARO:3003018, 1 seq, 1 organism(s)); Sven_rox (ARO:3007209, 1 seq, 1 organism(s)) |
| `AAAAGTGGTC` | RandomForest#11 (score=0.007775) | exact | YES | TEM-17 (ARO:3000888, 1 seq, 1 organism(s)); PC1_blaZ (ARO:3000621, 1 seq, 1 organism(s)); OKP-B-12 (ARO:3002445, 1 seq, 1 organism(s)); ARL-2 (ARO:3004735, 1 seq, 1 organism(s)); ARL-3 (ARO:3004736, 1 seq, 1 organism(s)); ... +4 more distinct gene(s) |
| `AAATACCTGG` | RandomForest#12 (score=0.007697) | exact | YES | PER-2 (ARO:3002364, 1 seq, 1 organism(s)); PER-6 (ARO:3002368, 1 seq, 1 organism(s)); PER-14 (ARO:3006135, 1 seq, 1 organism(s)); RSD1-1 (ARO:3009042, 1 seq, 1 organism(s)); Mtub_ponA1_RFB (ARO:3004960, 1 seq, 1 organism(s)) |
| `CTGTCGCTAC` | RandomForest#13 (score=0.007682) | exact | YES | CMY-70 (ARO:3002083, 1 seq, 1 organism(s)); CMY-39 (ARO:3002050, 1 seq, 1 organism(s)); CMY-100 (ARO:3002112, 1 seq, 1 organism(s)); CMY-98 (ARO:3002111, 1 seq, 1 organism(s)); CMY-35 (ARO:3002046, 1 seq, 1 organism(s)); ... +34 more distinct gene(s) |
| `GTAGAAGCTA` | RandomForest#14 (score=0.007552) | approximate | YES | Cper_cplR (ARO:3007644, 3 seq, 1 organism(s)); IMP-91 (ARO:3007460, 2 seq, 1 organism(s)); IMP-86 (ARO:3008220, 2 seq, 1 organism(s)); IMP-87 (ARO:3008221, 2 seq, 1 organism(s)); tet(48) (ARO:3004581, 2 seq, 1 organism(s)); ... +139 more distinct gene(s) |
| `AAGATACACC` | RandomForest#16 (score=0.007484) | exact | no | dfrA35 (ARO:3005164, 1 seq, 1 organism(s)) |
| `ACCGACAGCA` | RandomForest#17 (score=0.007396) | exact | YES | MexA (ARO:3000377, 1 seq, 1 organism(s)); OXA-198 (ARO:3001805, 1 seq, 1 organism(s)); lsaB (ARO:3003111, 1 seq, 1 organism(s)); CARB-7 (ARO:3002246, 1 seq, 1 organism(s)); arr-1 (ARO:3002846, 1 seq, 1 organism(s)); ... +5 more distinct gene(s) |
| `GGCCGCGAAC` | RandomForest#18 (score=0.00731) | exact | YES | PEN-A1 (ARO:3008997, 2 seq, 1 organism(s)); APH(7'')-Ia (ARO:3002661, 1 seq, 1 organism(s)); ANT(4')-IIa (ARO:3002624, 1 seq, 1 organism(s)); opcM (ARO:3003037, 1 seq, 1 organism(s)); MexV (ARO:3003030, 1 seq, 1 organism(s)); ... +46 more distinct gene(s) |
| `AAATCCAAAC` | RandomForest#19 (score=0.0072) | exact | YES | AAC(6')-IIc (ARO:3002596, 1 seq, 1 organism(s)); msrC (ARO:3002819, 1 seq, 1 organism(s)); GOB-14 (ARO:3004806, 1 seq, 1 organism(s)); LpeB (ARO:3004100, 1 seq, 1 organism(s)); OXA-274 (ARO:3001730, 1 seq, 1 organism(s)); ... +3 more distinct gene(s) |
| `CCGATCTCCC` | RandomForest#20 (score=0.007141) | exact | YES | PDC-8 (ARO:3002507, 1 seq, 1 organism(s)); PDC-5 (ARO:3002502, 1 seq, 1 organism(s)); PDC-1 (ARO:3002497, 1 seq, 1 organism(s)); PDC-7 (ARO:3002506, 1 seq, 1 organism(s)); PDC-10 (ARO:3002509, 1 seq, 1 organism(s)); ... +580 more distinct gene(s) |

## Limitations

- **Pilot size**: candidates come from a 70-genome training refit for one
  antibiotic - far too small to treat any ranking as stable, let alone validated.
- **Feature representation**: k=10 canonical k-mers are short enough that, as shown above, CARD's
  own reference set nearly saturates the possible sequence space - sequence-identity evidence at
  this k is inherently weak. The project's own proposal-vs-instructor k=10-vs-31 discrepancy
  (documented in `reports/pilot_ciprofloxacin_report.md`) is directly relevant here: a larger k would
  give much stronger per-match evidence, at the cost of a far larger, less-saturated feature space.
- **Reference coverage**: CARD is curated from published, characterized resistance determinants;
  it does not contain every possible resistance-associated sequence, and a k-mer matching nothing
  in CARD (`unmapped`) does not mean it has no biological relevance - only that this specific,
  limited reference set did not contain a close match.
- **Protein-variant-model hits specifically**: a match to a `nucleotide_fasta_protein_variant_model`
  sequence (e.g. a `gyrA`/`parC` reference locus) indicates sequence similarity to the **wild-type**
  reference gene location, not the presence of a resistance-conferring mutation - CARD's variant
  model requires secondary SNP screening (not performed here) to call resistance from these loci.
- **No causal claim**: nothing in this report establishes that any matched gene causes the
  Resistant/Susceptible label observed in this pilot. A predictive association from a 100-genome
  pilot, combined with a short-k sequence match, is not evidence of a causal or even a confirmed
  correlative biological mechanism - it is, at most, a reason to investigate further with stronger
  methods (larger k, full-length alignment, confirmed mutation screening) before any such claim.

## What is verified vs. speculative

- **Verified by direct computation, inspectable in the saved CSV**: which k-mers were ranked
  highly by each model on the training partition; whether each candidate's canonical sequence (or
  a single-substitution neighbor) is present in the downloaded CARD reference data, and which
  specific CARD accessions/ARO terms support that; CARD's own version and retrieval date.
- **Speculative, not established by this report**: that any specific matched gene is the
  biological cause of ciprofloxacin resistance/susceptibility in this pilot's genomes; that an
  unmapped k-mer lacks biological relevance; that this candidate ranking would replicate on a
  larger cohort.
