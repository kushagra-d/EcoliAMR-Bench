"""Entry point: audit conflicting AST phenotype calls and compare
hypothetical label-curation policies, using only the existing
reconnaissance CSVs under data/processed/ (read-only).

Usage (from the project root, with the project venv active):

    python src/run_phenotype_conflict_analysis.py

Writes reports/phenotype_conflict_analysis.md and a derived
data/processed/phenotype_conflict_policy_summary.csv. Does not modify the
existing per-antibiotic CSVs or the original feasibility report, and
performs no network access, genome download, or label assignment.
"""

from bvbrc_recon.conflict_pipeline import main

if __name__ == "__main__":
    main()
