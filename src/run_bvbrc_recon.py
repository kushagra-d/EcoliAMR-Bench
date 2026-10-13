"""Entry point: run the metadata-only BV-BRC reconnaissance pipeline.

Usage (from the project root, with the project venv active):

    python src/run_bvbrc_recon.py

Writes one CSV per antibiotic under data/processed/ and a combined
feasibility report under reports/. Downloads no genome sequence data.
"""

from bvbrc_recon.pipeline import main

if __name__ == "__main__":
    main()
