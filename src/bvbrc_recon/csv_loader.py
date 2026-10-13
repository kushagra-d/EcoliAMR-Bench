"""Read-only loader for the existing per-antibiotic reconnaissance CSVs.

Exists because a naive ``pandas.read_csv`` auto-infers ``genome_id`` as
float64 when every value looks like a decimal (e.g. "562.229500"), which
silently drops trailing zeros. Verified against the real CSVs: this
actually merges distinct genome_id strings in this dataset (fewer unique
float values than unique string values for all three antibiotics) - not
a theoretical risk. This loader forces `genome_id` and `id` to stay
strings, and never writes back to the source file.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd


def load_antibiotic_csv(path: Path) -> list[dict]:
    df = pd.read_csv(path, dtype={"genome_id": str, "id": str})
    records = df.to_dict("records")
    # `.where(df.notna(), None)` on the DataFrame does not reliably survive
    # column dtype coercion for numeric columns (NaN can come back instead
    # of None), so NaN -> None is applied explicitly per value instead.
    for record in records:
        for key, value in record.items():
            if isinstance(value, float) and math.isnan(value):
                record[key] = None
    return records
