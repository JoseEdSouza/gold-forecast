"""
migrate_emb_to_list.py
======================
Converts a Parquet file from the old wide format (one column per embedding
dimension, e.g. emb_nomic_1d_dim0 … emb_nomic_1d_dim767) to the new compact
format (one list[float] column per embedding, e.g. emb_nomic_1d).

Also handles deviation columns (emb_*_dev_1d_7d_dim*  →  emb_*_dev_1d_7d).

Usage:
  python migrate_emb_to_list.py gold_embeddings.parquet
  python migrate_emb_to_list.py gold_embeddings.parquet --output gold_v2.parquet
"""

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Matches:  emb_nomic_1d_dim0  /  emb_3_small_7d_dim42  /  emb_nomic_dev_1d_7d_dim0
_DIM_RE = re.compile(r"^(emb_.+)_dim\d+$")


def _find_prefixes(columns: list[str]) -> dict[str, list[str]]:
    """Groups dim columns by their common prefix."""
    groups: dict[str, list[str]] = {}
    for col in columns:
        m = _DIM_RE.match(col)
        if m:
            prefix = m.group(1)
            groups.setdefault(prefix, []).append(col)
    # Sort each group by dim index so the list order is correct
    for prefix in groups:
        groups[prefix].sort(key=lambda c: int(c.rsplit("_dim", 1)[1]))
    return groups


def migrate(input_path: Path, output_path: Path) -> None:
    log.info("─── migrate ────────────────────────────────────────")
    log.info("    input  : %s", input_path)

    df = pd.read_parquet(input_path)
    log.info("    shape  : %s", df.shape)

    groups = _find_prefixes(df.columns.tolist())

    if not groups:
        log.info("    no dim columns found — nothing to migrate")
        return

    drop_cols = []
    for prefix, dim_cols in groups.items():
        arr = df[dim_cols].to_numpy()
        df[prefix] = arr.tolist()
        drop_cols.extend(dim_cols)
        log.info("    ✓ %-30s  %d rows × %d dims", prefix, len(arr), arr.shape[1])

    df = df.drop(columns=drop_cols)
    log.info("    dropped %d dim columns", len(drop_cols))

    df.to_parquet(output_path, index=False, compression="zstd")
    log.info("    saved  : %s  %s", output_path, df.shape)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Migrate emb dim columns to list[float]."
    )
    parser.add_argument("input", help="Input parquet file (old wide format)")
    parser.add_argument(
        "--output", default=None, help="Output path (default: input_v2.parquet)"
    )
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output) if args.output else inp.with_stem(inp.stem + "_v2")

    migrate(inp, out)
