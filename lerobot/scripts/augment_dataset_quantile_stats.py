#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add quantile statistics (q01/q99) to an existing LeRobot dataset.

`NormalizationMode.QUANTILES` — which pi0.5 uses for state and actions — needs
q01/q99 per feature, but datasets built before this only carry
min/max/mean/std/count. This script computes them from the parquet files and
writes them back into `meta/episodes_stats.jsonl` (and `meta/stats.json` when
present).

Quantiles are computed over the *whole* dataset and the same values are stored
on every episode. That is deliberate: normalization should use a dataset-wide
scale, and it makes the count-weighted averaging in `aggregate_feature_stats`
exact, so training on an episode subset gets the same scale as training on all
of them.

Only numeric features are processed. Image and video features are skipped:
pi0.5 normalizes them with IDENTITY, and decoding every frame to get pixel
quantiles would cost far more than it is worth.

Example:

```bash
python lerobot/scripts/augment_dataset_quantile_stats.py \
    --root lerobot_dataset/teleop-v1.4
```
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.common.datasets.utils import (
    EPISODES_STATS_PATH,
    STATS_PATH,
    load_json,
    write_json,
)

DEFAULT_QUANTILES = (0.01, 0.99)

# Features whose stats are per-channel image summaries, or bookkeeping columns
# that no policy normalizes.
SKIPPED_DTYPES = ("image", "video", "string")


def quantile_key(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def numeric_feature_keys(info: dict) -> list[str]:
    return [key for key, ft in info["features"].items() if ft["dtype"] not in SKIPPED_DTYPES]


def collect_feature_columns(root: Path, info: dict, keys: list[str]) -> dict[str, np.ndarray]:
    """Read `keys` out of every episode parquet and stack them frame-wise."""
    parquet_paths = sorted((root / "data").rglob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    logging.info(f"Reading {len(keys)} feature(s) from {len(parquet_paths)} episode file(s)")

    chunks: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for path in parquet_paths:
        table = pq.read_table(path, columns=keys)
        for key in keys:
            column = table.column(key).to_numpy(zero_copy_only=False)
            # Vector features come back as an object array of per-frame lists.
            array = np.stack([np.asarray(v) for v in column]) if column.dtype == object else column
            chunks[key].append(np.atleast_2d(array.astype(np.float64)).reshape(len(array), -1))

    return {key: np.concatenate(values, axis=0) for key, values in chunks.items()}


def compute_quantile_stats(
    root: Path, quantiles: tuple[float, ...] = DEFAULT_QUANTILES
) -> dict[str, dict[str, np.ndarray]]:
    """Compute dataset-wide quantiles for every numeric feature."""
    info = load_json(root / "meta/info.json")
    keys = numeric_feature_keys(info)
    columns = collect_feature_columns(root, info, keys)

    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, data in columns.items():
        shape = tuple(info["features"][key]["shape"])
        stats[key] = {
            quantile_key(q): np.quantile(data, q, axis=0).reshape(shape) for q in quantiles
        }
        logging.info(f"  {key}: {len(data)} frames, shape {shape}")

    return stats


def _merge_into_serialized(serialized: dict, quantile_stats: dict[str, dict[str, np.ndarray]], where: str) -> int:
    """Insert quantiles into a JSON-serializable stats dict, in place.

    A feature missing here is not skippable: `aggregate_feature_stats` only
    keeps a quantile when *every* episode has it, so one gap would silently
    drop the quantiles for the whole dataset and surface much later as an
    unrelated "requires the 'q01' statistic" error at training time.
    """
    absent = [key for key in quantile_stats if key not in serialized]
    if absent:
        raise ValueError(
            f"{where} has no stats entry for {absent}, but those features exist in the parquet files. "
            "Quantiles must cover every episode, so refusing to write a partial update."
        )

    for key, qstats in quantile_stats.items():
        for qkey, value in qstats.items():
            serialized[key][qkey] = value.tolist()
    return len(quantile_stats)


def augment_dataset(root: Path, quantiles: tuple[float, ...], overwrite: bool) -> None:
    episodes_stats_path = root / EPISODES_STATS_PATH
    stats_path = root / STATS_PATH

    if not episodes_stats_path.exists() and not stats_path.exists():
        raise FileNotFoundError(f"No stats found in {root / 'meta'}; is this a LeRobot dataset?")

    qkeys = [quantile_key(q) for q in quantiles]
    if not overwrite and episodes_stats_path.exists():
        first = json.loads(episodes_stats_path.read_text().splitlines()[0])
        present = next(iter(first["stats"].values()))
        if all(qkey in present for qkey in qkeys):
            logging.info("Dataset already has quantile stats. Use --overwrite to recompute.")
            return

    quantile_stats = compute_quantile_stats(root, quantiles)

    if episodes_stats_path.exists():
        lines = episodes_stats_path.read_text().splitlines()
        out = []
        for line in lines:
            record = json.loads(line)
            _merge_into_serialized(
                record["stats"], quantile_stats, f"episode {record['episode_index']}"
            )
            out.append(json.dumps(record))
        episodes_stats_path.write_text("\n".join(out) + "\n")
        logging.info(f"Updated {len(out)} episode(s) in {episodes_stats_path}")

    if stats_path.exists():
        serialized = load_json(stats_path)
        n = _merge_into_serialized(serialized, quantile_stats, str(stats_path))
        write_json(serialized, stats_path)
        logging.info(f"Updated {n} feature(s) in {stats_path}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Path to the dataset root directory.")
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=list(DEFAULT_QUANTILES),
        help="Quantiles to compute (default: 0.01 0.99, as expected by pi0.5).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Recompute even if quantile stats already exist."
    )
    args = parser.parse_args()

    augment_dataset(args.root.expanduser(), tuple(args.quantiles), args.overwrite)


if __name__ == "__main__":
    main()
