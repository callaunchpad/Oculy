"""Split a merged OpenSignals/keypress CSV into train/val/test sets by movement groups."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


def _validate_ratios(train: float, val: float, test: float | None) -> Dict[str, float]:
    """Ensure the provided ratios are positive and sum to 1."""

    if test is None:
        test = 1.0 - train - val

    ratios = {"train": float(train), "val": float(val), "test": float(test)}
    for name, value in ratios.items():
        if value <= 0 or value >= 1:
            raise ValueError(f"{name} ratio must be between 0 and 1, got {value}")

    total = sum(ratios.values())
    if not 0.999 <= total <= 1.001:
        raise ValueError(
            f"Ratios must sum to 1. Got train={ratios['train']}, "
            f"val={ratios['val']}, test={ratios['test']} (sum={total})."
        )
    return ratios


def _assign_groups(
    group_sizes: pd.Series, ratios: Dict[str, float], seed: int
) -> Dict[str, List[int]]:
    """Assign contiguous movement groups to splits while respecting ratios."""

    total_rows = int(group_sizes.sum())
    train_threshold = int(round(ratios["train"] * total_rows))
    val_threshold = train_threshold + int(round(ratios["val"] * total_rows))

    rng = random.Random(seed)
    groups: List[int] = list(group_sizes.index)
    rng.shuffle(groups)

    assignments: Dict[str, List[int]] = {"train": [], "val": [], "test": []}
    cumulative = 0
    for movement_id in groups:
        size = int(group_sizes.loc[movement_id])
        if cumulative < train_threshold:
            split = "train"
        elif cumulative < val_threshold:
            split = "val"
        else:
            split = "test"
        assignments[split].append(movement_id)
        cumulative += size

    return assignments


def _build_movement_ids(labels: pd.Series) -> pd.Series:
    """Create contiguous movement identifiers whenever labels change."""

    label_changes = labels.fillna("__NA__").ne(labels.fillna("__NA__").shift(fill_value="__NA__"))
    movement_ids = label_changes.cumsum()
    movement_ids.name = "movement_id"
    return movement_ids


def _write_split(df: pd.DataFrame, groups: Sequence[int], output_path: Path) -> int:
    """Persist a subset of the dataframe corresponding to the provided groups."""

    subset = df[df["_movement_id"].isin(groups)].drop(columns="_movement_id")
    subset.to_csv(output_path, index=False)
    return len(subset)


def split_dataset(
    input_csv: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float | None,
    label_column: str,
    seed: int,
) -> None:
    """Split the dataset into train/val/test sets, keeping whole movements together."""

    ratios = _validate_ratios(train_ratio, val_ratio, test_ratio)
    df = pd.read_csv(input_csv)
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' was not found in {input_csv}.")

    df = df.copy()
    df["_movement_id"] = _build_movement_ids(df[label_column])
    group_sizes = df.groupby("_movement_id").size()

    assignments = _assign_groups(group_sizes, ratios, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split_name, groups in assignments.items():
        output_file = output_dir / f"{input_csv.stem}_{split_name}.csv"
        counts[split_name] = _write_split(df, groups, output_file)
        print(
            f"Wrote {counts[split_name]} rows ({len(groups)} movements) to {output_file}"
        )

    total_rows = sum(counts.values())
    print(
        f"Totals -> train: {counts['train']}, val: {counts['val']}, "
        f"test: {counts['test']} (total {total_rows})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test by movement.")
    parser.add_argument("--input", required=True, help="Path to the merged CSV file.")
    parser.add_argument(
        "--output-dir",
        default="splits",
        help="Directory where the split CSV files will be stored (default: %(default)s).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data to allocate to the training set (default: %(default)s).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data to allocate to the validation set (default: %(default)s).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=None,
        help="Fraction for the test set. If omitted, it is computed as 1 - train - val.",
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the column that contains gesture/movement labels (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle movement groups before splitting (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input)
    output_dir = Path(args.output_dir)
    split_dataset(
        input_csv=input_csv,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        label_column=args.label_column,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
