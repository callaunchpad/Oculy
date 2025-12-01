"""Split a merged OpenSignals/keypress CSV into train/val/test sets by movement groups."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import pandas as pd

## python3 split_test_train_0.1_buffer_no_label_change.py --input test_output_devin.csv --output-dir test_splits
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


def _build_movement_ids(
    labels: pd.Series, timestamps: pd.Series, buffer_seconds: float = 0.01
) -> pd.Series:
    """Create contiguous movement identifiers whenever labels change, with buffer zones.
    
    For each label transition at time T:
    - The previous movement is extended forward to include rows up to T + buffer
    - The next movement is extended backward to include rows from T - buffer
    """

    label_changes = labels.fillna("__NA__").ne(labels.fillna("__NA__").shift(fill_value="__NA__"))
    base_movement_ids = label_changes.cumsum()
    
    # Find transition points (where labels change)
    transition_indices = labels.index[label_changes].tolist()
    
    # Convert buffer to milliseconds
    buffer_ms = buffer_seconds * 1000.0
    
    # Create a copy to modify
    movement_ids = base_movement_ids.copy()
    
    # For each transition, extend the previous movement forward and next movement backward
    for trans_idx in transition_indices:
        if trans_idx == labels.index[0]:  # Skip first row (no previous movement)
            continue
        
        trans_timestamp = timestamps.loc[trans_idx]
        buffer_start = trans_timestamp - buffer_ms
        buffer_end = trans_timestamp + buffer_ms
        
        # Get the movement IDs before and after transition
        prev_idx = labels.index[labels.index < trans_idx]
        if len(prev_idx) == 0:
            continue
        prev_movement_id = base_movement_ids.loc[prev_idx[-1]]
        next_movement_id = base_movement_ids.loc[trans_idx]
        
        # Extend previous movement forward: include rows with timestamps in (T, T + buffer]
        # These rows currently have the next movement ID, but we extend the previous movement to include them
        prev_mask = (timestamps > trans_timestamp) & (timestamps <= buffer_end)
        movement_ids.loc[prev_mask] = prev_movement_id
        
        # Extend next movement backward: include rows with timestamps in [T - buffer, T)
        # These rows currently have the previous movement ID, but we extend the next movement to include them
        next_mask = (timestamps >= buffer_start) & (timestamps < trans_timestamp)
        movement_ids.loc[next_mask] = next_movement_id
    
    movement_ids.name = "movement_id"
    return movement_ids


def _movement_order(df: pd.DataFrame, timestamp_col: str) -> List[int]:
    """Return movement IDs sorted by their starting timestamp."""

    grouped = df.groupby("_movement_id")[timestamp_col].min()
    return grouped.sort_values().index.tolist()


def _movement_label(df: pd.DataFrame, movement_id: int, label_col: str) -> str:
    """Return the lowercase label associated with the provided movement ID."""

    mask = df["_movement_id"] == movement_id
    if not mask.any():
        return ""
    first_value = df.loc[mask, label_col].iloc[0]
    return str(first_value).strip().lower()


def _movement_indices(df: pd.DataFrame, movement_id: int) -> pd.Index:
    """Return the dataframe indices that belong to the given movement."""

    mask = df["_movement_id"] == movement_id
    if not mask.any():
        return pd.Index([])
    return df.index[mask]


def _movement_metadata(
    df: pd.DataFrame, label_col: str, timestamp_col: str
) -> List[Dict[str, object]]:
    """Collect metadata for each movement_id in chronological order."""

    metadata: List[Dict[str, object]] = []
    for movement_id, group in df.groupby("_movement_id"):
        indices = df.index[df["_movement_id"] == movement_id]
        if indices.empty:
            continue
        start_idx = int(indices[0])
        end_idx = int(indices[-1])
        timestamp = float(df.at[start_idx, timestamp_col])
        label = str(df.at[start_idx, label_col]).strip().lower()
        metadata.append(
            {
                "movement_id": movement_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "size": len(indices),
                "label": label,
                "start_ts": timestamp,
            }
        )
    metadata.sort(key=lambda item: item["start_ts"])
    return metadata


def _expand_window_indices(
    start_idx: int, end_idx: int, target_rows: int, total_rows: int
) -> List[int]:
    """Return a list of dataframe indices spanning target_rows samples."""

    if target_rows <= 0 or total_rows <= 0:
        return list(range(start_idx, end_idx + 1))

    current_len = end_idx - start_idx + 1
    if current_len >= target_rows:
        excess = current_len - target_rows
        trim_before = excess // 2
        new_start = start_idx + trim_before
        return list(range(new_start, new_start + target_rows))

    needed = target_rows - current_len
    extend_before = needed // 2
    extend_after = needed - extend_before

    start = start_idx - extend_before
    end = end_idx + extend_after
    if start < 0:
        end += -start
        start = 0
    if end >= total_rows:
        shift = end - (total_rows - 1)
        start = max(0, start - shift)
        end = total_rows - 1

    window_len = end - start + 1
    if window_len < target_rows:
        deficit = target_rows - window_len
        if start == 0:
            end = min(total_rows - 1, end + deficit)
        else:
            start = max(0, start - deficit)
        window_len = end - start + 1

    if window_len > target_rows:
        excess = window_len - target_rows
        trim_before = excess // 2
        start += trim_before
        end = start + target_rows - 1

    if end >= total_rows:
        end = total_rows - 1
        start = max(0, end - target_rows + 1)

    return list(range(start, end + 1))


def _normalize_movement_windows(
    df: pd.DataFrame,
    label_col: str,
    timestamp_col: str,
    target_labels: Iterable[str] = ("left", "right", "up", "down", "blink"),
) -> Tuple[pd.DataFrame, int, float]:
    """
    Create a new dataframe where each movement_id spans an equal number of rows.

    Non-'stare' movements are padded using rows before/after their original span.
    'stare' movements are cropped to the same length. Rows may appear in multiple
    movement windows to preserve context around each event.
    """

    if df.empty or "_movement_id" not in df.columns:
        return df, 0, 0.0

    normalized_targets: Set[str] = {label.strip().lower() for label in target_labels}
    movements = _movement_metadata(df, label_col, timestamp_col)
    target_rows = max(
        (movement["size"] for movement in movements if movement["label"] in normalized_targets),
        default=0,
    )
    if target_rows <= 0:
        return df, 0, 0.0

    total_rows = len(df)
    padded_chunks: List[pd.DataFrame] = []
    for movement in movements:
        indices = _expand_window_indices(
            movement["start_idx"],
            movement["end_idx"],
            target_rows,
            total_rows,
        )
        chunk = df.iloc[indices].copy()
        chunk["_movement_id"] = movement["movement_id"]
        padded_chunks.append(chunk)

    normalized_df = pd.concat(padded_chunks, ignore_index=True)

    diffs = df[timestamp_col].diff().dropna()
    median_step = float(diffs.median()) if not diffs.empty else 1.0
    approx_ms = target_rows * median_step

    return normalized_df, target_rows, approx_ms


def _write_split(df: pd.DataFrame, groups: Sequence[int], output_path: Path) -> int:
    """Persist a subset of the dataframe corresponding to the provided groups."""

    subset = df[df["_movement_id"].isin(groups)].copy()
    subset["movement_id"] = subset["_movement_id"].astype(int)
    subset.drop(columns="_movement_id", inplace=True)
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
    timestamp_column: str = "timestamp_epoch_ms",
    buffer_seconds: float = 0.1,
) -> None:
    """Split the dataset into train/val/test sets, keeping whole movements together."""

    ratios = _validate_ratios(train_ratio, val_ratio, test_ratio)
    df = pd.read_csv(input_csv)
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' was not found in {input_csv}.")
    if timestamp_column not in df.columns:
        raise ValueError(f"Column '{timestamp_column}' was not found in {input_csv}.")

    df = df.copy()
    df.sort_values(timestamp_column, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["_movement_id"] = _build_movement_ids(
        df[label_column], df[timestamp_column], buffer_seconds=buffer_seconds
    )
    normalized_df, target_rows, window_ms = _normalize_movement_windows(
        df, label_column, timestamp_column
    )
    if target_rows > 0:
        print(
            "Normalized movement windows to "
            f"{target_rows} samples (~{window_ms:.0f} ms) based on longest non-'stare' event."
        )
        df = normalized_df
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
    parser.add_argument(
        "--timestamp-column",
        default="timestamp_epoch_ms",
        help="Name of the timestamp column for buffer calculations (default: %(default)s).",
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=0.1,
        help="Buffer time in seconds to include before and after each event split (default: %(default)s).",
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
        timestamp_column=args.timestamp_column,
        buffer_seconds=args.buffer_seconds,
    )


if __name__ == "__main__":
    main()
