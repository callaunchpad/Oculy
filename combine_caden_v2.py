"""Combine Caden v2 data from subfolders caden1-caden5 into combined CSVs."""

import json
import re
import glob
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


# --- Helper functions copied and adapted from parse_opensignals.py ---

def _read_header_and_data(file_path: Path) -> Tuple[List[str], List[str]]:
    """Split a text file into header (comment) lines and data lines."""
    header_lines: List[str] = []
    data_lines: List[str] = []
    in_header = True

    with open(file_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if in_header and line.startswith("#"):
                if line.startswith("# EndOfHeader"):
                    in_header = False
                    continue
                header_lines.append(line)
                continue
            in_header = False
            data_lines.append(line)

    return header_lines, data_lines


def parse_opensignals_txt(file_path: str | Path) -> Tuple[Dict, pd.DataFrame]:
    """Parse an OpenSignals .txt file into metadata (dict) and a pandas DataFrame."""
    file_path = Path(file_path)
    header_lines, data_lines = _read_header_and_data(file_path)

    metadata: Dict = {}
    for line in header_lines:
        if line.startswith("# {"):
            try:
                metadata = json.loads(line[2:])  # strip "# " prefix
            except json.JSONDecodeError:
                pass

    device_info = list(metadata.values())[0] if metadata else {}
    columns = device_info.get("column", [])

    numeric_rows: List[List[float]] = []
    for row in data_lines:
        try:
            numeric_rows.append([float(value) for value in row.split()])
        except ValueError as exc:
            raise ValueError(f"Failed to parse numeric row in {file_path}: {row}") from exc

    if not numeric_rows:
        raise ValueError(f"No sample rows found in OpenSignals file: {file_path}")

    inferred_columns: Iterable[str] | None = (
        columns if columns and len(columns) == len(numeric_rows[0]) else None
    )
    df = pd.DataFrame(numeric_rows, columns=inferred_columns)

    if "timestamp_epoch_ms" not in df.columns:
        epoch_series = _compute_timestamp_epoch(metadata, len(df))
        if epoch_series is not None:
            df["timestamp_epoch_ms"] = epoch_series

    return metadata, df


def _compute_timestamp_epoch(
    metadata: Dict, num_rows: int, timezone_offset_ms: float | None = None
) -> np.ndarray | None:
    """Return per-sample epoch timestamps (ms) if metadata supplies date/time."""
    if num_rows <= 0:
        return None

    start_epoch_ms = _metadata_epoch_start_ms(metadata)
    sampling_rate = _extract_opensignals_sampling_rate(metadata)
    if start_epoch_ms is None or sampling_rate <= 0:
        return None

    if timezone_offset_ms is not None:
        start_epoch_ms += timezone_offset_ms

    ms_per_sample = 1000.0 / sampling_rate
    return start_epoch_ms + np.arange(num_rows, dtype=float) * ms_per_sample


def _extract_opensignals_sampling_rate(metadata: Dict, default: float = 1000.0) -> float:
    """Return the sampling rate (Hz) declared in the OpenSignals metadata."""
    if not metadata:
        return default

    device_info = next(iter(metadata.values()), {})
    rate = device_info.get("sampling rate")
    try:
        return float(rate)
    except (TypeError, ValueError):
        return default


def _metadata_epoch_start_ms(metadata: Dict) -> float | None:
    """Parse the recording start timestamp from metadata and convert to epoch ms."""
    if not metadata:
        return None

    device_info = next(iter(metadata.values()), {})
    date_str = device_info.get("date")
    time_str = device_info.get("time")
    if not date_str or not time_str:
        return None

    date_time = _parse_datetime_string(f"{date_str} {time_str}")
    if not date_time:
        return None

    start_dt = date_time.replace(tzinfo=timezone.utc)
    return start_dt.timestamp() * 1000.0


def _parse_datetime_string(value: str) -> datetime | None:
    """Parse a variety of datetime string formats."""
    value = value.strip()
    formats = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d")
    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def parse_keypress_labels(file_path: str | Path) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Parse the keypress label file produced by record_keypresses.py."""
    file_path = Path(file_path)
    header_lines, data_lines = _read_header_and_data(file_path)

    header_info: Dict[str, str] = {}
    columns: List[str] = []
    for line in header_lines:
        trimmed = line[1:].strip()  # remove leading '#'
        if ":" in trimmed:
            key, value = trimmed.split(":", 1)
            header_info[key.strip()] = value.strip()
        if trimmed.lower().startswith("columns:"):
            columns = [col.strip() for col in trimmed.split(":", 1)[1].split(",")]

    if not columns:
        columns = ["sample_number", "timestamp_ms", "elapsed_ms", "label"]

    if len(columns) < 4:
        # It's possible the header is malformed or missing columns, but we'll try with default
         if len(data_lines) > 0:
             # Basic check if it has 4 columns
             if len(data_lines[0].split('\t')) >= 4:
                  columns = ["sample_number", "timestamp_ms", "elapsed_ms", "label"]
             else:
                raise ValueError(
                    f"Expected at least 4 columns for keypress data but found {len(columns)}: {columns}"
                )

    sample_col, timestamp_col, elapsed_col, label_col = columns[:4]

    records: List[Dict[str, object]] = []
    for row in data_lines:
        parts = [part.strip() for part in row.split("\t") if part.strip()]
        if len(parts) != len(columns):
            parts = row.split()
        if len(parts) != len(columns):
            # Skip malformed lines if any (or raise error)
            # raise ValueError(f"Row in {file_path} has {len(parts)} columns but expected {len(columns)}: {row}")
            continue

        record = {
            sample_col: int(parts[0]),
            timestamp_col: float(parts[1]),
            elapsed_col: float(parts[2]),
            label_col: parts[3],
        }
        records.append(record)

    df = pd.DataFrame(records, columns=columns)
    header_info.setdefault("sample_column", sample_col)
    header_info.setdefault("timestamp_column", timestamp_col)
    header_info.setdefault("elapsed_column", elapsed_col)
    header_info.setdefault("label_column", label_col)
    return header_info, df


def _infer_keypress_sampling_rate(header_info: Dict[str, str], default: float = 1000.0) -> float:
    """Best-effort parse of the sampling rate from the keypress header info."""
    for key, value in header_info.items():
        if key.lower().startswith("sampling rate"):
            match = re.search(r"(\d+(?:\.\d+)?)\s*hz", value, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
    return default


def _infer_timezone_offset_ms(
    header_info: Dict[str, str],
    keypress_df: pd.DataFrame,
    timestamp_col: str,
    elapsed_col: str,
) -> float | None:
    """Estimate timezone offset between header timestamps and epoch values."""
    if keypress_df.empty:
        return None

    start_value = None
    for key, value in header_info.items():
        if key.lower().startswith("recording started"):
            start_value = value
            break

    if not start_value:
        return None

    start_dt = _parse_datetime_string(start_value)
    if not start_dt:
        return None

    header_epoch_ms = start_dt.replace(tzinfo=timezone.utc).timestamp() * 1000.0

    timestamps = keypress_df[timestamp_col].to_numpy(dtype=float)
    elapsed = keypress_df[elapsed_col].to_numpy(dtype=float)
    sample_window = min(100, len(timestamps))
    # Avoid subtract if empty
    if sample_window == 0:
        return 0.0
    approx_start_epoch = float(np.median(timestamps[:sample_window] - elapsed[:sample_window]))

    offset = approx_start_epoch - header_epoch_ms
    return float(round(offset))


def align_opensignals_and_keypress(
    opensignals_path: str | Path,
    keypress_path: str | Path,
) -> pd.DataFrame:
    """
    Align OpenSignals samples with keypress labels based on timestamps.
    Returns a combined DataFrame with OpenSignals data and corresponding labels.
    Does NOT drop initial 'stare' or trailing data.
    """
    metadata, opensignals_df = parse_opensignals_txt(opensignals_path)
    keypress_metadata, keypress_df = parse_keypress_labels(keypress_path)

    if opensignals_df.empty or keypress_df.empty:
        print(f"Warning: Empty data in {opensignals_path} or {keypress_path}")
        return pd.DataFrame()

    label_col = keypress_metadata.get("label_column", "label")
    timestamp_col = keypress_metadata.get("timestamp_column", "timestamp_ms")
    elapsed_col = keypress_metadata.get("elapsed_column", "elapsed_ms")
    sample_col = keypress_metadata.get("sample_column", "sample_number")
    
    # Fallback column names if not in df
    if label_col not in keypress_df.columns: label_col = keypress_df.columns[3]
    if timestamp_col not in keypress_df.columns: timestamp_col = keypress_df.columns[1]
    if elapsed_col not in keypress_df.columns: elapsed_col = keypress_df.columns[2]
    if sample_col not in keypress_df.columns: sample_col = keypress_df.columns[0]

    opensignals_rate = _extract_opensignals_sampling_rate(metadata)
    keypress_rate = _infer_keypress_sampling_rate(keypress_metadata)
    timezone_offset_ms = _infer_timezone_offset_ms(
        keypress_metadata, keypress_df, timestamp_col, elapsed_col
    )

    if opensignals_rate <= 0: opensignals_rate = 1000.0
    if keypress_rate <= 0: keypress_rate = 1000.0

    # Compute OpenSignals timestamps
    epoch_values = _compute_timestamp_epoch(metadata, len(opensignals_df), timezone_offset_ms)
    if epoch_values is None:
        raise ValueError("Unable to compute OpenSignals timestamps.")
    opensignals_df["timestamp_epoch_ms"] = epoch_values

    # Determine intersection based on timestamps
    open_timestamps = opensignals_df["timestamp_epoch_ms"].to_numpy()
    open_start = float(np.min(open_timestamps))
    open_end = float(np.max(open_timestamps))

    # Trim keypress to OpenSignals range (roughly)
    trimmed_keypress_df = keypress_df[
        (keypress_df[timestamp_col] >= open_start) & (keypress_df[timestamp_col] <= open_end)
    ].copy()
    trimmed_keypress_df.sort_values(timestamp_col, inplace=True)
    trimmed_keypress_df.reset_index(drop=True, inplace=True)

    if trimmed_keypress_df.empty:
        print(f"Warning: No keypress overlap for {opensignals_path}")
        return pd.DataFrame()

    # Determine exact overlap window
    key_min = float(trimmed_keypress_df[timestamp_col].min())
    key_max = float(trimmed_keypress_df[timestamp_col].max())
    
    # Filter OpenSignals to the Keypress window
    open_mask = (open_timestamps >= key_min) & (open_timestamps <= key_max)
    if not np.any(open_mask):
         print(f"Warning: No OpenSignals overlap for {keypress_path}")
         return pd.DataFrame()

    opensignals_df = opensignals_df.loc[open_mask].reset_index(drop=True)
    open_timestamps = opensignals_df["timestamp_epoch_ms"].to_numpy()
    
    # Map keypress labels to OpenSignals samples
    keypress_interval_ms = 1000.0 / keypress_rate
    keypress_timestamps = trimmed_keypress_df[timestamp_col].to_numpy()
    
    # For each OpenSignals sample, find the preceding Keypress sample
    # searchsorted returns index such that t[i-1] <= val < t[i] if side='right'
    keypress_indices = np.searchsorted(keypress_timestamps, open_timestamps, side="right") - 1
    keypress_indices = np.clip(keypress_indices, 0, len(trimmed_keypress_df) - 1)

    label_values = trimmed_keypress_df[label_col].to_numpy()
    timestamp_values = trimmed_keypress_df[timestamp_col].to_numpy()
    elapsed_values = trimmed_keypress_df[elapsed_col].to_numpy()
    sample_values = trimmed_keypress_df[sample_col].to_numpy()

    combined = opensignals_df.copy()
    combined["label"] = label_values[keypress_indices]
    combined["keypress_sample_number"] = sample_values[keypress_indices]
    combined["keypress_timestamp_ms"] = timestamp_values[keypress_indices]
    
    # Optional: Calculate interpolation fraction if needed, but labels are discrete.
    # We just take the nearest previous label.

    return combined


def main():
    root_dir = Path("data/caden/v2")
    subdirs = [f"caden{i}" for i in range(1, 6)]
    
    all_combined_dfs = []

    for subdir_name in subdirs:
        subdir_path = root_dir / subdir_name
        if not subdir_path.exists():
            print(f"Directory not found: {subdir_path}")
            continue

        # Find opensignals file
        opensignals_files = list(subdir_path.glob("opensignals*.txt"))
        # Find keypress file
        keypress_files = list(subdir_path.glob("keypress*.txt"))

        if not opensignals_files or not keypress_files:
            print(f"Missing files in {subdir_name}: opensignals={len(opensignals_files)}, keypress={len(keypress_files)}")
            continue
        
        # Assume one of each per folder
        os_file = opensignals_files[0]
        kp_file = keypress_files[0]
        
        print(f"Processing {subdir_name}...")
        combined = align_opensignals_and_keypress(os_file, kp_file)
        
        if not combined.empty:
            combined["session_id"] = subdir_name
            all_combined_dfs.append(combined)

    if not all_combined_dfs:
        print("No data combined.")
        return

    full_df = pd.concat(all_combined_dfs, ignore_index=True)

    # Override "up" and "down" labels with "stare"
    if "label" in full_df.columns:
        print("Overriding 'up' and 'down' labels with 'stare'...")
        full_df["label"] = full_df["label"].replace(["up", "down"], "stare")
    
    # Split into Data (OpenSignals) and Labels (Keypress)
    # OpenSignals columns: usually 0, 1, 2, 3 (digital), then A1-A6.
    # The columns from OpenSignals parsing are numeric inputs 0-3 then A1-A6.
    # Inferred columns might be ['A1', 'A2'...] or just indices if headers missing.
    # Let's inspect columns.
    
    print("Columns:", full_df.columns.tolist())
    
    # Identify label columns
    label_cols = ["label", "keypress_sample_number", "keypress_timestamp_ms", "session_id", "timestamp_epoch_ms"]
    
    # Identify data columns (everything else)
    data_cols = [c for c in full_df.columns if c not in label_cols]
    # We should probably keep 'session_id' and 'timestamp_epoch_ms' in data csv too for linking?
    # User asked for "one with the data (opensignals), and one with the labels (keypress)".
    # Usually data CSV has just the features. But alignment key is useful.
    # I will include timestamp in both.
    
    data_df = full_df[data_cols + ["timestamp_epoch_ms", "session_id"]].copy()
    labels_df = full_df[label_cols].copy()
    
    # Reorder columns for neatness
    # Data: timestamp, session, signals...
    # Labels: timestamp, session, label...
    
    # Save
    data_out = root_dir / "combined_caden_v2_data.csv"
    labels_out = root_dir / "combined_caden_v2_labels.csv"
    
    data_df.to_csv(data_out, index=False)
    labels_df.to_csv(labels_out, index=False)
    
    print(f"Saved {len(full_df)} rows.")
    print(f"Data CSV: {data_out}")
    print(f"Labels CSV: {labels_out}")

if __name__ == "__main__":
    main()

