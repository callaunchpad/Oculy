"""Utilities to parse OpenSignals recordings and align them with keypress labels."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


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
    """
    Parse an OpenSignals .txt file into metadata (dict) and a pandas DataFrame.

    Args:
        file_path: Path to the OpenSignals .txt file.

    Returns:
        metadata: Header metadata extracted from JSON.
        df: Data section as a pandas DataFrame.
    """

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
        except ValueError as exc:  # pragma: no cover - defensive guard rail
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
            raise ValueError(
                f"Row in {file_path} has {len(parts)} columns but expected {len(columns)}: {row}"
            )

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
    approx_start_epoch = float(np.median(timestamps[:sample_window] - elapsed[:sample_window]))

    offset = approx_start_epoch - header_epoch_ms
    return float(round(offset))


def merge_opensignals_and_keypress(
    opensignals_path: str | Path,
    keypress_path: str | Path,
) -> Tuple[Dict, Dict[str, str], pd.DataFrame]:
    """
    Align OpenSignals samples with keypress labels based on the declared sampling rates.

    Each keypress label covers a 1 ms window. All OpenSignals samples that fall inside the
    same 1 ms window receive that label and derive adjusted timestamp/elapsed values using
    the OpenSignals sampling interval.
    """

    metadata, opensignals_df = parse_opensignals_txt(opensignals_path)
    keypress_metadata, keypress_df = parse_keypress_labels(keypress_path)

    if opensignals_df.empty or keypress_df.empty:
        raise ValueError("One of the input files does not contain any data rows.")

    label_col = keypress_metadata.get("label_column", keypress_df.columns[3])
    timestamp_col = keypress_metadata.get("timestamp_column", keypress_df.columns[1])
    elapsed_col = keypress_metadata.get("elapsed_column", keypress_df.columns[2])
    sample_col = keypress_metadata.get("sample_column", keypress_df.columns[0])

    opensignals_rate = _extract_opensignals_sampling_rate(metadata)
    keypress_rate = _infer_keypress_sampling_rate(keypress_metadata)
    timezone_offset_ms = _infer_timezone_offset_ms(
        keypress_metadata, keypress_df, timestamp_col, elapsed_col
    )

    if opensignals_rate <= 0:
        raise ValueError(f"Invalid OpenSignals sampling rate: {opensignals_rate}")
    if keypress_rate <= 0:
        raise ValueError(f"Invalid keypress sampling rate: {keypress_rate}")

    epoch_values = _compute_timestamp_epoch(metadata, len(opensignals_df), timezone_offset_ms)
    if epoch_values is None:
        raise ValueError(
            "Unable to compute OpenSignals epoch timestamps from metadata; "
            "ensure the header declares date/time."
        )
    opensignals_df["timestamp_epoch_ms"] = epoch_values

    open_timestamps = opensignals_df["timestamp_epoch_ms"].to_numpy()
    open_start = float(np.min(open_timestamps))
    open_end = float(np.max(open_timestamps))

    trimmed_keypress_df = keypress_df[
        (keypress_df[timestamp_col] >= open_start) & (keypress_df[timestamp_col] <= open_end)
    ].copy()
    trimmed_keypress_df.sort_values(timestamp_col, inplace=True)
    trimmed_keypress_df.reset_index(drop=True, inplace=True)

    if trimmed_keypress_df.empty:
        raise ValueError(
            "No keypress labels fall within the OpenSignals recording interval "
            f"({open_start} ms to {open_end} ms)."
        )

    trimmed_count = len(keypress_df) - len(trimmed_keypress_df)
    if trimmed_count > 0:
        print(f"Trimmed {trimmed_count} keypress rows outside the OpenSignals window.")

    key_min = float(trimmed_keypress_df[timestamp_col].min())
    key_max = float(trimmed_keypress_df[timestamp_col].max())
    open_mask = (open_timestamps >= key_min) & (open_timestamps <= key_max)
    if not np.any(open_mask):
        raise ValueError(
            "OpenSignals recording does not overlap the keypress timestamp interval "
            f"({key_min} ms to {key_max} ms)."
        )

    dropped_open = int(len(opensignals_df) - open_mask.sum())
    if dropped_open > 0:
        print(f"Dropped {dropped_open} OpenSignals samples outside the keypress window.")
    opensignals_df = opensignals_df.loc[open_mask].reset_index(drop=True)
    open_timestamps = opensignals_df["timestamp_epoch_ms"].to_numpy()
    open_start = float(np.min(open_timestamps))
    open_end = float(np.max(open_timestamps))

    trimmed_keypress_df = trimmed_keypress_df[
        (trimmed_keypress_df[timestamp_col] >= open_start)
        & (trimmed_keypress_df[timestamp_col] <= open_end)
    ].reset_index(drop=True)
    if trimmed_keypress_df.empty:
        raise ValueError(
            "All keypress labels were trimmed after aligning with the OpenSignals window."
        )

    samples_per_keypress = opensignals_rate / keypress_rate
    if samples_per_keypress <= 0:
        raise ValueError(
            "Computed samples_per_keypress <= 0. Check sampling rates in the input files."
        )

    keypress_interval_ms = 1000.0 / keypress_rate
    keypress_timestamps = trimmed_keypress_df[timestamp_col].to_numpy()
    keypress_indices = np.searchsorted(keypress_timestamps, open_timestamps, side="right") - 1
    keypress_indices = np.clip(keypress_indices, 0, len(trimmed_keypress_df) - 1)

    within_fraction = (open_timestamps - keypress_timestamps[keypress_indices]) / keypress_interval_ms
    within_fraction = np.clip(within_fraction, 0.0, 1.0)

    label_values = trimmed_keypress_df[label_col].to_numpy()
    timestamp_values = trimmed_keypress_df[timestamp_col].to_numpy()
    elapsed_values = trimmed_keypress_df[elapsed_col].to_numpy()
    sample_values = trimmed_keypress_df[sample_col].to_numpy()

    combined = opensignals_df.reset_index(drop=True).copy()
    combined["label"] = label_values[keypress_indices]
    combined["keypress_sample_number"] = sample_values[keypress_indices]
    combined["keypress_elapsed_ms"] = (
        elapsed_values[keypress_indices] + within_fraction * keypress_interval_ms
    )
    combined["keypress_timestamp_ms"] = (
        timestamp_values[keypress_indices] + within_fraction * keypress_interval_ms
    )

    non_stare_mask = combined["label"].astype(str).str.lower() != "stare"
    if non_stare_mask.any():
        first_idx = int(np.argmax(non_stare_mask.to_numpy()))
        if first_idx > 0:
            print(f"Dropping {first_idx} baseline samples before first non-'stare' label.")
            combined = combined.iloc[first_idx:].reset_index(drop=True)
    else:
        print("Warning: No non-'stare' labels found; retaining all rows.")

    if not combined.empty:
        end_timestamp = float(combined["timestamp_epoch_ms"].iloc[-1])
        cutoff = end_timestamp - 2000.0
        trimmed_combined = combined[combined["timestamp_epoch_ms"] < cutoff].reset_index(drop=True)
        removed_tail = len(combined) - len(trimmed_combined)
        if removed_tail > 0:
            print(f"Removed {removed_tail} samples from the final 2 seconds of recording.")
        combined = trimmed_combined

    return metadata, keypress_metadata, combined


def _write_output(df: pd.DataFrame, output_path: str) -> str:
    """Persist the combined DataFrame based on output extension."""

    df.to_csv(output_path)
    return output_path


def main() -> None:

    open_signals_path = "/Users/adelinac/Documents/Oculy/data/Devin/Devinsample.txt"
    keypress_path = "/Users/adelinac/Documents/Oculy/data/Devin/Devin-macbookkeypress_labels_2025-11-09_18-10-46.txt"

    metadata, keypress_metadata, combined_df = merge_opensignals_and_keypress(
        open_signals_path, keypress_path
    )

    print("=== OpenSignals metadata ===")
    print(json.dumps(metadata, indent=2))
    print("\n=== Keypress metadata ===")
    print(json.dumps(keypress_metadata, indent=2))
    print(f"\n=== Combined data (first 5 rows) ===")
    print(combined_df.head())

    output_path = "test_output_devin.csv"
    saved_path = _write_output(combined_df, output_path)
    print(f"\nCombined data saved to {saved_path}")


if __name__ == "__main__":
    main()
