"""Utilities to parse OpenSignals recordings and align them with keypress labels."""

from __future__ import annotations

import argparse
import json
import re
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

    return metadata, df


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

    if opensignals_rate <= 0:
        raise ValueError(f"Invalid OpenSignals sampling rate: {opensignals_rate}")
    if keypress_rate <= 0:
        raise ValueError(f"Invalid keypress sampling rate: {keypress_rate}")

    samples_per_keypress = opensignals_rate / keypress_rate
    if samples_per_keypress <= 0:
        raise ValueError(
            "Computed samples_per_keypress <= 0. Check sampling rates in the input files."
        )

    open_indices = np.arange(len(opensignals_df), dtype=float)
    raw_keypress_position = open_indices / samples_per_keypress
    keypress_indices = np.floor(raw_keypress_position).astype(int)
    max_idx = len(keypress_df) - 1

    if keypress_indices.max() > max_idx:
        print(
            "Warning: Keypress data is shorter than OpenSignals duration. "
            "Repeating the final label for remaining samples."
        )
        overflow_mask = keypress_indices > max_idx
        keypress_indices[overflow_mask] = max_idx
        raw_keypress_position[overflow_mask] = keypress_indices[overflow_mask]

    within_fraction = raw_keypress_position - keypress_indices
    within_fraction = np.clip(within_fraction, 0.0, 1.0)

    keypress_interval_ms = 1000.0 / keypress_rate
    label_values = keypress_df[label_col].to_numpy()
    timestamp_values = keypress_df[timestamp_col].to_numpy()
    elapsed_values = keypress_df[elapsed_col].to_numpy()
    sample_values = keypress_df[sample_col].to_numpy()

    combined = opensignals_df.reset_index(drop=True).copy()
    combined["label"] = label_values[keypress_indices]
    combined["keypress_sample_number"] = sample_values[keypress_indices]
    combined["keypress_elapsed_ms"] = (
        elapsed_values[keypress_indices] + within_fraction * keypress_interval_ms
    )
    combined["keypress_timestamp_ms"] = (
        timestamp_values[keypress_indices] + within_fraction * keypress_interval_ms
    )

    return metadata, keypress_metadata, combined


def _write_output(df: pd.DataFrame, output_path: str) -> str:
    """Persist the combined DataFrame based on output extension."""

    df.to_csv(output_path)
    return output_path


def main() -> None:

    open_signals_path = "data/sample_data/opensignals_84BA20AEBFDA_2025-10-04_12-25-49.txt"
    keypress_path = "keypress_labels_2025-11-02_18-00-51.txt"

    metadata, keypress_metadata, combined_df = merge_opensignals_and_keypress(
        open_signals_path, keypress_path
    )

    print("=== OpenSignals metadata ===")
    print(json.dumps(metadata, indent=2))
    print("\n=== Keypress metadata ===")
    print(json.dumps(keypress_metadata, indent=2))
    print(f"\n=== Combined data (first 5 rows) ===")
    print(combined_df.head())

    output_path = "test_output_1.csv"
    saved_path = _write_output(combined_df, output_path)
    print(f"\nCombined data saved to {saved_path}")


if __name__ == "__main__":
    main()
