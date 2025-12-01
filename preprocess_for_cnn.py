"""Preprocess EOG signal data and keypress labels for 1D CNN training.

This script:
1. Loads signal and label data
2. Aligns timestamps between the two datasets
3. Creates segments from continuous regions of the same label
4. Analyzes class distribution
5. Normalizes all segments to uniform length
6. Outputs NumPy arrays ready for training
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from parse_opensignals import (
    parse_keypress_labels,
    parse_opensignals_txt,
    _extract_opensignals_sampling_rate,
    _infer_keypress_sampling_rate,
)


def extract_label_start_timestamp(label_df: pd.DataFrame, label_metadata: Dict[str, str]) -> float:
    """Extract the recording start timestamp from label data.
    
    Uses the first timestamp in the label data, which is the actual start time.
    """
    timestamp_col = label_metadata.get("timestamp_column", label_df.columns[1])
    if len(label_df) == 0:
        raise ValueError("Label dataframe is empty")
    return float(label_df[timestamp_col].iloc[0])


def extract_signal_start_timestamp(metadata: Dict) -> float:
    """Extract the recording start timestamp from OpenSignals metadata."""
    device_info = list(metadata.values())[0] if metadata else {}
    date_str = device_info.get("date", "")
    time_str = device_info.get("time", "")
    
    if not date_str or not time_str:
        raise ValueError("Could not find date/time in OpenSignals metadata")
    
    # Parse format: date="2025-11-16", time="17:32:53.625"
    try:
        dt_str = f"{date_str} {time_str}"
        dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S.%f")
        return dt.timestamp() * 1000  # Convert to milliseconds
    except ValueError:
        raise ValueError(f"Could not parse timestamp: {dt_str}")


def align_timestamps(
    signal_df: pd.DataFrame,
    signal_metadata: Dict,
    label_df: pd.DataFrame,
    label_metadata: Dict[str, str],
) -> pd.DataFrame:
    """Align signal timestamps with label timestamps based on start times."""
    # Extract start timestamps
    label_start_ms = extract_label_start_timestamp(label_df, label_metadata)
    signal_start_ms = extract_signal_start_timestamp(signal_metadata)
    
    # Calculate offset
    time_offset_ms = label_start_ms - signal_start_ms
    print(f"Label start timestamp: {label_start_ms:.3f} ms")
    print(f"Signal start timestamp: {signal_start_ms:.3f} ms")
    print(f"Time offset: {time_offset_ms:.3f} ms ({time_offset_ms/1000:.3f} seconds)")
    
    # Get sampling rates
    signal_rate = _extract_opensignals_sampling_rate(signal_metadata)
    label_rate = _infer_keypress_sampling_rate(label_metadata)
    
    # Create signal timestamps based on start time and sample indices
    # Signal samples are indexed sequentially starting from 0
    signal_sample_indices = np.arange(len(signal_df))
    signal_timestamps_ms = signal_start_ms + (signal_sample_indices / signal_rate * 1000)
    
    # Get label timestamps and values
    timestamp_col = label_metadata.get("timestamp_column", label_df.columns[1])
    label_col = label_metadata.get("label_column", label_df.columns[3])
    label_timestamps_ms = label_df[timestamp_col].values
    label_values = label_df[label_col].values
    
    # Match signal samples to labels based on timestamps
    # Signal timestamps need to be matched to label timestamps
    # Since signal starts at signal_start_ms and labels start at label_start_ms,
    # we need to find which label timestamp corresponds to each signal timestamp
    
    # Use searchsorted to find the index where each signal timestamp would be inserted
    # into the sorted label timestamps array
    # Convert to numpy array to avoid pandas Series issues
    label_timestamps_array = np.asarray(label_timestamps_ms)
    signal_timestamps_array = np.asarray(signal_timestamps_ms)
    label_indices = np.searchsorted(label_timestamps_array, signal_timestamps_array, side='left')
    
    # Handle edge cases: clip to valid range
    label_indices = np.clip(label_indices, 0, len(label_timestamps_ms) - 1)
    
    # For indices > 0, check if previous label is closer
    mask = label_indices > 0
    if np.any(mask):
        prev_closer = np.abs(label_timestamps_array[label_indices[mask] - 1] - signal_timestamps_array[mask]) < \
                      np.abs(label_timestamps_array[label_indices[mask]] - signal_timestamps_array[mask])
        label_indices[mask] = np.where(prev_closer, label_indices[mask] - 1, label_indices[mask])
    
    # Get aligned labels
    aligned_labels = label_values[label_indices]
    
    # Create aligned dataframe
    aligned_df = signal_df.copy()
    aligned_df["label"] = aligned_labels
    aligned_df["signal_timestamp_ms"] = signal_timestamps_ms
    aligned_df["label_timestamp_ms"] = label_timestamps_array[label_indices]
    
    return aligned_df


def create_segments(df: pd.DataFrame) -> List[Tuple[int, int, str]]:
    """Create segments from consecutive samples with the same label.
    
    Returns:
        List of (start_idx, end_idx, label) tuples.
        end_idx is exclusive (like Python slicing).
    """
    labels = df["label"].values
    segments = []
    
    if len(labels) == 0:
        return segments
    
    current_label = labels[0]
    start_idx = 0
    
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            # End of current segment
            segments.append((start_idx, i, current_label))
            current_label = labels[i]
            start_idx = i
    
    # Add final segment
    segments.append((start_idx, len(labels), current_label))
    
    return segments


def analyze_distribution(segments: List[Tuple[int, int, str]]) -> Dict[str, Dict]:
    """Analyze segment distribution by label class.
    
    Returns:
        Dictionary mapping label to statistics dict with keys:
        - count: number of segments
        - lengths: list of segment lengths
        - median_length: median segment length
    """
    from collections import defaultdict
    
    # Use defaultdict with a factory function
    def make_stats():
        return {"count": 0, "lengths": []}
    
    label_stats: Dict[str, Dict] = defaultdict(make_stats)
    
    for start_idx, end_idx, label in segments:
        length = end_idx - start_idx
        label_stats[label]["count"] += 1
        label_stats[label]["lengths"].append(length)
    
    # Calculate medians
    result: Dict[str, Dict] = {}
    all_lengths: List[int] = []
    
    for label, stats in label_stats.items():
        lengths = stats["lengths"]
        lengths_array = np.array(lengths)
        median_length = int(np.median(lengths_array)) if len(lengths_array) > 0 else 0
        result[label] = {
            "count": stats["count"],
            "lengths": lengths,
            "median_length": median_length,
        }
        all_lengths.extend(lengths)
    
    overall_median = int(np.median(all_lengths)) if all_lengths else 0
    result["_overall"] = {"median_length": overall_median}
    
    return result


def normalize_segment_length(
    segment_data: np.ndarray,
    target_length: int,
    method: str = "pad_truncate",
) -> np.ndarray:
    """Normalize a segment to target length.
    
    Args:
        segment_data: Array of shape (length, n_channels)
        target_length: Desired length
        method: 'pad_truncate' (pad with zeros, truncate if too long) or
                'interpolate' (use scipy interpolation)
    
    Returns:
        Normalized segment array of shape (target_length, n_channels)
    """
    current_length, n_channels = segment_data.shape
    
    if current_length == target_length:
        return segment_data
    
    if method == "pad_truncate":
        if current_length < target_length:
            # Pad with zeros
            padding = np.zeros((target_length - current_length, n_channels))
            return np.vstack([segment_data, padding])
        else:
            # Truncate
            return segment_data[:target_length]
    
    elif method == "interpolate":
        # Use scipy interpolation to resample
        if current_length < target_length:
            # Upsample
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, target_length)
            resampled = np.zeros((target_length, n_channels))
            for ch in range(n_channels):
                resampled[:, ch] = np.interp(x_new, x_old, segment_data[:, ch])
            return resampled
        else:
            # Downsample
            x_old = np.linspace(0, 1, current_length)
            x_new = np.linspace(0, 1, target_length)
            resampled = np.zeros((target_length, n_channels))
            for ch in range(n_channels):
                resampled[:, ch] = np.interp(x_new, x_old, segment_data[:, ch])
            return resampled
    
    else:
        raise ValueError(f"Unknown method: {method}")


def preprocess_for_cnn(
    signal_path: str | Path,
    label_path: str | Path,
    output_path: str | Path | None = None,
    window_length: int | None = None,
    normalization_method: str = "pad_truncate",
    channels_to_use: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Preprocess EOG data for 1D CNN training.
    
    Args:
        signal_path: Path to OpenSignals .txt file
        label_path: Path to keypress labels .txt file
        output_path: Optional path to save preprocessed data (.npz file)
        window_length: Target window length. If None, uses median segment length.
        normalization_method: Method for normalizing segment lengths
        channels_to_use: List of channel names to use (e.g., ['A4']). If None, uses all signal channels.
    
    Returns:
        X: Feature array of shape (n_segments, window_length, n_channels)
        y: Label array of shape (n_segments,)
        stats: Dictionary with preprocessing statistics
    """
    print("=" * 60)
    print("Preprocessing EOG Data for 1D CNN")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    signal_metadata, signal_df = parse_opensignals_txt(signal_path)
    label_metadata, label_df = parse_keypress_labels(label_path)
    
    print(f"  Signal samples: {len(signal_df)}")
    print(f"  Signal channels: {signal_df.shape[1]}")
    print(f"  Label samples: {len(label_df)}")
    
    # Step 2: Align timestamps
    print("\n[2/6] Aligning timestamps...")
    aligned_df = align_timestamps(signal_df, signal_metadata, label_df, label_metadata)
    
    # Step 3: Create segments
    print("\n[3/6] Creating segments...")
    segments = create_segments(aligned_df)
    print(f"  Total segments: {len(segments)}")
    
    # Step 4: Analyze distribution
    print("\n[4/6] Analyzing distribution...")
    stats = analyze_distribution(segments)
    
    print("\n  Segment Distribution:")
    print("  " + "-" * 50)
    for label in sorted([k for k in stats.keys() if not k.startswith("_")]):
        label_stats = stats[label]
        print(f"  {label:10s}: {label_stats['count']:5d} segments, "
              f"median length: {label_stats['median_length']:5d} samples")
    
    overall_median = stats["_overall"]["median_length"]
    print(f"\n  Overall median segment length: {overall_median} samples")
    
    # Step 5: Normalize segment lengths
    print("\n[5/6] Normalizing segment lengths...")
    if window_length is None:
        # Determine longest non-"stare" event so stares can be cropped accordingly
        priority_labels = {"left", "right", "up", "down", "blink"}
        longest_non_stare = 0
        fallback_longest = 0
        for label, label_stats in stats.items():
            if label.startswith("_"):
                continue
            label_name = str(label).lower()
            if label_name == "stare":
                continue
            lengths = label_stats.get("lengths", [])
            if lengths:
                max_length = max(lengths)
                if label_name in priority_labels:
                    longest_non_stare = max(longest_non_stare, max_length)
                fallback_longest = max(fallback_longest, max_length)

        if longest_non_stare == 0:
            longest_non_stare = fallback_longest

        if longest_non_stare > 0:
            window_length = longest_non_stare
            print(
                "  Using longest non-'stare' segment for window size: "
                f"{window_length} samples"
            )
        else:
            window_length = overall_median
            print(
                "  No non-'stare' segments found; falling back to median "
                f"window size: {window_length} samples"
            )
    else:
        print(f"  Using specified window size: {window_length} samples")
    
    # Ensure window_length is an integer (type checker doesn't know it can't be None here)
    final_window_length = int(window_length) if window_length is not None else overall_median
    
    # Extract signal channels (exclude metadata columns we added)
    # Use columns from aligned_df but exclude the ones we added during alignment
    added_cols = {"label", "signal_timestamp_ms", "label_timestamp_ms"}
    all_signal_cols = [col for col in aligned_df.columns if col not in added_cols]
    
    # Filter to only requested channels if specified
    if channels_to_use is not None:
        available_channels = set(all_signal_cols)
        # Convert requested channels to match available column types
        # Handle both string and int channel identifiers
        requested_channels = set()
        for ch in channels_to_use:
            # Try to match the channel
            if ch in available_channels:
                requested_channels.add(ch)
            elif isinstance(ch, str) and ch.isdigit():
                # Convert string number to int if columns are ints
                ch_int = int(ch)
                if ch_int in available_channels:
                    requested_channels.add(ch_int)
                else:
                    requested_channels.add(ch)  # Will fail in check below
            elif isinstance(ch, int):
                # Try as string if columns are strings
                if str(ch) in available_channels:
                    requested_channels.add(str(ch))
                else:
                    requested_channels.add(ch)  # Will fail in check below
            else:
                requested_channels.add(ch)  # Will fail in check below
        
        missing_channels = requested_channels - available_channels
        if missing_channels:
            raise ValueError(f"Requested channels not found: {missing_channels}. Available channels: {available_channels}")
        signal_cols = [col for col in all_signal_cols if col in requested_channels]
    else:
        signal_cols = all_signal_cols
    
    n_channels = len(signal_cols)
    
    if n_channels == 0:
        raise ValueError("No signal channels found in data. Check column extraction.")
    
    print(f"  Using {n_channels} signal channel(s): {signal_cols}")
    
    # Build feature matrix and label array
    X_list = []
    y_list = []
    
    for start_idx, end_idx, label in segments:
        segment_data = aligned_df.iloc[start_idx:end_idx][signal_cols].values
        normalized_segment = normalize_segment_length(
            segment_data, final_window_length, normalization_method
        )
        X_list.append(normalized_segment)
        y_list.append(label)
    
    X = np.array(X_list)  # Shape: (n_segments, window_length, n_channels)
    y = np.array(y_list)   # Shape: (n_segments,)
    
    print(f"  Final shape: X={X.shape}, y={y.shape}")
    
    # Step 6: Save output
    print("\n[6/6] Saving preprocessed data...")
    if output_path is None:
        output_path = Path(signal_path).parent / "preprocessed_cnn_data.npz"
    
    output_path = Path(output_path)
    # Save stats as JSON string since npz doesn't support nested dicts directly
    import json
    stats_json = json.dumps(stats)
    np.savez(
        output_path,
        X=X,
        y=y,
        window_length=np.array([final_window_length]),  # Save as array to avoid type issues
        n_channels=np.array([n_channels]),
        stats_json=np.array([stats_json], dtype=object),
    )
    
    # Also save as separate .npy files for convenience
    X_path = output_path.with_suffix(".X.npy")
    y_path = output_path.with_suffix(".y.npy")
    np.save(X_path, X)
    np.save(y_path, y)
    
    print(f"  Saved to: {output_path}")
    print(f"  Also saved: {X_path}, {y_path}")
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    
    return X, y, stats


def main():
    """Main entry point."""
    signal_path = Path("data/caden/processed/opensignals_84_BA_20_AE_BF_DA_2025-11-17T10-45-58.txt")
    label_path = Path("data/caden/processed/keypress_labels_edited_2025-11-17T10-45-58.txt")
    
    X, y, stats = preprocess_for_cnn(signal_path, label_path)
    
    print(f"\nReady for training!")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Classes: {np.unique(y)}")


if __name__ == "__main__":
    main()
