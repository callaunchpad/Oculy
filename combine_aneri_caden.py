"""Combine aneri data (filtered, no blink) with cleaned caden_v2 data."""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prepare_aneri(aneri_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load aneri data and prepare it to match caden format.
    Returns (data_df, labels_df) with matching columns to caden format.
    """
    print(f"Loading aneri data from {aneri_path}...")
    df = pd.read_csv(aneri_path)
    
    print(f"  Original rows: {len(df)}")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")
    
    # Filter out blink
    df = df[df['label'] != 'blink'].reset_index(drop=True)
    print(f"  After removing blink: {len(df)}")
    print(f"  Remaining labels: {df['label'].value_counts().to_dict()}")
    
    # Map aneri columns to caden format
    # Aneri: num, nSeq, I1, I2, O1, O2, A1, A2, A3, A4, A5, A6, timestamp_epoch_ms, label, ...
    # Caden data: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, timestamp_epoch_ms, session_id
    # Caden labels: label, keypress_sample_number, keypress_timestamp_ms, session_id, timestamp_epoch_ms
    
    # Create data_df matching caden format
    data_df = pd.DataFrame({
        '0': df['num'],
        '1': df['nSeq'],
        '2': df['I1'],
        '3': df['I2'],
        '4': df['O1'],
        '5': df['A1'],  # Maps to column 5
        '6': df['A2'],  # Maps to column 6
        '7': df['A3'],  # Maps to column 7
        '8': df['A4'],  # Maps to column 8 (this is what we use for training)
        '9': df['A5'],  # Maps to column 9
        '10': df['A6'], # Maps to column 10
        'timestamp_epoch_ms': df['timestamp_epoch_ms'],
        'session_id': 'aneri'  # Add session identifier
    })
    
    # Create labels_df matching caden format
    labels_df = pd.DataFrame({
        'label': df['label'],
        'keypress_sample_number': df['keypress_sample_number'],
        'keypress_timestamp_ms': df['keypress_timestamp_ms'],
        'session_id': 'aneri',
        'timestamp_epoch_ms': df['timestamp_epoch_ms']
    })
    
    return data_df, labels_df


def load_caden_cleaned() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned caden data."""
    data_path = Path("data/caden/v2/cleaned_caden_v2_data.csv")
    labels_path = Path("data/caden/v2/cleaned_caden_v2_labels.csv")
    
    print(f"\nLoading cleaned caden data...")
    data_df = pd.read_csv(data_path)
    labels_df = pd.read_csv(labels_path)
    
    print(f"  Data rows: {len(data_df)}")
    print(f"  Labels: {labels_df['label'].value_counts().to_dict()}")
    
    return data_df, labels_df


def main():
    aneri_path = Path("data/caden/v2/aneri_merged_run1and2.csv")
    output_dir = Path("data/caden/v2")
    
    # Load and prepare aneri data
    aneri_data, aneri_labels = load_and_prepare_aneri(aneri_path)
    
    # Load caden data
    caden_data, caden_labels = load_caden_cleaned()
    
    # Combine
    print("\nCombining datasets...")
    combined_data = pd.concat([caden_data, aneri_data], ignore_index=True)
    combined_labels = pd.concat([caden_labels, aneri_labels], ignore_index=True)
    
    print(f"  Combined data rows: {len(combined_data)}")
    print(f"  Combined labels: {combined_labels['label'].value_counts().to_dict()}")
    print(f"  Sessions: {combined_data['session_id'].unique().tolist()}")
    
    # Save combined data
    combined_data_path = output_dir / "combined_aneri_caden_data.csv"
    combined_labels_path = output_dir / "combined_aneri_caden_labels.csv"
    
    combined_data.to_csv(combined_data_path, index=False)
    combined_labels.to_csv(combined_labels_path, index=False)
    
    print(f"\nSaved:")
    print(f"  Data: {combined_data_path}")
    print(f"  Labels: {combined_labels_path}")


if __name__ == "__main__":
    main()
