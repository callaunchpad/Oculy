"""Remove outlier segments based on duration for left and right labels."""

import pandas as pd
import numpy as np


def identify_segments(df):
    """
    Identify contiguous segments in the data.
    Returns a DataFrame with segment info and the segment IDs for each row.
    """
    df = df.sort_values(by=['session_id', 'timestamp_epoch_ms']).reset_index(drop=True)
    
    all_segments = []
    segment_id_per_row = np.zeros(len(df), dtype=int)
    current_segment_id = 0
    
    for session_id, group in df.groupby('session_id', sort=False):
        group_indices = group.index.tolist()
        
        # Identify changes in label
        segment_ids_in_group = (group['label'] != group['label'].shift()).cumsum()
        
        for seg_id_in_session, seg_group in group.groupby(segment_ids_in_group):
            label = seg_group['label'].iloc[0]
            count = len(seg_group)
            duration_sec = count / 1000.0  # assuming 1000Hz sampling rate
            
            seg_indices = seg_group.index.tolist()
            start_idx = seg_indices[0]
            end_idx = seg_indices[-1]
            
            all_segments.append({
                'segment_id': current_segment_id,
                'session_id': session_id,
                'label': label,
                'count': count,
                'duration_sec': duration_sec,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
            
            # Mark segment ID for each row
            segment_id_per_row[seg_indices] = current_segment_id
            current_segment_id += 1
    
    segment_df = pd.DataFrame(all_segments)
    return segment_df, segment_id_per_row


def find_outlier_segments(segment_df, label, lower_multiplier=1.5, upper_multiplier=1.5):
    """
    Find outlier segments for a specific label using IQR method.
    Returns list of segment IDs that are outliers.
    """
    label_segments = segment_df[segment_df['label'] == label]
    
    if len(label_segments) == 0:
        return []
    
    durations = label_segments['duration_sec']
    Q1 = durations.quantile(0.25)
    Q3 = durations.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - lower_multiplier * IQR
    upper_bound = Q3 + upper_multiplier * IQR
    
    print(f"\n{label.upper()} segments:")
    print(f"  Q1: {Q1:.3f}s, Q3: {Q3:.3f}s, IQR: {IQR:.3f}s")
    print(f"  Lower bound: {lower_bound:.3f}s")
    print(f"  Upper bound: {upper_bound:.3f}s")
    
    outliers = label_segments[
        (label_segments['duration_sec'] < lower_bound) | 
        (label_segments['duration_sec'] > upper_bound)
    ]
    
    print(f"  Total {label} segments: {len(label_segments)}")
    print(f"  Outliers found: {len(outliers)}")
    
    if len(outliers) > 0:
        print(f"  Outlier durations: {sorted(outliers['duration_sec'].tolist())}")
    
    return outliers['segment_id'].tolist()


def main():
    data_path = "data/caden/v2/cleaned_caden_v2_data.csv"
    labels_path = "data/caden/v2/cleaned_caden_v2_labels.csv"
    
    print("Loading data...")
    data_df = pd.read_csv(data_path)
    labels_df = pd.read_csv(labels_path)
    
    print(f"Original data rows: {len(data_df)}")
    print(f"Original labels rows: {len(labels_df)}")
    
    # Verify alignment
    assert len(data_df) == len(labels_df), "Data and labels must have same number of rows"
    
    # Identify segments using labels
    print("\nIdentifying segments...")
    segment_df, segment_id_per_row = identify_segments(labels_df)
    
    print(f"Total segments identified: {len(segment_df)}")
    
    # Find outliers for left and right labels
    outlier_segments = []
    for label in ['left', 'right']:
        outliers = find_outlier_segments(segment_df, label)
        outlier_segments.extend(outliers)
    
    print(f"\nTotal outlier segments to remove: {len(outlier_segments)}")
    
    if len(outlier_segments) == 0:
        print("No outliers found. No changes made.")
        return
    
    # Create mask for rows to keep (not in outlier segments)
    rows_to_keep = ~np.isin(segment_id_per_row, outlier_segments)
    
    # Count rows to remove
    rows_to_remove = np.sum(~rows_to_keep)
    print(f"Rows to remove: {rows_to_remove}")
    
    # Apply mask
    cleaned_data_df = data_df[rows_to_keep].reset_index(drop=True)
    cleaned_labels_df = labels_df[rows_to_keep].reset_index(drop=True)
    
    print(f"\nCleaned data rows: {len(cleaned_data_df)}")
    print(f"Cleaned labels rows: {len(cleaned_labels_df)}")
    
    # Save cleaned data
    cleaned_data_path = "data/caden/v2/cleaned_caden_v2_data.csv"
    cleaned_labels_path = "data/caden/v2/cleaned_caden_v2_labels.csv"
    
    cleaned_data_df.to_csv(cleaned_data_path, index=False)
    cleaned_labels_df.to_csv(cleaned_labels_path, index=False)
    
    print(f"\nSaved cleaned data to: {cleaned_data_path}")
    print(f"Saved cleaned labels to: {cleaned_labels_path}")
    
    # Show summary of remaining segments
    print("\n--- Summary of remaining data ---")
    remaining_segment_df, _ = identify_segments(cleaned_labels_df)
    for label in ['left', 'right', 'stare']:
        label_segs = remaining_segment_df[remaining_segment_df['label'] == label]
        if len(label_segs) > 0:
            print(f"\n{label.upper()}:")
            print(f"  Segments: {len(label_segs)}")
            print(f"  Duration stats: min={label_segs['duration_sec'].min():.3f}s, "
                  f"max={label_segs['duration_sec'].max():.3f}s, "
                  f"mean={label_segs['duration_sec'].mean():.3f}s")


if __name__ == "__main__":
    main()
