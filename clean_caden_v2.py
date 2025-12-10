import pandas as pd
import numpy as np
from pathlib import Path

def clean_data():
    data_path = Path("data/caden/v2/combined_caden_v2_data.csv")
    labels_path = Path("data/caden/v2/combined_caden_v2_labels.csv")
    
    print("Loading data...")
    data_df = pd.read_csv(data_path)
    labels_df = pd.read_csv(labels_path)
    
    if len(data_df) != len(labels_df):
        raise ValueError(f"Length mismatch: Data {len(data_df)} vs Labels {len(labels_df)}")
    
    # We need to process segments based on labels
    # Combine temporarily to identify segments
    combined = labels_df.copy()
    
    # Identify segments
    # Sort by session and timestamp (should be already sorted)
    # But let's be safe. We need to respect the original order if they are row-aligned.
    # Assuming they are already sorted by time.
    
    # Create a unique segment ID
    # A segment changes when label changes OR session changes
    combined['label_change'] = (combined['label'] != combined['label'].shift())
    combined['session_change'] = (combined['session_id'] != combined['session_id'].shift())
    combined['segment_id'] = (combined['label_change'] | combined['session_change']).cumsum()
    
    print("Identifying segments...")
    segment_stats = []
    
    # Group by segment_id to get duration
    # Since sampling rate is ~1000Hz, count is roughly duration in ms.
    # We can also use timestamp diff if available.
    
    for seg_id, group in combined.groupby('segment_id'):
        count = len(group)
        duration_sec = count / 1000.0 # Approximation
        
        # Check if we can get more precise duration from timestamps
        # Data and Labels both have timestamp_epoch_ms
        if 'timestamp_epoch_ms' in group.columns:
            ts = group['timestamp_epoch_ms']
            if len(ts) > 1:
                precise_duration = (ts.iloc[-1] - ts.iloc[0]) / 1000.0
                # Add one sample interval roughly
                precise_duration += 0.001
                duration_sec = precise_duration
        
        segment_stats.append({
            'segment_id': seg_id,
            'label': group['label'].iloc[0],
            'count': count,
            'duration_sec': duration_sec,
            'indices': group.index.tolist()
        })
    
    seg_df = pd.DataFrame(segment_stats)
    print(f"Found {len(seg_df)} segments.")
    
    # 1. Filter: Remove segments < 0.5s
    short_segments = seg_df[seg_df['duration_sec'] < 0.5]
    print(f"Segments < 0.5s: {len(short_segments)} (Total duration: {short_segments['duration_sec'].sum():.2f}s)")
    
    # 2. Filter: Remove outliers (per label)
    # We'll use IQR method
    outlier_segment_ids = set()
    
    print("\nOutlier detection (1.5 * IQR):")
    for label in seg_df['label'].unique():
        label_segs = seg_df[seg_df['label'] == label]
        if len(label_segs) < 2:
            continue
            
        Q1 = label_segs['duration_sec'].quantile(0.25)
        Q3 = label_segs['duration_sec'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = label_segs[(label_segs['duration_sec'] < lower_bound) | (label_segs['duration_sec'] > upper_bound)]
        outlier_segment_ids.update(outliers['segment_id'].tolist())
        
        print(f"  Label '{label}': Q1={Q1:.3f}s, Q3={Q3:.3f}s, IQR={IQR:.3f}s. Range: [{lower_bound:.3f}, {upper_bound:.3f}]")
        print(f"    Found {len(outliers)} outliers.")

    # Combine filters
    # We want to KEEP segments that are >= 0.5s AND NOT outliers
    # Note: A segment could be < 0.5s AND an outlier (if IQR range is high), or valid duration but outlier.
    
    # Let's list IDs to remove
    ids_to_remove = set(short_segments['segment_id'].tolist()) | outlier_segment_ids
    
    print(f"\nTotal segments to remove: {len(ids_to_remove)}")
    
    valid_segments = seg_df[~seg_df['segment_id'].isin(ids_to_remove)]
    print(f"Retaining {len(valid_segments)} segments.")
    
    # Get all valid indices
    valid_indices = []
    for idx_list in valid_segments['indices']:
        valid_indices.extend(idx_list)
    
    valid_indices = sorted(valid_indices)
    
    print(f"\nFiltering rows...")
    print(f"Original rows: {len(data_df)}")
    print(f"Rows after cleaning: {len(valid_indices)}")
    
    clean_data_df = data_df.iloc[valid_indices].reset_index(drop=True)
    clean_labels_df = labels_df.iloc[valid_indices].reset_index(drop=True)
    
    # Save
    data_out = Path("data/caden/v2/cleaned_caden_v2_data.csv")
    labels_out = Path("data/caden/v2/cleaned_caden_v2_labels.csv")
    
    clean_data_df.to_csv(data_out, index=False)
    clean_labels_df.to_csv(labels_out, index=False)
    
    print(f"\nSaved cleaned data to:\n  {data_out}\n  {labels_out}")

if __name__ == "__main__":
    clean_data()

