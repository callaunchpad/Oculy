import pandas as pd
import numpy as np
import sys

def analyze_segments(file_path):
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Ensure sorted by session and timestamp (though it should be)
    df = df.sort_values(by=['session_id', 'timestamp_epoch_ms'])
    
    all_segments = []

    # Process each session separately to avoid bridging segments across sessions
    for session_id, group in df.groupby('session_id'):
        # Identify changes in label
        # (group['label'] != group['label'].shift()) is True where label changes
        # .cumsum() gives a unique ID for each contiguous segment
        segment_ids = (group['label'] != group['label'].shift()).cumsum()
        
        # Group by the segment ID
        segments = group.groupby(segment_ids)
        
        for _, seg in segments:
            label = seg['label'].iloc[0]
            count = len(seg)
            # Duration in seconds (assuming 1ms per sample or using timestamps)
            if 'timestamp_epoch_ms' in seg.columns and count > 1:
                duration = (seg['timestamp_epoch_ms'].max() - seg['timestamp_epoch_ms'].min()) / 1000.0
                # Add 1ms for the last sample interval if we assume point timestamps
                # closer approximation for 1000Hz: count / 1000.0
                # Let's use count / 1000.0 as it's cleaner for "duration covered"
                duration_from_count = count / 1000.0
            else:
                duration_from_count = count / 1000.0
            
            all_segments.append({
                'session_id': session_id,
                'label': label,
                'count': count,
                'duration_sec': duration_from_count
            })

    segment_df = pd.DataFrame(all_segments)
    
    if segment_df.empty:
        print("No segments found.")
        return

    print("\n--- Segment Analysis ---")
    print(f"Total segments identified: {len(segment_df)}")
    
    # Group by label and calculate stats
    stats = segment_df.groupby('label')['duration_sec'].describe()
    
    # Format the output
    print("\nDistribution of Segment Durations (seconds) by Label:")
    print(stats)

    print("\nDetailed Stats (Count, Mean, Min, Max):")
    for label, metrics in stats.iterrows():
        print(f"\nLabel: {label}")
        print(f"  Number of segments: {int(metrics['count'])}")
        print(f"  Mean duration:      {metrics['mean']:.3f} s")
        print(f"  Std deviation:      {metrics['std']:.3f} s")
        print(f"  Min duration:       {metrics['min']:.3f} s")
        print(f"  25%:                {metrics['25%']:.3f} s")
        print(f"  Median (50%):       {metrics['50%']:.3f} s")
        print(f"  75%:                {metrics['75%']:.3f} s")
        print(f"  Max duration:       {metrics['max']:.3f} s")
        
    # Also show total duration per label
    total_durations = segment_df.groupby('label')['duration_sec'].sum()
    print("\nTotal Duration per Label (seconds):")
    print(total_durations)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/caden/v2/combined_caden_v2_labels.csv"
    analyze_segments(path)

