"""Split cleaned caden_v2 data into 80/20 train/val using stratified segment-based approach."""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def identify_segments(labels_df):
    """
    Identify contiguous segments in the data.
    Returns a list of segment info dictionaries.
    """
    labels_df = labels_df.reset_index(drop=True)
    
    all_segments = []
    
    for session_id, group in labels_df.groupby('session_id', sort=False):
        group = group.sort_values('timestamp_epoch_ms').reset_index(drop=True)
        group_start_idx = group.index[0]
        
        # Get original indices
        original_indices = labels_df[labels_df['session_id'] == session_id].index.tolist()
        
        # Identify changes in label
        segment_ids_in_group = (group['label'] != group['label'].shift()).cumsum()
        
        for seg_id_in_session, seg_group in group.groupby(segment_ids_in_group):
            label = seg_group['label'].iloc[0]
            count = len(seg_group)
            
            # Get the actual row indices in the original dataframe
            seg_local_indices = seg_group.index.tolist()
            seg_original_indices = [original_indices[i] for i in seg_local_indices]
            
            all_segments.append({
                'session_id': session_id,
                'label': label,
                'count': count,
                'indices': seg_original_indices
            })
    
    return all_segments


def stratified_segment_split(segments, test_size=0.2, random_state=42):
    """
    Split segments into train/val sets, stratified by label.
    """
    # Group segments by label
    segments_by_label = {}
    for seg in segments:
        label = seg['label']
        if label not in segments_by_label:
            segments_by_label[label] = []
        segments_by_label[label].append(seg)
    
    train_segments = []
    val_segments = []
    
    print("\nStratified split by label:")
    for label, label_segments in segments_by_label.items():
        n_segments = len(label_segments)
        
        if n_segments < 2:
            # If only 1 segment, put it in train
            train_segments.extend(label_segments)
            print(f"  {label}: {n_segments} segments -> all to train (too few to split)")
            continue
        
        # Split this label's segments
        train_segs, val_segs = train_test_split(
            label_segments, 
            test_size=test_size, 
            random_state=random_state
        )
        
        train_segments.extend(train_segs)
        val_segments.extend(val_segs)
        
        print(f"  {label}: {n_segments} segments -> {len(train_segs)} train, {len(val_segs)} val")
    
    return train_segments, val_segments


def main():
    data_path = Path("data/caden/v2/cleaned_caden_v2_data.csv")
    labels_path = Path("data/caden/v2/cleaned_caden_v2_labels.csv")
    output_dir = Path("splits/caden_v2_cleaned")
    
    print("Loading cleaned data...")
    data_df = pd.read_csv(data_path)
    labels_df = pd.read_csv(labels_path)
    
    print(f"Data rows: {len(data_df)}")
    print(f"Labels rows: {len(labels_df)}")
    
    assert len(data_df) == len(labels_df), "Data and labels must have same number of rows"
    
    # Identify segments
    print("\nIdentifying segments...")
    segments = identify_segments(labels_df)
    print(f"Total segments: {len(segments)}")
    
    # Count by label
    label_counts = {}
    for seg in segments:
        label = seg['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    print("Segments by label:", label_counts)
    
    # Stratified split
    train_segments, val_segments = stratified_segment_split(segments, test_size=0.2, random_state=42)
    
    # Gather indices
    train_indices = []
    for seg in train_segments:
        train_indices.extend(seg['indices'])
    train_indices = sorted(train_indices)
    
    val_indices = []
    for seg in val_segments:
        val_indices.extend(seg['indices'])
    val_indices = sorted(val_indices)
    
    print(f"\nTrain samples: {len(train_indices)}")
    print(f"Val samples: {len(val_indices)}")
    
    # Create combined dataframes (data + label column)
    train_data = data_df.iloc[train_indices].reset_index(drop=True)
    train_labels = labels_df.iloc[train_indices].reset_index(drop=True)
    train_combined = train_data.copy()
    train_combined['label'] = train_labels['label'].values
    
    val_data = data_df.iloc[val_indices].reset_index(drop=True)
    val_labels = labels_df.iloc[val_indices].reset_index(drop=True)
    val_combined = val_data.copy()
    val_combined['label'] = val_labels['label'].values
    
    # Verify label distribution
    print("\nLabel distribution:")
    print("  Train:", train_combined['label'].value_counts().to_dict())
    print("  Val:", val_combined['label'].value_counts().to_dict())
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    
    train_combined.to_csv(train_path, index=False)
    val_combined.to_csv(val_path, index=False)
    
    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_combined)} rows)")
    print(f"  Val: {val_path} ({len(val_combined)} rows)")


if __name__ == "__main__":
    main()
