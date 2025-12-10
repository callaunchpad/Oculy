import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
# import seaborn as sns

def plot_segment_analysis(file_path, output_path="segment_analysis_cleaned.png"):
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Ensure sorted
    df = df.sort_values(by=['session_id', 'timestamp_epoch_ms'])
    
    all_segments = []

    # Process each session separately
    for session_id, group in df.groupby('session_id'):
        segment_ids = (group['label'] != group['label'].shift()).cumsum()
        segments = group.groupby(segment_ids)
        
        for _, seg in segments:
            label = seg['label'].iloc[0]
            count = len(seg)
            # Duration in seconds (assuming 1000Hz approx)
            duration_sec = count / 1000.0
            
            all_segments.append({
                'session_id': session_id,
                'label': label,
                'duration_sec': duration_sec
            })

    segment_df = pd.DataFrame(all_segments)
    
    if segment_df.empty:
        print("No segments found.")
        return

    # Set up the plot style (basic matplotlib since seaborn might not be installed)
    # sns.set_theme(style="whitegrid")
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Colors
    colors = ['#66c2a5', '#fc8d62', '#8da0cb'] # Set2 like colors
    labels = sorted(segment_df['label'].unique())
    label_map = {l: i for i, l in enumerate(labels)}
    
    # 1. Box Plot
    ax1 = fig.add_subplot(gs[0, 0])
    data_to_plot = [segment_df[segment_df['label'] == l]['duration_sec'] for l in labels]
    ax1.boxplot(data_to_plot, labels=labels, patch_artist=True)
    ax1.set_title('Distribution of Segment Durations by Label')
    ax1.set_ylabel('Duration (seconds)')
    ax1.set_xlabel('Label')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Scatter/Strip Plot (alternative to violin without seaborn)
    ax2 = fig.add_subplot(gs[0, 1])
    for i, label in enumerate(labels):
        y = segment_df[segment_df['label'] == label]['duration_sec']
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax2.scatter(x, y, alpha=0.5, label=label)
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    ax2.set_title('Individual Segment Durations')
    ax2.set_ylabel('Duration (seconds)')
    ax2.set_xlabel('Label')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 3. Histogram of Counts
    ax3 = fig.add_subplot(gs[1, 0])
    counts = segment_df['label'].value_counts()
    # Align counts with labels order
    counts_ordered = [counts.get(l, 0) for l in labels]
    ax3.bar(labels, counts_ordered, color=colors[:len(labels)])
    ax3.set_title('Number of Segments per Label')
    ax3.set_ylabel('Count')
    ax3.set_xlabel('Label')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Total Duration Bar Plot
    ax4 = fig.add_subplot(gs[1, 1])
    total_durations = segment_df.groupby('label')['duration_sec'].sum()
    totals_ordered = [total_durations.get(l, 0) for l in labels]
    ax4.bar(labels, totals_ordered, color=colors[:len(labels)])
    ax4.set_title('Total Duration Recorded per Label')
    ax4.set_ylabel('Total Duration (seconds)')
    ax4.set_xlabel('Label')
    ax4.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/caden/v2/combined_caden_v2_labels.csv"
    plot_segment_analysis(path)

