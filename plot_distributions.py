import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

def plot_distributions(file_path, output_path="segment_distributions_cleaned.png"):
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
            duration_sec = count / 1000.0
            
            all_segments.append({
                'label': label,
                'duration_sec': duration_sec
            })

    segment_df = pd.DataFrame(all_segments)
    
    if segment_df.empty:
        print("No segments found.")
        return

    labels = sorted(segment_df['label'].unique())
    colors = ['#66c2a5', '#fc8d62', '#8da0cb']  # Set2 colors
    
    # Create figure
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 4 * len(labels)), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    
    # Try to import scipy for KDE
    try:
        from scipy.stats import gaussian_kde
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("Scipy not found, skipping KDE lines.")

    for i, label in enumerate(labels):
        ax = axes[i]
        data = segment_df[segment_df['label'] == label]['duration_sec']
        
        # Plot Histogram (Probability Density)
        ax.hist(data, bins=30, density=True, alpha=0.6, color=colors[i % len(colors)], label=f'{label} hist')
        
        # Plot KDE
        if has_scipy and len(data) > 1:
            try:
                density = gaussian_kde(data)
                xs = np.linspace(0, data.max() * 1.1, 200)
                ax.plot(xs, density(xs), color=colors[i % len(colors)], linewidth=2, label=f'{label} KDE')
            except Exception as e:
                print(f"Could not plot KDE for {label}: {e}")

        ax.set_title(f'Probability Distribution of Duration: {label.upper()}')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add stats annotation
        stats_text = (
            f"Mean: {data.mean():.2f}s\n"
            f"Std: {data.std():.2f}s\n"
            f"N: {len(data)}"
        )
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1].set_xlabel('Duration (seconds)')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Distribution plot saved to {output_path}")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/caden/v2/combined_caden_v2_labels.csv"
    plot_distributions(path)

