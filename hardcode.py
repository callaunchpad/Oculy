import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from parse_opensignals import parse_opensignals_txt, merge_opensignals_and_keypress

def read_file(input_path, keypress_path=None):
    """
    Reads the OpenSignals text file and returns a pandas DataFrame.
    If keypress_path is provided, merges the data with keypress labels.
    """
    print(f"Reading file: {input_path}")
    try:
        if keypress_path:
            print(f"Merging with keypress labels: {keypress_path}")
            # merge_opensignals_and_keypress returns (metadata, keypress_metadata, combined_df)
            _, _, df = merge_opensignals_and_keypress(input_path, keypress_path)
        else:
            _, df = parse_opensignals_txt(input_path)
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def apply_classification(df):
    """
    Applies hardcoded thresholds to the A4 column to classify samples.
    
    Thresholds:
    - A4 >= 1000: "left"
    - A4 >= 750: "blink" (and < 1000)
    - A4 < 5: "right"
    - Otherwise: "neutral"
    """
    if df is None:
        return None
        
    if "A4" not in df.columns:
        print("Error: Column 'A4' not found in the data.")
        print(f"Available columns: {df.columns.tolist()}")
        return None

    # Vectorized classification
    conditions = [
        (df["A4"] >= 1000),
        (df["A4"] >= 750),
        (df["A4"] < 5)
    ]
    choices = ["left", "blink", "right"]
    
    # np.select evaluates conditions in order. First true wins.
    df["classification"] = np.select(conditions, choices, default="neutral")
    
    return df

def evaluate_performance(df):
    """
    Compares the hardcoded classification against the ground truth labels.
    Calculates precision for non-neutral predictions.
    """
    if "label" not in df.columns:
        print("No ground truth labels found (column 'label' missing). Skipping evaluation.")
        return

    print("\n=== Performance Evaluation ===")
    
    # Normalize labels for comparison (lowercase)
    # Keypress labels: 'stare', 'left', 'right', 'blink', 'up', 'down'
    # Classification: 'neutral', 'left', 'right', 'blink'
    
    # Map 'neutral' classification to 'stare' for full comparison, 
    # or just focus on when we predict an event.
    
    # The user asked: "when we label something as 'left' or 'right' ... we check if we're right"
    # This implies we care about the precision of our event predictions.
    
    predicted_events = df[df["classification"] != "neutral"]
    
    if predicted_events.empty:
        print("No events predicted (all classified as neutral).")
        return

    total_predictions = len(predicted_events)
    correct_predictions = 0
    
    # Detailed breakdown by class
    for cls in ["left", "right", "blink"]:
        subset = predicted_events[predicted_events["classification"] == cls]
        count = len(subset)
        if count == 0:
            continue
            
        # Check matches. Assuming ground truth label matches the string exactly (case-insensitive just in case)
        matches = subset[subset["label"].astype(str).str.lower() == cls]
        hits = len(matches)
        precision = hits / count * 100 if count > 0 else 0
        
        print(f"Class '{cls}': Predicted {count} times. Correct: {hits}. Precision: {precision:.2f}%")
        correct_predictions += hits

    overall_precision = correct_predictions / total_predictions * 100
    print(f"\nOverall Precision (weighted): {overall_precision:.2f}% ({correct_predictions}/{total_predictions})")
    
    # Check for false positives - what were they actually?
    print("\nFalse Positive Analysis (Top 5 actual labels for incorrect predictions):")
    incorrect = predicted_events[predicted_events["classification"] != predicted_events["label"].astype(str).str.lower()]
    if not incorrect.empty:
        print(incorrect.groupby(["classification", "label"]).size().sort_values(ascending=False).head(10))
    else:
        print("No incorrect predictions found!")


def process_pipeline(input_file, keypress_file=None):
    # 1. Read Data
    df = read_file(input_file, keypress_file)
    if df is None:
        return

    # 2. Classify Data
    df_classified = apply_classification(df)
    if df_classified is None:
        print("Classification failed.")
        return

    # 3. Output/Analysis
    print("\nClassification Summary:")
    print(df_classified["classification"].value_counts())

    # 4. Evaluate if labels exist
    if keypress_file:
        evaluate_performance(df_classified)

    # Save output
    output_path = Path(input_file).with_suffix('.classified.csv')
    df_classified.to_csv(output_path, index=False)
    print(f"\nSaved classified data to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify OpenSignals data based on hardcoded thresholds.")
    parser.add_argument("input_file", help="Path to the raw OpenSignals text file")
    parser.add_argument("--keypress_file", help="Path to the keypress labels file for validation", default=None)
    args = parser.parse_args()

    process_pipeline(args.input_file, args.keypress_file)
