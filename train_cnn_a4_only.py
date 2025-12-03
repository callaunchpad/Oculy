"""Train 1D CNN on combined CSV data using only A4 channel (subject-agnostic)"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ============================================================================
# HARDCODED THRESHOLDS - Modify these values as needed
# ============================================================================
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence (probability) to accept a prediction
# Set to 0.0 to disable thresholding (accept all predictions)

# Signal amplitude thresholds for eye movement classification
# These values determine when a signal amplitude indicates a specific movement
THRESHOLD_RIGHT = 600.0   # If A4 > this value, classify as "right"
THRESHOLD_LEFT = 400.0    # If A4 < this value, classify as "left"
THRESHOLD_UP = 550.0      # If A4 > this value, classify as "up"
THRESHOLD_DOWN = 450.0    # If A4 < this value, classify as "down"
THRESHOLD_BLINK = 700.0   # If A4 > this value, classify as "blink"
# Values between thresholds are classified as "stare" or based on context
# ============================================================================


def create_segments(df: pd.DataFrame, label_column: str = "label") -> List[Tuple[int, int, str]]:
    """Create segments from consecutive samples with the same label."""
    labels = df[label_column].values
    segments = []
    
    if len(labels) == 0:
        return segments
    
    current_label = labels[0]
    start_idx = 0
    
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append((start_idx, i, current_label))
            current_label = labels[i]
            start_idx = i
    
    segments.append((start_idx, len(labels), current_label))
    return segments


def analyze_distribution(segments: List[Tuple[int, int, str]]) -> Dict[str, Dict]:
    """Analyze segment distribution by label class."""
    label_stats: Dict[str, Dict] = defaultdict(lambda: {"count": 0, "lengths": []})
    
    for start_idx, end_idx, label in segments:
        length = end_idx - start_idx
        label_stats[label]["count"] += 1
        label_stats[label]["lengths"].append(length)
    
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


def classify_by_threshold(signal_value: float) -> str:
    """Classify eye movement based on signal amplitude thresholds.
    
    Args:
        signal_value: The A4 channel signal amplitude value
        
    Returns:
        Classification label: 'right', 'left', 'up', 'down', 'blink', or 'stare'
    """
    if signal_value > THRESHOLD_BLINK:
        return "blink"
    elif signal_value > THRESHOLD_RIGHT:
        return "right"
    elif signal_value > THRESHOLD_UP:
        return "up"
    elif signal_value < THRESHOLD_LEFT:
        return "left"
    elif signal_value < THRESHOLD_DOWN:
        return "down"
    else:
        return "stare"


def normalize_segment_length(
    segment_data: np.ndarray,
    target_length: int,
    method: str = "pad_truncate",
) -> np.ndarray:
    """Normalize a segment to target length."""
    current_length = len(segment_data)
    
    if current_length == target_length:
        return segment_data
    
    if method == "pad_truncate":
        if current_length < target_length:
            padding = np.zeros(target_length - current_length)
            return np.concatenate([segment_data, padding])
        else:
            return segment_data[:target_length]
    
    elif method == "interpolate":
        x_old = np.linspace(0, 1, current_length)
        x_new = np.linspace(0, 1, target_length)
        return np.interp(x_new, x_old, segment_data)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def preprocess_csv_for_cnn(
    csv_path: Path,
    label_column: str = "label",
    signal_column: str = "A4",  # Only A4 channel
    window_length: int | None = None,
    normalization_method: str = "pad_truncate",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Preprocess CSV file for CNN training using only A4 channel."""
    print(f"\nProcessing {csv_path.name}...")
    
    df = pd.read_csv(csv_path)
    
    if label_column not in df.columns:
        raise ValueError(f"Column '{label_column}' not found in {csv_path}")
    if signal_column not in df.columns:
        raise ValueError(f"Column '{signal_column}' not found in {csv_path}")
    
    print(f"  Using signal column: {signal_column} (single channel)")
    
    # Create segments
    segments = create_segments(df, label_column)
    print(f"  Total segments: {len(segments)}")
    
    # Analyze distribution
    stats = analyze_distribution(segments)
    
    print("\n  Segment Distribution:")
    print("  " + "-" * 50)
    for label in sorted([k for k in stats.keys() if not k.startswith("_")]):
        label_stats = stats[label]
        print(f"  {label:10s}: {label_stats['count']:5d} segments, "
              f"median length: {label_stats['median_length']:5d} samples")
    
    overall_median = stats["_overall"]["median_length"]
    print(f"\n  Overall median segment length: {overall_median} samples")
    
    # Determine window length
    if window_length is None:
        priority_labels = {"left", "right", "up", "down", "blink"}
        longest_non_stare = 0
        for label, label_stats in stats.items():
            if label.startswith("_"):
                continue
            label_name = str(label).lower()
            if label_name == "stare":
                continue
            lengths = label_stats.get("lengths", [])
            if lengths and label_name in priority_labels:
                longest_non_stare = max(longest_non_stare, max(lengths))
        
        if longest_non_stare > 0:
            window_length = longest_non_stare
            print(f"  Using longest non-'stare' segment for window size: {window_length} samples")
        else:
            window_length = overall_median
            print(f"  Using median window size: {window_length} samples")
    else:
        print(f"  Using specified window size: {window_length} samples")
    
    # Build feature matrix and label array (single channel)
    X_list = []
    y_list = []
    
    for start_idx, end_idx, label in segments:
        segment_data = df.iloc[start_idx:end_idx][signal_column].values
        normalized_segment = normalize_segment_length(
            segment_data, window_length, normalization_method
        )
        X_list.append(normalized_segment)
        y_list.append(label)
    
    # Reshape to (n_segments, window_length, 1) for single channel
    X = np.array(X_list)  # Shape: (n_segments, window_length)
    X = X[:, :, np.newaxis]  # Shape: (n_segments, window_length, 1)
    y = np.array(y_list)   # Shape: (n_segments,)
    
    print(f"  Final shape: X={X.shape}, y={y.shape}")
    
    return X, y, stats


# Define 1D CNN model (single channel input, single convolution)
class CNN1D(nn.Module):
    def __init__(self, input_channels, num_classes, window_length):
        super(CNN1D, self).__init__()
        
        # Single convolutional block
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened size after convolution
        flattened_size = (window_length // 2) * 64
        
        # Fully connected layers
        self.fc1 = nn.Linear(flattened_size, 64)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Single conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class EOGDataset(Dataset):
    def __init__(self, X, y):
        # Convert to float32 and transpose to (batch, channels, sequence_length)
        self.X = torch.FloatTensor(X).transpose(1, 2)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    output_dir: Path,
) -> None:
    """Create and save training visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {plot_path}")
    plt.close()


def create_confusion_matrix_plot(
    all_labels: List[int],
    all_preds: List[int],
    label_encoder: LabelEncoder,
    output_dir: Path,
    dataset_name: str = "validation",
) -> None:
    """Create and save confusion matrix visualization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )
    plt.title(f'Confusion Matrix - {dataset_name.capitalize()} Set (A4 Channel Only)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = output_dir / f"confusion_matrix_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {plot_path}")
    plt.close()


def train_cnn(
    train_csv: Path,
    val_csv: Path,
    test_csv: Path | None = None,
    label_column: str = "label",
    signal_column: str = "A4",
    window_length: int | None = None,
    normalization_method: str = "pad_truncate",
    batch_size: int = 16,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    output_dir: Path | None = None,
) -> None:
    """Train 1D CNN on CSV files using only A4 channel."""
    
    print("=" * 60)
    print("Training 1D CNN on Combined Data (A4 Channel Only)")
    print("=" * 60)
    
    # Preprocess train data
    X_train, y_train, train_stats = preprocess_csv_for_cnn(
        train_csv, label_column, signal_column, window_length, normalization_method
    )
    
    # Preprocess validation data
    if window_length is None:
        window_length = X_train.shape[1]
    X_val, y_val, val_stats = preprocess_csv_for_cnn(
        val_csv, label_column, signal_column, window_length, normalization_method
    )
    
    # Preprocess test data if provided
    X_test, y_test, test_stats = None, None, None
    if test_csv and test_csv.exists():
        X_test, y_test, test_stats = preprocess_csv_for_cnn(
            test_csv, label_column, signal_column, window_length, normalization_method
        )
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    if y_test is not None:
        y_test_encoded = label_encoder.transform(y_test)
    
    print(f"\nLabel encoding:")
    print(f"  Classes: {label_encoder.classes_}")
    
    # Create datasets
    train_dataset = EOGDataset(X_train, y_train_encoded)
    val_dataset = EOGDataset(X_val, y_val_encoded)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_loader = None
    if X_test is not None:
        test_dataset = EOGDataset(X_test, y_test_encoded)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model (1 input channel for A4 only)
    n_channels = 1  # Single channel (A4)
    n_classes = len(label_encoder.classes_)
    
    model = CNN1D(input_channels=n_channels, num_classes=n_classes, window_length=window_length)
    model = model.to(device)
    
    print(f"\nModel created:")
    print(f"  Input channels: {n_channels} (A4 only)")
    print(f"  Window length: {window_length}")
    print(f"  Number of classes: {n_classes}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"\nTraining setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    if X_test is not None:
        print(f"  Test samples: {len(X_test)}")
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        train_loss /= len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    
    # Save model
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "cnn_model_a4.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder': label_encoder,
            'n_channels': n_channels,
            'window_length': window_length,
            'n_classes': n_classes,
        }, model_path)
        print(f"\nModel saved to: {model_path}")
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Validation Set Evaluation")
    print("=" * 60)
    if CONFIDENCE_THRESHOLD > 0.0:
        print(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            
            # Apply threshold if enabled
            if CONFIDENCE_THRESHOLD > 0.0:
                probs = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probs, 1)
                # Mark low-confidence predictions as -1 (rejected)
                predicted = torch.where(max_probs >= CONFIDENCE_THRESHOLD, 
                                       predicted, 
                                       torch.tensor(-1).to(device))
            else:
                _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.numpy())
    
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=label_encoder.classes_,
        labels=range(len(label_encoder.classes_)),
        zero_division=0
    ))
    
    if output_dir:
        create_confusion_matrix_plot(
            all_labels, all_preds, label_encoder, output_dir, "validation"
        )
    
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_acc = (np.array(all_preds)[class_mask] == i).sum() / class_mask.sum() * 100
            print(f"  {class_name:10s}: {class_acc:.2f}%")
    
    # Evaluate on test set if available
    if test_loader is not None:
        print("\n" + "=" * 60)
        print("Test Set Evaluation")
        print("=" * 60)
        if CONFIDENCE_THRESHOLD > 0.0:
            print(f"Using confidence threshold: {CONFIDENCE_THRESHOLD}")
        all_test_preds = []
        all_test_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                
                # Apply threshold if enabled
                if CONFIDENCE_THRESHOLD > 0.0:
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    max_probs, predicted = torch.max(probs, 1)
                    # Mark low-confidence predictions as -1 (rejected)
                    predicted = torch.where(max_probs >= CONFIDENCE_THRESHOLD, 
                                           predicted, 
                                           torch.tensor(-1).to(device))
                else:
                    _, predicted = torch.max(outputs, 1)
                
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(batch_y.numpy())
        
        print("\nClassification Report:")
        print(classification_report(
            all_test_labels, 
            all_test_preds, 
            target_names=label_encoder.classes_,
            labels=range(len(label_encoder.classes_)),
            zero_division=0
        ))
        
        if output_dir:
            create_confusion_matrix_plot(
                all_test_labels, all_test_preds, label_encoder, output_dir, "test"
            )
        
        print("\nPer-Class Accuracy (Test Set):")
        for i, class_name in enumerate(label_encoder.classes_):
            class_mask = np.array(all_test_labels) == i
            if class_mask.sum() > 0:
                class_acc = (np.array(all_test_preds)[class_mask] == i).sum() / class_mask.sum() * 100
                print(f"  {class_name:10s}: {class_acc:.2f}%")
    
    # Create visualizations
    if output_dir:
        create_training_plots(
            train_losses, val_losses, train_accuracies, val_accuracies,
            output_dir
        )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 1D CNN on combined CSV data using only A4 channel"
    )
    parser.add_argument(
        "--train-csv",
        required=True,
        help="Path to training CSV file"
    )
    parser.add_argument(
        "--val-csv",
        required=True,
        help="Path to validation CSV file"
    )
    parser.add_argument(
        "--test-csv",
        default=None,
        help="Path to test CSV file (optional)"
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Name of the label column (default: %(default)s)"
    )
    parser.add_argument(
        "--signal-column",
        default="A4",
        help="Signal column to use (default: %(default)s - A4 only)"
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Target window length. If not specified, uses median segment length."
    )
    parser.add_argument(
        "--normalization-method",
        choices=["pad_truncate", "interpolate"],
        default="pad_truncate",
        help="Method for normalizing segment lengths (default: %(default)s)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: %(default)s)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: %(default)s)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save the trained model (optional)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    train_csv = Path(args.train_csv)
    val_csv = Path(args.val_csv)
    test_csv = Path(args.test_csv) if args.test_csv else None
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")
    if test_csv and not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    
    train_cnn(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        label_column=args.label_column,
        signal_column=args.signal_column,
        window_length=args.window_length,
        normalization_method=args.normalization_method,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()


