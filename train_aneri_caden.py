"""Train 1D CNN on combined aneri+caden data with 1000-sample windows and segment mean padding."""

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
# CONFIGURATION
# ============================================================================
WINDOW_LENGTH = 1000  # 1 second at 1000Hz
SIGNAL_COLUMN = 8     # Column index for A4 channel
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001


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


def normalize_segment_length(
    segment_data: np.ndarray,
    target_length: int,
) -> np.ndarray:
    """
    Normalize a segment to target length using segment mean for padding.
    - If shorter: pad with segment mean
    - If longer: truncate
    """
    current_length = len(segment_data)
    
    if current_length == target_length:
        return segment_data
    
    if current_length < target_length:
        # Pad with segment mean (not zeros!)
        pad_value = np.mean(segment_data)
        padding = np.full(target_length - current_length, pad_value)
        return np.concatenate([segment_data, padding])
    else:
        # Truncate
        return segment_data[:target_length]


def preprocess_csv_for_cnn(
    csv_path: Path,
    label_column: str = "label",
    signal_column: int = SIGNAL_COLUMN,
    window_length: int = WINDOW_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess CSV file for CNN training using single channel."""
    print(f"\nProcessing {csv_path.name}...")
    
    df = pd.read_csv(csv_path)
    
    # Get signal column name (it's stored as integer column name)
    signal_col_name = str(signal_column)
    if signal_col_name not in df.columns:
        # Try as integer
        cols = df.columns.tolist()
        if signal_column < len(cols):
            signal_col_name = cols[signal_column]
        else:
            raise ValueError(f"Column {signal_column} not found. Available: {cols}")
    
    print(f"  Using signal column: {signal_col_name}")
    print(f"  Window length: {window_length}")
    
    # Create segments
    segments = create_segments(df, label_column)
    print(f"  Total segments: {len(segments)}")
    
    # Count by label
    label_counts = {}
    for _, _, label in segments:
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"  Segments by label: {label_counts}")
    
    # Build feature matrix and label array
    X_list = []
    y_list = []
    
    for start_idx, end_idx, label in segments:
        segment_data = df.iloc[start_idx:end_idx][signal_col_name].values.astype(float)
        normalized_segment = normalize_segment_length(segment_data, window_length)
        X_list.append(normalized_segment)
        y_list.append(label)
    
    # Reshape to (n_segments, window_length, 1) for single channel
    X = np.array(X_list)  # Shape: (n_segments, window_length)
    X = X[:, :, np.newaxis]  # Shape: (n_segments, window_length, 1)
    y = np.array(y_list)   # Shape: (n_segments,)
    
    print(f"  Final shape: X={X.shape}, y={y.shape}")
    
    return X, y


class CNN1D(nn.Module):
    """1D CNN for EOG classification."""
    def __init__(self, input_channels: int, num_classes: int, window_length: int):
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
    plt.title(f'Confusion Matrix - {dataset_name.capitalize()} Set (Aneri + Caden)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plot_path = output_dir / f"confusion_matrix_{dataset_name}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {plot_path}")
    plt.close()


def train():
    """Main training function."""
    train_csv = Path("splits/aneri_caden_combined/train.csv")
    val_csv = Path("splits/aneri_caden_combined/val.csv")
    output_dir = Path("models/aneri_caden_combined")
    
    print("=" * 60)
    print("Training 1D CNN on Combined Aneri + Caden Data")
    print("=" * 60)
    print(f"Window length: {WINDOW_LENGTH}")
    print(f"Signal column: {SIGNAL_COLUMN}")
    
    # Preprocess data
    X_train, y_train = preprocess_csv_for_cnn(train_csv)
    X_val, y_val = preprocess_csv_for_cnn(val_csv)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    
    print(f"\nLabel encoding:")
    print(f"  Classes: {label_encoder.classes_}")
    
    # Create datasets
    train_dataset = EOGDataset(X_train, y_train_encoded)
    val_dataset = EOGDataset(X_val, y_val_encoded)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    n_channels = 1
    n_classes = len(label_encoder.classes_)
    
    model = CNN1D(input_channels=n_channels, num_classes=n_classes, window_length=WINDOW_LENGTH)
    model = model.to(device)
    
    print(f"\nModel created:")
    print(f"  Input channels: {n_channels}")
    print(f"  Window length: {WINDOW_LENGTH}")
    print(f"  Number of classes: {n_classes}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    print(f"\nTraining setup:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Train segments: {len(X_train)}")
    print(f"  Validation segments: {len(X_val)}")
    
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
    
    for epoch in range(NUM_EPOCHS):
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
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
            print()
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "cnn_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'n_channels': n_channels,
        'window_length': WINDOW_LENGTH,
        'n_classes': n_classes,
        'signal_column': SIGNAL_COLUMN,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Validation Set Evaluation")
    print("=" * 60)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
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
    
    # Create visualizations
    create_confusion_matrix_plot(all_labels, all_preds, label_encoder, output_dir, "validation")
    create_training_plots(train_losses, val_losses, train_accuracies, val_accuracies, output_dir)
    
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = np.array(all_labels) == i
        if class_mask.sum() > 0:
            class_acc = (np.array(all_preds)[class_mask] == i).sum() / class_mask.sum() * 100
            print(f"  {class_name:10s}: {class_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    train()
