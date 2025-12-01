# Oculy

**Machine Learning for Eye Tracking with EOG Sensors**

Blink and you’ll miss it—your eyes are in control.

Blending neuroscience, biosignals, and ML to make your eyes the ultimate interface!

Oculy is a project that processes raw electrooculography (EOG) signals and converts them into machine learning applications for eye tracking and movement analysis. The system captures EOG data using BITalino sensors and applies signal processing techniques to extract meaningful patterns for ML model training.

Our Goal: Build an interactive prototype where users can control digital interfaces (games, cursors, or menus) using only eye movements and blinks detected from biosignals. Examples: hands-free gaming, accessibility-focused communication systems, or artistic “eye-drawn” visualizations.

## Features

- Raw EOG signal acquisition and processing
- Signal filtering and noise reduction
- Eye movement pattern detection
- Blink detection algorithms
- Synthetic EOG data generation for testing
- Machine learning pipeline for eye tracking applications

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd oculy
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Install OpenSignals software for data acquisition:
   - Download from: https://support.pluxbiosignals.com/knowledge-base/introducing-opensignals-revolution/#download-opensignals

### Getting Started

1. Connect your BITalino hardware (see Hardware Setup below)
2. Use OpenSignals to collect EOG data
3. Run the signal processing tutorials in the `tutorials/` directory:
   - `signal_processing_updated.ipynb` - Main signal processing pipeline
   - `synthetic_eog_demo.ipynb` - Synthetic data generation and testing

## Hardware Setup

### Required Components

| Component | Quantity | Description |
|-----------|----------|-------------|
| BITalino (r)evolution Board Kit BLE/BT | 1x | Main acquisition board with Bluetooth connectivity |
| BITalino EOG UC-E6 sensors | 2x | Electrooculography sensor boards |

### Purchase Links

- [BITalino Board Kit](https://www.pluxbiosignals.com/products/bitalino-revolution-board-kit-ble-bt?variant=41371974533311)
- EOG sensors (bare boards) - Available from PLUX Biosignals

### Connection Setup

1. Connect EOG sensors to the BITalino board
2. Pair the device via Bluetooth
3. Configure OpenSignals for EOG data acquisition
4. Place electrodes according to EOG positioning guidelines

## Project Structure

```
oculy/
├── data/
│   └── sample_data/          # Sample EOG recordings
├── tutorials/
│   ├── signal_processing_updated.ipynb    # Main processing pipeline
│   └── synthetic_eog_demo.ipynb          # Synthetic data generation
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Usage

The project includes Jupyter notebooks that demonstrate:

- Loading and parsing OpenSignals data files (.txt and .h5 formats)
- Signal filtering and preprocessing
- Feature extraction from EOG signals

## Data Collection
- Red connects to right of temple, black connects to left. 
- Connect to EEG channel and open OpenSignals.
- Run labelling script and start recording
- Label full left look with "A" and return to origin as "D"

- Synthetic EOG signal generation for testing
- Visualization of eye movement patterns
