# Memory-Efficient BFRB Detection using Sensor Data

This project implements a memory-optimized solution for the [Child Mind Institute - Detect BFRB behaviors](https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data) competition on Kaggle. The goal is to detect Body-Focused Repetitive Behaviors (BFRBs) using sensor data while minimizing memory usage. The project includes a full pipeline from data preprocessing to Kaggle-compliant submission.

## Key Features

- **Memory-efficient data processing**: Handles the large 1GB+ dataset through chunked reading
- **Comparative sensor analysis**: Evaluates performance with IMU-only vs all sensor types
- **Kaggle-compliant API**: Follows the required sequence-by-sequence prediction format
- **PyTorch-based models**: Includes models for both statistical features and raw time-series
- **Subject-based cross-validation**: Prevents data leakage between subjects

## Directory Structure

```
kaggle_bfrb/
├── README.md               # This file
├── requirements.txt        # Required packages
├── run.py                  # Command-line interface to all components
├── src/
│   ├── preprocessing.py    # Memory-efficient data processing 
│   ├── models.py           # PyTorch models (multi-branch & simplified)
│   ├── train.py            # Training with competition metrics
│   └── inference.py        # Kaggle-compliant submission API
├── notebooks/
│   └── 01_EDA.ipynb        # Exploratory data analysis
├── data/                   # Processed data (created at runtime)
├── models/                 # Saved models (created at runtime)
├── cache/                  # Intermediate files
└── submission/             # Generated submissions
```

## Exact Steps to Reproduce

Follow these steps exactly to reproduce the results and create a valid submission file:

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Download Competition Data

Download the competition data zip file from Kaggle:
- https://www.kaggle.com/competitions/cmi-detect-behavior-with-sensor-data/data

### 3. Update Data Path

Edit the `run.py` file to update the default data path:

```python
parser.add_argument('--data-path', type=str, 
                   default='/path/to/cmi-detect-behavior-with-sensor-data.zip',
                   help='Path to the competition data zip file')
```

### 4. Run Full Pipeline (Test Mode)

To test the entire pipeline with a small subset of data (faster):

```bash
python run.py all --max-sequences 300
```

This will:
- Create all necessary directories
- Preprocess the data (300 sequences)
- Train models with cross-validation
- Generate a submission file in `submission/submission_TIMESTAMP.csv`

### 5. Run Full Pipeline (Complete Mode)

To process all data and create the final submission:

```bash
python run.py all
```

This will take longer but will use all available data for better performance.

### 6. Generate Submission File Only

If you already have trained models and just want to generate a submission file:

```bash
python run.py inference
```

The submission will be saved to `submission/submission_TIMESTAMP.csv`.

## Kaggle Evaluation Metric

The system implements the exact competition metric:
1. Binary F1: Macro F1 score for target vs non-target classification
2. Gesture F1: Macro F1 score for gesture classification (only for target sequences)
3. Final score: Average of Binary F1 and Gesture F1

The submission file follows Kaggle's required format:
- `sequence_id`: Unique identifier for each sequence
- `gesture`: Predicted gesture or "non_target" for non-BFRB sequences

## Memory Usage Comparison

This implementation significantly reduces memory usage:

| Approach | Peak Memory Usage |
|----------|------------------|
| Loading entire dataset | 1GB+ |
| Chunked processing | ~250MB |
| Feature extraction | ~100MB |

## Performance Results

Our analysis shows modest but meaningful improvements when using all sensors compared to IMU-only:

| Metric | IMU-Only | All Sensors | Improvement |
|--------|----------|-------------|-------------|
| Binary F1 | 0.8375 | 0.8547 | +2.06% |
| Gesture F1 | 0.2069 | 0.2208 | +6.75% |
| Combined Score | 0.5222 | 0.5378 | +2.98% |

## Troubleshooting

- If you encounter memory errors, try reducing `--max-sequences`
- Ensure your system has Python 3.7+ and sufficient RAM (4GB minimum)
- Check that the data path is correctly set to your downloaded zip file
