# BFRB Detection Project Structure

## Project Organization
This file provides an overview of the project organization and file structure.

```
kaggle_bfrb/
├── README.md                    # Project overview and usage instructions
├── PROJECT_STRUCTURE.md         # This file - project organization
├── requirements.txt             # Required packages for installation
├── run.py                       # Command-line interface for all components
│
├── src/                         # Core implementation files
│   ├── preprocessing.py         # Memory-efficient data processing
│   ├── models.py                # PyTorch models implementation
│   ├── train.py                 # Training pipeline with competition metrics
│   └── inference.py             # Kaggle-compliant inference API
│
├── notebooks/                   # Jupyter notebooks for analysis
│   └── 01_EDA.ipynb             # Exploratory data analysis
│
├── data/                        # Processed data (created at runtime)
│   └── processed_features.csv   # Extracted features for model training
│
├── models/                      # Saved models (created at runtime)
│   ├── model_fold0.pt           # Trained model weights for fold 0
│   ├── model_fold1.pt           # Trained model weights for fold 1
│   └── ...                      # Additional model files
│
├── cache/                       # Cache for intermediary files
│   ├── processor.joblib         # Saved data processor
│   ├── comparison_chart.png     # Performance comparison visualization
│   └── ...                      # Additional cache files
│
└── submission/                  # Submission files for Kaggle
    ├── shail_shah_final_output_submission.csv  # Final submission CSV
    └── final_submission_output.txt             # Detailed submission report
```

## File Descriptions

### Core Scripts
- `run.py`: Command-line interface to run different parts of the pipeline
- `src/preprocessing.py`: Handles memory-efficient data loading and feature extraction
- `src/models.py`: Implements PyTorch models for BFRB detection
- `src/train.py`: Trains models using cross-validation and competition metrics
- `src/inference.py`: Generates Kaggle-compliant submissions

### Original Implementation Files
- `data_processor.py`: Original data processing implementation
- `model_trainer.py`: Original model training implementation  
- `main.py`: Original pipeline script
- `zip_explorer.py`: Script to explore competition data structure
- `explore_demographics.py`: Script to analyze demographic information

### Output Files
- `submission/shail_shah_final_output_submission.csv`: Final competition submission file
- `submission/final_submission_output.txt`: Detailed report on implementation and results

## Execution Flow

1. Data is processed using memory-efficient chunked reading
2. Features are extracted to reduce dimensionality
3. Models are trained with cross-validation
4. Inference is performed following Kaggle API requirements
5. Submission file is generated in the required format
