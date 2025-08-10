#!/usr/bin/env python
"""
BFRB Detection Pipeline Runner

This script serves as an entry point to run different components of the BFRB detection pipeline.
"""

import os
import argparse
import subprocess
import sys
from datetime import datetime


def setup_directories():
    """Create required directories if they don't exist"""
    dirs = ['data', 'models', 'submission', 'cache']
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def preprocess_data(args):
    """Run data preprocessing"""
    cmd = [
        'python', 'src/preprocessing.py',
        '--data-path', args.data_path,
        '--output-dir', args.output_dir,
        '--max-sequences', str(args.max_sequences) if args.max_sequences else 'None'
    ]
    
    print(f"Running preprocessing with {args.max_sequences if args.max_sequences else 'all'} sequences...")
    subprocess.run(cmd)


def train_model(args):
    """Train models"""
    cmd = [
        'python', 'src/train.py',
        '--data-path', args.data_path,
        '--features-file', args.features_file,
        '--output-dir', args.output_dir,
        '--batch-size', str(args.batch_size),
        '--epochs', str(args.epochs),
        '--folds', str(args.folds),
    ]
    
    if args.imu_only:
        cmd.append('--imu-only')
    
    if args.max_sequences:
        cmd.extend(['--max-sequences', str(args.max_sequences)])
    
    if args.group_cv:
        cmd.append('--group-cv')
    
    print(f"Training model with {'IMU-only' if args.imu_only else 'all sensors'} data...")
    print(f"Using {args.folds} folds for cross-validation")
    subprocess.run(cmd)


def run_inference(args):
    """Run inference and generate submission file"""
    # Use the provided output file or generate one with a timestamp
    if args.output_file:
        submission_file = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = f"submission/submission_{timestamp}.csv"
    
    cmd = [
        'python', 'src/inference.py',
        '--data-path', args.data_path,
        '--model-dir', args.model_dir,
        '--output-file', submission_file
    ]
    
    if args.api_example:
        cmd.append('--api-example')
    
    print(f"Running inference using models from {args.model_dir}...")
    subprocess.run(cmd)
    print(f"Submission file saved to {submission_file}")


def run_eda(args):
    """Launch Jupyter notebook for exploratory data analysis"""
    notebook_path = 'notebooks/01_EDA.ipynb'
    cmd = ['jupyter', 'notebook', notebook_path]
    
    print(f"Launching EDA notebook: {notebook_path}")
    subprocess.run(cmd)


def run_all(args):
    """Run the entire pipeline from preprocessing to submission"""
    print("Running the complete BFRB detection pipeline...")
    
    # 1. Preprocess data
    preprocess_args = argparse.Namespace(
        data_path=args.data_path,
        output_dir='data',
        max_sequences=args.max_sequences
    )
    preprocess_data(preprocess_args)
    
    # 2. Train models
    train_args = argparse.Namespace(
        data_path=args.data_path,
        features_file='data/processed_features.csv',
        output_dir='models',
        batch_size=32,
        epochs=30,
        folds=5,
        imu_only=False,
        max_sequences=args.max_sequences,
        group_cv=True
    )
    train_model(train_args)
    
    # 3. Run inference
    inference_args = argparse.Namespace(
        data_path=args.data_path,
        model_dir='models',
        api_example=False
    )
    run_inference(inference_args)


def main():
    """Parse arguments and run selected pipeline component"""
    parser = argparse.ArgumentParser(description="BFRB Detection Pipeline Runner")
    
    # Common arguments
    parser.add_argument('--data-path', type=str, 
                       default='/Users/shail/Downloads/cmi-detect-behavior-with-sensor-data.zip',
                       help='Path to the competition data zip file')
    parser.add_argument('--max-sequences', type=int, default=None,
                       help='Maximum number of sequences to process (for testing)')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Pipeline component to run')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess data')
    preprocess_parser.add_argument('--output-dir', type=str, default='data',
                                 help='Directory to save processed data')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--features-file', type=str, default='data/processed_features.csv',
                            help='Path to the processed features file')
    train_parser.add_argument('--output-dir', type=str, default='models',
                            help='Directory to save trained models')
    train_parser.add_argument('--batch-size', type=int, default=32,
                            help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=50,
                            help='Number of training epochs')
    train_parser.add_argument('--folds', type=int, default=5,
                            help='Number of cross-validation folds')
    train_parser.add_argument('--imu-only', action='store_true',
                            help='Use only IMU features')
    train_parser.add_argument('--group-cv', action='store_true',
                            help='Use GroupKFold for cross-validation')
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--model-dir', type=str, default='models',
                               help='Directory containing trained models')
    inference_parser.add_argument('--output-file', type=str, default=None,
                               help='Path to save the submission file')
    inference_parser.add_argument('--api-example', action='store_true',
                               help='Run the API example instead of full inference')
    
    # EDA command
    eda_parser = subparsers.add_parser('eda', help='Launch EDA notebook')
    
    # Run all command
    all_parser = subparsers.add_parser('all', help='Run the entire pipeline')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up directories
    setup_directories()
    
    # Run selected command
    if args.command == 'preprocess':
        preprocess_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'inference':
        run_inference(args)
    elif args.command == 'eda':
        run_eda(args)
    elif args.command == 'all':
        run_all(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
