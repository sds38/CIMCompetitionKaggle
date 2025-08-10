import zipfile
import pandas as pd
import numpy as np

def explore_demographics(zip_path):
    """Explore the demographics data from the competition zip file"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Load demographics data
        with zip_ref.open('train_demographics.csv') as f:
            train_demo = pd.read_csv(f)
            
        print("\n=== Training Demographics Data ===")
        print(f"Shape: {train_demo.shape}")
        print("\nFirst 5 rows:")
        print(train_demo.head())
        print("\nInformation:")
        print(train_demo.dtypes)
        print("\nSummary statistics:")
        print(train_demo.describe())
        
        # Load test demographics if available
        try:
            with zip_ref.open('test_demographics.csv') as f:
                test_demo = pd.read_csv(f)
                
            print("\n=== Test Demographics Data ===")
            print(f"Shape: {test_demo.shape}")
            print("\nFirst 5 rows:")
            print(test_demo.head())
        except Exception as e:
            print(f"\nError loading test_demographics.csv: {e}")
        
        # Also check a sample of training data to see label distribution
        try:
            with zip_ref.open('train.csv') as f:
                # Read just a sample to analyze targets
                train_sample = pd.read_csv(f, nrows=10000)
                
            if 'gesture' in train_sample.columns:
                print("\n=== Target Distribution (from sample) ===")
                target_counts = train_sample['gesture'].value_counts()
                print(target_counts)
                
                if 'sequence_type' in train_sample.columns:
                    print("\nSequence type distribution:")
                    print(train_sample['sequence_type'].value_counts())
        except Exception as e:
            print(f"\nError sampling train.csv: {e}")

if __name__ == "__main__":
    explore_demographics('/Users/shail/Downloads/cmi-detect-behavior-with-sensor-data.zip')
