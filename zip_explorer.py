import os
import zipfile
import pandas as pd
import numpy as np
import sys
import gc
from tqdm import tqdm

# Configure pandas to use less memory
pd.options.mode.chained_assignment = None

def get_zip_structure(zip_path):
    """Examine zip file structure without full extraction"""
    print(f"Analyzing zip file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f"\nFound {len(file_list)} files in the archive:")
        
        # Group files by extension for better organization
        extensions = {}
        for file in file_list:
            ext = os.path.splitext(file)[1]
            if ext not in extensions:
                extensions[ext] = []
            extensions[ext].append(file)
        
        # Print file structure by extension
        for ext, files in extensions.items():
            print(f"\n{ext} files ({len(files)}):")
            for file in sorted(files):
                size = zip_ref.getinfo(file).file_size
                print(f"  - {file} ({size / (1024*1024):.2f} MB)")
        
        # If there are CSV files, analyze the first one to understand data structure
        csv_files = [f for f in file_list if f.endswith('.csv')]
        if csv_files:
            sample_file = csv_files[0]
            print(f"\nAnalyzing sample from {sample_file}...")
            
            # Read only first 1000 rows to analyze structure without loading entire file
            with zip_ref.open(sample_file) as f:
                sample_df = pd.read_csv(f, nrows=1000)
                
            print(f"\nSample DataFrame shape: {sample_df.shape}")
            print("\nColumn names:")
            for col in sample_df.columns:
                print(f"  - {col}")
            
            print("\nData types:")
            for col, dtype in sample_df.dtypes.items():
                print(f"  - {col}: {dtype}")
            
            print("\nMemory usage per column (MB):")
            for col, mem in sample_df.memory_usage(deep=True).items():
                if col != 'Index':
                    print(f"  - {col}: {mem / (1024*1024):.6f} MB")

            # Check for missing values
            missing_data = sample_df.isna().sum()
            if missing_data.sum() > 0:
                print("\nColumns with missing values:")
                for col, missing_count in missing_data.items():
                    if missing_count > 0:
                        percent = (missing_count / len(sample_df)) * 100
                        print(f"  - {col}: {missing_count} missing values ({percent:.2f}%)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python zip_explorer.py <path_to_zip_file>")
        return
    
    zip_path = sys.argv[1]
    get_zip_structure(zip_path)

if __name__ == "__main__":
    main()
