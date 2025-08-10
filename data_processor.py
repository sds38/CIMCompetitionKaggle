import os
import zipfile
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')

class MemoryEfficientProcessor:
    def __init__(self, zip_path, cache_dir='./cache'):
        """
        Initialize a memory-efficient data processor for the competition.
        
        Args:
            zip_path: Path to the competition zip file
            cache_dir: Directory to store cached data and models
        """
        self.zip_path = zip_path
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize encoders
        self.gesture_encoder = LabelEncoder()
        self.binary_encoder = LabelEncoder()
        
        # Feature group definitions
        self.imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
        self.thermo_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
        self.tof_prefixes = [f'tof_{i}_' for i in range(1, 6)]
        
        print("Processor initialized.")
    
    def _list_csv_files(self):
        """List all CSV files in the zip archive"""
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            return [f for f in zip_ref.namelist() if f.endswith('.csv')]
    
    def _get_demographics(self):
        """Load demographics data"""
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open('train_demographics.csv') as f:
                train_demo = pd.read_csv(f)
            
            try:
                with zip_ref.open('test_demographics.csv') as f:
                    test_demo = pd.read_csv(f)
            except:
                test_demo = None
                
        return train_demo, test_demo
    
    def _get_unique_sequences(self, chunk_size=10000, max_chunks=None):
        """Get all unique sequence IDs from the training data"""
        unique_sequences = set()
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open('train.csv') as f:
                chunk_count = 0
                for chunk in tqdm(pd.read_csv(f, chunksize=chunk_size), desc="Collecting sequence IDs"):
                    unique_sequences.update(chunk['sequence_id'].unique())
                    chunk_count += 1
                    
                    if max_chunks is not None and chunk_count >= max_chunks:
                        print(f"Warning: Only processed {max_chunks} chunks for sequence ID collection")
                        break
                    
                    # Force garbage collection
                    gc.collect()
                    
        return list(unique_sequences)
    
    def _extract_sequence_features(self, sequence_df, include_tof=True):
        """
        Extract features from a single sequence
        
        Args:
            sequence_df: DataFrame containing data for a single sequence
            include_tof: Whether to include time-of-flight sensor data
        
        Returns:
            Dictionary of features for the sequence
        """
        features = {}
        
        # Basic metadata features
        features['subject'] = sequence_df['subject'].iloc[0]
        if 'gesture' in sequence_df.columns:
            features['gesture'] = sequence_df['gesture'].iloc[0]
        if 'sequence_type' in sequence_df.columns:
            features['sequence_type'] = sequence_df['sequence_type'].iloc[0]
            
        # ===== IMU Features =====
        for col in self.imu_cols:
            if col in sequence_df.columns:
                # Basic statistical features
                features[f'{col}_mean'] = sequence_df[col].mean()
                features[f'{col}_std'] = sequence_df[col].std()
                features[f'{col}_min'] = sequence_df[col].min()
                features[f'{col}_max'] = sequence_df[col].max()
                features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
                features[f'{col}_median'] = sequence_df[col].median()
                features[f'{col}_q25'] = sequence_df[col].quantile(0.25)
                features[f'{col}_q75'] = sequence_df[col].quantile(0.75)
                features[f'{col}_iqr'] = features[f'{col}_q75'] - features[f'{col}_q25']
                
                # Rate of change features
                diff = np.diff(sequence_df[col].values)
                features[f'{col}_diff_mean'] = diff.mean() if len(diff) > 0 else 0
                features[f'{col}_diff_std'] = diff.std() if len(diff) > 0 else 0
                features[f'{col}_diff_max'] = diff.max() if len(diff) > 0 else 0
                features[f'{col}_diff_min'] = diff.min() if len(diff) > 0 else 0
        
        # ===== Thermopile Features =====
        if include_tof:  # Only include if we're using full sensor data
            for col in self.thermo_cols:
                if col in sequence_df.columns:
                    # Basic statistical features
                    features[f'{col}_mean'] = sequence_df[col].mean()
                    features[f'{col}_std'] = sequence_df[col].std()
                    features[f'{col}_min'] = sequence_df[col].min()
                    features[f'{col}_max'] = sequence_df[col].max()
                    features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
                    
                    # Missing value count
                    features[f'{col}_missing'] = sequence_df[col].isna().sum() / len(sequence_df)
            
            # ===== Time-of-Flight Features =====
            # For TOF data, we'll compute aggregated features across all 64 pixels
            for prefix in self.tof_prefixes:
                tof_cols = [col for col in sequence_df.columns if col.startswith(prefix)]
                if tof_cols:
                    # Get all TOF data for this sensor
                    tof_data = sequence_df[tof_cols].values
                    
                    # Replace -1 values (no reflection) with NaN
                    tof_data[tof_data == -1] = np.nan
                    
                    # Compute aggregated features
                    features[f'{prefix}mean'] = np.nanmean(tof_data)
                    features[f'{prefix}std'] = np.nanstd(tof_data)
                    features[f'{prefix}min'] = np.nanmin(tof_data)
                    features[f'{prefix}max'] = np.nanmax(tof_data)
                    features[f'{prefix}median'] = np.nanmedian(tof_data)
                    features[f'{prefix}missing_rate'] = np.isnan(tof_data).sum() / tof_data.size
                    
                    # Compute how many pixels are active (non-NaN) on average
                    active_pixels = np.sum(~np.isnan(tof_data), axis=1)
                    features[f'{prefix}active_pixels_mean'] = np.mean(active_pixels)
                    features[f'{prefix}active_pixels_std'] = np.std(active_pixels)
        
        return features
    
    def process_sequences_by_chunk(self, output_file='processed_features.csv', 
                                 chunk_size=5000, max_sequences=None, 
                                 include_tof=True):
        """
        Process sequences in chunks and save features to a CSV file
        
        Args:
            output_file: Path to save processed features
            chunk_size: Number of rows to read in each chunk
            max_sequences: Maximum number of sequences to process (for testing)
            include_tof: Whether to include time-of-flight sensor data
        """
        # Get sequence IDs (optionally limited for testing)
        print("Collecting unique sequence IDs...")
        sequence_ids = self._get_unique_sequences(chunk_size=chunk_size, 
                                                  max_chunks=None if max_sequences is None else max_sequences//10)
        
        if max_sequences is not None:
            sequence_ids = sequence_ids[:max_sequences]
            print(f"Limited to {max_sequences} sequences for testing.")
        
        print(f"Found {len(sequence_ids)} unique sequences.")
        
        # Process sequences in chunks
        processed_sequences = []
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open('train.csv') as f:
                # Read data in chunks
                for chunk in tqdm(pd.read_csv(f, chunksize=chunk_size), desc="Processing chunks"):
                    # Group by sequence_id
                    for sequence_id, group in chunk.groupby('sequence_id'):
                        if sequence_id in sequence_ids:
                            # Extract features from this sequence
                            features = self._extract_sequence_features(group, include_tof=include_tof)
                            features['sequence_id'] = sequence_id
                            processed_sequences.append(features)
                            
                            # Remove this ID from our list to avoid duplicates
                            sequence_ids.remove(sequence_id)
                    
                    # If we've processed all sequences, we can stop
                    if not sequence_ids:
                        break
                    
                    # Force garbage collection
                    gc.collect()
        
        # Convert to DataFrame
        print("Converting to DataFrame...")
        features_df = pd.DataFrame(processed_sequences)
        
        # Save to CSV
        print(f"Saving {len(features_df)} processed sequences to {output_file}")
        features_df.to_csv(output_file, index=False)
        
        return features_df
    
    def prepare_features(self, features_df, label_col='gesture', binary_col='sequence_type',
                        drop_cols=None, imu_only=False):
        """
        Prepare features for model training
        
        Args:
            features_df: DataFrame of processed features
            label_col: Column name for the target labels
            binary_col: Column name for binary classification
            drop_cols: Columns to drop before training
            imu_only: Whether to use only IMU features
            
        Returns:
            X: Feature matrix
            y_multi: Labels for multi-class classification
            y_binary: Labels for binary classification
        """
        # Make a copy to avoid modifying the original
        df = features_df.copy()
        
        # Encode labels
        if label_col in df.columns:
            self.gesture_encoder.fit(df[label_col])
            y_multi = self.gesture_encoder.transform(df[label_col])
            # Save mapping for later
            self.gesture_classes_ = self.gesture_encoder.classes_
        else:
            y_multi = None
        
        if binary_col in df.columns:
            self.binary_encoder.fit(df[binary_col])
            y_binary = self.binary_encoder.transform(df[binary_col])
            # Save mapping for later
            self.binary_classes_ = self.binary_encoder.classes_
        else:
            y_binary = None
        
        # Drop non-feature columns
        cols_to_drop = ['sequence_id']
        if label_col in df.columns:
            cols_to_drop.append(label_col)
        if binary_col in df.columns:
            cols_to_drop.append(binary_col)
        if drop_cols is not None:
            cols_to_drop.extend(drop_cols)
        
        df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # If IMU only, drop thermopile and TOF features
        if imu_only:
            for col in df.columns:
                if any(col.startswith(f'thm_') for thm in self.thermo_cols) or \
                   any(col.startswith(prefix) for prefix in self.tof_prefixes):
                    df = df.drop(columns=[col], errors='ignore')
        
        # Handle missing values
        df = df.fillna(0)
        
        # Return features and labels
        return df.values, y_multi, y_binary
    
    def save_processor(self, filename='data_processor.joblib'):
        """Save the processor object for later use"""
        filepath = os.path.join(self.cache_dir, filename)
        joblib.dump(self, filepath)
        print(f"Processor saved to {filepath}")
    
    @classmethod
    def load_processor(cls, filepath):
        """Load a saved processor object"""
        return joblib.load(filepath)


if __name__ == "__main__":
    # Simple test
    processor = MemoryEfficientProcessor('/Users/shail/Downloads/cmi-detect-behavior-with-sensor-data.zip')
    
    # Process a small number of sequences for testing
    features_df = processor.process_sequences_by_chunk(
        output_file='test_features.csv',
        max_sequences=100,  # Process only 100 sequences for testing
        include_tof=True
    )
    
    print(f"Processed {len(features_df)} sequences")
    print(f"Feature columns: {features_df.columns}")
    
    # Save the processor
    processor.save_processor()
