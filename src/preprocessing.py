import pandas as pd
import numpy as np
import zipfile
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm
import joblib
import gc
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder


class SensorDataProcessor:
    """
    Memory-efficient data processor for CMI BFRB sensor data.
    
    Features:
    - Chunked reading to handle large files
    - Reshaping ToF data to 8x8 grids
    - Feature extraction for both raw and statistical approaches
    - Support for IMU-only and full sensor scenarios
    """
    
    def __init__(self, zip_path: str, cache_dir: str = './cache'):
        """
        Initialize the data processor.
        
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
        
        # Define sensor columns
        self.imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
        self.thermo_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
        self.tof_prefixes = [f'tof_{i}_' for i in range(1, 6)]
        
        print("Sensor Data Processor initialized.")
    
    def load_demographics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load demographics data from the zip file"""
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            # Load train demographics
            with zip_ref.open('train_demographics.csv') as f:
                train_demo = pd.read_csv(f)
            
            # Load test demographics
            with zip_ref.open('test_demographics.csv') as f:
                test_demo = pd.read_csv(f)
        
        return train_demo, test_demo
    
    def get_sequence_ids(self, file_path: str, chunk_size: int = 10000, max_chunks: Optional[int] = None) -> List[str]:
        """
        Get all unique sequence IDs from the dataset.
        
        Args:
            file_path: Path within the zip file
            chunk_size: Number of rows to read in each chunk
            max_chunks: Maximum number of chunks to process (for testing)
            
        Returns:
            List of unique sequence IDs
        """
        unique_sequences = set()
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open(file_path) as f:
                chunk_count = 0
                for chunk in tqdm(pd.read_csv(f, chunksize=chunk_size), desc="Collecting sequence IDs"):
                    unique_sequences.update(chunk['sequence_id'].unique())
                    chunk_count += 1
                    
                    if max_chunks is not None and chunk_count >= max_chunks:
                        print(f"Warning: Only processed {max_chunks} chunks for sequence ID collection")
                        break
                    
                    # Force garbage collection
                    gc.collect()
        
        return sorted(list(unique_sequences))
    
    def extract_sequence(self, file_path: str, sequence_id: str, chunk_size: int = 10000) -> pd.DataFrame:
        """
        Extract a single sequence from the dataset.
        
        Args:
            file_path: Path within the zip file
            sequence_id: ID of the sequence to extract
            chunk_size: Number of rows to read in each chunk
            
        Returns:
            DataFrame containing the sequence data
        """
        sequence_data = None
        
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            with zip_ref.open(file_path) as f:
                for chunk in pd.read_csv(f, chunksize=chunk_size):
                    # Find rows matching the sequence ID
                    seq_rows = chunk[chunk['sequence_id'] == sequence_id]
                    
                    # If we found rows, concatenate them
                    if len(seq_rows) > 0:
                        if sequence_data is None:
                            sequence_data = seq_rows
                        else:
                            sequence_data = pd.concat([sequence_data, seq_rows])
                    
                    # If we found the whole sequence, break
                    if 'sequence_counter' in seq_rows.columns:
                        if seq_rows['sequence_counter'].max() == seq_rows['sequence_counter'].count() - 1:
                            break
                    
                    # Force garbage collection
                    gc.collect()
        
        return sequence_data
    
    def preprocess_sequence(self, sequence_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Preprocess a single sequence for model input.
        
        Args:
            sequence_df: DataFrame containing a single sequence
            
        Returns:
            Dictionary with processed data arrays:
            - 'imu': Array of shape [seq_length, 7]
            - 'thermo': Array of shape [seq_length, 5]
            - 'tof': Array of shape [seq_length, 320] (5 sensors * 64 pixels)
            - 'metadata': Dict with sequence metadata
        """
        # Get sequence metadata
        metadata = {
            'sequence_id': sequence_df['sequence_id'].iloc[0],
            'subject': sequence_df['subject'].iloc[0]
        }
        
        if 'gesture' in sequence_df.columns:
            metadata['gesture'] = sequence_df['gesture'].iloc[0]
        if 'sequence_type' in sequence_df.columns:
            metadata['sequence_type'] = sequence_df['sequence_type'].iloc[0]
        
        # Sort by sequence_counter if available
        if 'sequence_counter' in sequence_df.columns:
            sequence_df = sequence_df.sort_values('sequence_counter')
        
        # Extract IMU data
        imu_data = sequence_df[self.imu_cols].values
        
        # Extract thermopile data
        thermo_data = sequence_df[self.thermo_cols].values
        
        # Extract ToF data
        tof_data = np.zeros((len(sequence_df), 5 * 64), dtype=np.float32)
        
        # Check if we have ToF data or if this is IMU-only
        tof_columns = [col for col in sequence_df.columns if col.startswith('tof_')]
        has_tof_data = len(tof_columns) > 0 and not sequence_df[tof_columns].isna().all().all()
        
        if has_tof_data:
            for i, prefix in enumerate(self.tof_prefixes):
                tof_cols = [col for col in sequence_df.columns if col.startswith(prefix)]
                tof_data[:, i*64:(i+1)*64] = sequence_df[tof_cols].values
        else:
            tof_data.fill(np.nan)
        
        return {
            'imu': imu_data,
            'thermo': thermo_data,
            'tof': tof_data,
            'metadata': metadata
        }
    
    def extract_statistical_features(self, sequence_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Extract statistical features from sequence data.
        
        Args:
            sequence_data: Dictionary from preprocess_sequence()
            
        Returns:
            Dictionary with statistical features
        """
        features = {}
        
        # Copy metadata
        features.update(sequence_data['metadata'])
        
        # === IMU Features ===
        imu_data = sequence_data['imu']
        for i, col in enumerate(self.imu_cols):
            col_data = imu_data[:, i]
            
            # Basic statistical features
            features[f'{col}_mean'] = np.mean(col_data)
            features[f'{col}_std'] = np.std(col_data)
            features[f'{col}_min'] = np.min(col_data)
            features[f'{col}_max'] = np.max(col_data)
            features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
            features[f'{col}_median'] = np.median(col_data)
            features[f'{col}_q25'] = np.percentile(col_data, 25)
            features[f'{col}_q75'] = np.percentile(col_data, 75)
            features[f'{col}_iqr'] = features[f'{col}_q75'] - features[f'{col}_q25']
            
            # Rate of change features
            diff = np.diff(col_data)
            if len(diff) > 0:
                features[f'{col}_diff_mean'] = np.mean(diff)
                features[f'{col}_diff_std'] = np.std(diff)
                features[f'{col}_diff_max'] = np.max(diff)
                features[f'{col}_diff_min'] = np.min(diff)
        
        # === Thermopile Features ===
        thermo_data = sequence_data['thermo']
        for i, col in enumerate(self.thermo_cols):
            col_data = thermo_data[:, i]
            
            # Check for all NaN
            if np.isnan(col_data).all():
                features[f'{col}_mean'] = 0
                features[f'{col}_std'] = 0
                features[f'{col}_min'] = 0
                features[f'{col}_max'] = 0
                features[f'{col}_range'] = 0
                features[f'{col}_missing'] = 1.0
                continue
            
            # Basic statistical features
            features[f'{col}_mean'] = np.nanmean(col_data)
            features[f'{col}_std'] = np.nanstd(col_data)
            features[f'{col}_min'] = np.nanmin(col_data)
            features[f'{col}_max'] = np.nanmax(col_data)
            features[f'{col}_range'] = features[f'{col}_max'] - features[f'{col}_min']
            
            # Missing value count
            features[f'{col}_missing'] = np.isnan(col_data).mean()
        
        # === ToF Features ===
        tof_data = sequence_data['tof']
        
        # Check if we have ToF data or if this is IMU-only
        has_tof_data = not np.isnan(tof_data).all()
        
        if has_tof_data:
            for i, prefix in enumerate(self.tof_prefixes):
                sensor_data = tof_data[:, i*64:(i+1)*64]
                
                # Replace -1 values (no reflection) with NaN
                sensor_data[sensor_data == -1] = np.nan
                
                # Compute aggregated features
                features[f'{prefix}mean'] = np.nanmean(sensor_data)
                features[f'{prefix}std'] = np.nanstd(sensor_data)
                features[f'{prefix}min'] = np.nanmin(sensor_data)
                features[f'{prefix}max'] = np.nanmax(sensor_data)
                features[f'{prefix}median'] = np.nanmedian(sensor_data)
                features[f'{prefix}missing_rate'] = np.isnan(sensor_data).mean()
                
                # Compute how many pixels are active (non-NaN) on average
                active_pixels = np.sum(~np.isnan(sensor_data), axis=1)
                features[f'{prefix}active_pixels_mean'] = np.mean(active_pixels)
                features[f'{prefix}active_pixels_std'] = np.std(active_pixels)
        else:
            # Fill with zeros for IMU-only data
            for prefix in self.tof_prefixes:
                features[f'{prefix}mean'] = 0
                features[f'{prefix}std'] = 0
                features[f'{prefix}min'] = 0
                features[f'{prefix}max'] = 0
                features[f'{prefix}median'] = 0
                features[f'{prefix}missing_rate'] = 1.0
                features[f'{prefix}active_pixels_mean'] = 0
                features[f'{prefix}active_pixels_std'] = 0
        
        return features
    
    def process_sequences_batch(
        self, 
        file_path: str, 
        output_file: str,
        sequence_ids: Optional[List[str]] = None,
        max_sequences: Optional[int] = None,
        chunk_size: int = 10000
    ) -> pd.DataFrame:
        """
        Process multiple sequences and save features to a file.
        
        Args:
            file_path: Path within the zip file
            output_file: Path to save the processed features
            sequence_ids: List of sequence IDs to process (if None, will get all)
            max_sequences: Maximum number of sequences to process (for testing)
            chunk_size: Number of rows to read in each chunk
            
        Returns:
            DataFrame with processed features
        """
        if sequence_ids is None:
            sequence_ids = self.get_sequence_ids(file_path, chunk_size)
        
        if max_sequences is not None:
            sequence_ids = sequence_ids[:max_sequences]
        
        print(f"Processing {len(sequence_ids)} sequences")
        
        all_features = []
        for seq_id in tqdm(sequence_ids):
            # Extract sequence
            seq_df = self.extract_sequence(file_path, seq_id, chunk_size)
            
            # Preprocess sequence
            seq_data = self.preprocess_sequence(seq_df)
            
            # Extract features
            features = self.extract_statistical_features(seq_data)
            
            all_features.append(features)
            
            # Force garbage collection
            gc.collect()
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Save to file
        features_df.to_csv(output_file, index=False)
        
        return features_df
    
    def prepare_for_training(
        self, 
        features_df: pd.DataFrame, 
        imu_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for training.
        
        Args:
            features_df: DataFrame with processed features
            imu_only: Whether to use only IMU features
            
        Returns:
            X: Features array
            y_binary: Binary labels (Target/Non-target)
            y_multi: Multi-class labels (Gesture type)
        """
        # Encode labels
        if 'gesture' in features_df.columns:
            self.gesture_encoder.fit(features_df['gesture'])
            y_multi = self.gesture_encoder.transform(features_df['gesture'])
        else:
            y_multi = None
        
        if 'sequence_type' in features_df.columns:
            self.binary_encoder.fit(features_df['sequence_type'])
            y_binary = self.binary_encoder.transform(features_df['sequence_type'])
        else:
            y_binary = None
        
        # Drop non-feature columns
        feature_cols = features_df.columns.drop([
            'sequence_id', 'subject', 'gesture', 'sequence_type'
        ], errors='ignore')
        
        # If IMU-only, drop thermopile and ToF features
        if imu_only:
            imu_feature_cols = [col for col in feature_cols if any(
                col.startswith(imu_col) for imu_col in self.imu_cols
            )]
            X = features_df[imu_feature_cols].values
        else:
            X = features_df[feature_cols].values
        
        return X, y_binary, y_multi
    
    def reshape_tof_for_cnn(self, sequence_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reshape ToF data for CNN processing.
        
        Args:
            sequence_data: Dictionary from preprocess_sequence()
            
        Returns:
            imu_tensor: Array of shape [seq_length, 7]
            thermo_tensor: Array of shape [seq_length, 5]
            tof_tensor: Array of shape [seq_length, 5, 8, 8]
        """
        seq_length = len(sequence_data['imu'])
        
        # IMU data can be used directly
        imu_tensor = sequence_data['imu']
        
        # Thermopile data can be used directly
        thermo_tensor = sequence_data['thermo']
        
        # Reshape ToF data from [seq_length, 320] to [seq_length, 5, 8, 8]
        tof_flat = sequence_data['tof']
        tof_tensor = np.zeros((seq_length, 5, 8, 8), dtype=np.float32)
        
        # Check if we have ToF data or if this is IMU-only
        has_tof_data = not np.isnan(tof_flat).all()
        
        if has_tof_data:
            for t in range(seq_length):
                for sensor in range(5):
                    # Get the data for this sensor at this time step
                    sensor_data = tof_flat[t, sensor*64:(sensor+1)*64]
                    
                    # Replace -1 values (no reflection) with 0
                    sensor_data[sensor_data == -1] = 0
                    
                    # Reshape to 8x8 grid
                    tof_tensor[t, sensor] = sensor_data.reshape(8, 8)
        
        return imu_tensor, thermo_tensor, tof_tensor
    
    def save_processor(self, filename: str = 'processor.joblib'):
        """Save processor to file"""
        path = os.path.join(self.cache_dir, filename)
        joblib.dump(self, path)
        print(f"Processor saved to {path}")
    
    @classmethod
    def load_processor(cls, path: str):
        """Load processor from file"""
        return joblib.load(path)
