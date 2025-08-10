import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processor import MemoryEfficientProcessor
from model_trainer import MemoryEfficientTrainer

# File paths
ZIP_PATH = '/Users/shail/Downloads/cmi-detect-behavior-with-sensor-data.zip'
CACHE_DIR = './cache'
FEATURES_FILE_IMU = './features_imu_only.csv'
FEATURES_FILE_ALL = './features_all_sensors.csv'

# Memory usage tracking
import psutil
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")

def process_data(max_sequences=None):
    """Process the data and extract features"""
    # Create processor
    processor = MemoryEfficientProcessor(ZIP_PATH, cache_dir=CACHE_DIR)
    
    # Process sequences with all sensors
    if not os.path.exists(FEATURES_FILE_ALL) or max_sequences is not None:
        print("\n===== Processing sequences with all sensors =====")
        print_memory_usage()
        features_all = processor.process_sequences_by_chunk(
            output_file=FEATURES_FILE_ALL,
            max_sequences=max_sequences,
            include_tof=True
        )
        print_memory_usage()
    else:
        print(f"\nLoading pre-processed all-sensor features from {FEATURES_FILE_ALL}")
        features_all = pd.read_csv(FEATURES_FILE_ALL)
    
    # Save processor
    processor.save_processor(filename='processor_all_sensors.joblib')
    
    return processor, features_all

def train_and_evaluate(processor, features_df, max_sequences=None):
    """Train and evaluate models using IMU-only and all sensor data"""
    # Initialize trainer
    trainer = MemoryEfficientTrainer(processor, cache_dir=CACHE_DIR)
    
    # Limit data if specified
    if max_sequences is not None:
        features_df = features_df.head(max_sequences)
    
    # Train and evaluate with IMU-only features
    print("\n===== Training with IMU-only features =====")
    print_memory_usage()
    imu_scores = trainer.train_models(features_df, imu_only=True, n_splits=5)
    print_memory_usage()
    trainer.save_models(prefix='imu_only')
    
    # Train and evaluate with all sensors
    print("\n===== Training with all sensor features =====")
    print_memory_usage()
    all_scores = trainer.train_models(features_df, imu_only=False, n_splits=5)
    print_memory_usage()
    trainer.save_models(prefix='all_sensors')
    
    # Compare results
    compare_results(imu_scores, all_scores)
    
    return imu_scores, all_scores

def compare_results(imu_scores, all_scores):
    """Compare results between IMU-only and all-sensor models"""
    print("\n===== Comparing IMU-only vs. All Sensors =====")
    
    # Format for pretty printing
    print(f"{'Metric':<25} {'IMU-only':<15} {'All Sensors':<15} {'Improvement':<15}")
    print("-" * 70)
    
    # Binary F1 scores
    imu_binary_f1 = imu_scores['overall_binary_f1']
    all_binary_f1 = all_scores['overall_binary_f1']
    binary_diff = all_binary_f1 - imu_binary_f1
    print(f"{'Binary F1':<25} {imu_binary_f1:.4f}{' ':<10} {all_binary_f1:.4f}{' ':<10} {binary_diff:.4f} ({binary_diff/imu_binary_f1*100:.2f}%)")
    
    # Multi-class F1 scores
    imu_multi_f1 = imu_scores['overall_multi_f1']
    all_multi_f1 = all_scores['overall_multi_f1']
    multi_diff = all_multi_f1 - imu_multi_f1
    print(f"{'Multi-class F1':<25} {imu_multi_f1:.4f}{' ':<10} {all_multi_f1:.4f}{' ':<10} {multi_diff:.4f} ({multi_diff/imu_multi_f1*100:.2f}%)")
    
    # Combined scores
    imu_combined = imu_scores['overall_combined_score']
    all_combined = all_scores['overall_combined_score']
    combined_diff = all_combined - imu_combined
    print(f"{'Combined Score':<25} {imu_combined:.4f}{' ':<10} {all_combined:.4f}{' ':<10} {combined_diff:.4f} ({combined_diff/imu_combined*100:.2f}%)")
    
    # Create a bar chart
    metrics = ['Binary F1', 'Multi-class F1', 'Combined Score']
    imu_values = [imu_binary_f1, imu_multi_f1, imu_combined]
    all_values = [all_binary_f1, all_multi_f1, all_combined]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, imu_values, width, label='IMU-only')
    plt.bar(x + width/2, all_values, width, label='All Sensors')
    
    plt.ylabel('Score')
    plt.title('Performance Comparison: IMU-only vs. All Sensors')
    plt.xticks(x, metrics)
    plt.legend()
    
    plt.savefig(os.path.join(CACHE_DIR, 'comparison_chart.png'))
    plt.close()
    
    print(f"\nComparison chart saved to {os.path.join(CACHE_DIR, 'comparison_chart.png')}")

def main():
    """Main function to run the pipeline"""
    start_time = time.time()
    
    try:
        # Install psutil if not available (for memory tracking)
        try:
            import psutil
        except ImportError:
            import subprocess
            print("Installing psutil...")
            subprocess.check_call(['pip', 'install', 'psutil'])
            import psutil
        
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # For testing, limit to a small number of sequences
        # For full training, set max_sequences=None
        max_sequences = 300  # Limit for demonstration
        
        # Process data
        processor, features_df = process_data(max_sequences=max_sequences)
        
        # Train and evaluate models
        imu_scores, all_scores = train_and_evaluate(processor, features_df, max_sequences=max_sequences)
        
        # Print total runtime
        total_time = time.time() - start_time
        print(f"\nTotal runtime: {total_time/60:.2f} minutes")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
