import os
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

class MemoryEfficientTrainer:
    def __init__(self, processor, cache_dir='./cache'):
        """
        Initialize a memory-efficient model trainer
        
        Args:
            processor: MemoryEfficientProcessor instance
            cache_dir: Directory to store cached models and results
        """
        self.processor = processor
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Store models
        self.binary_models = []
        self.multi_models = []
        
        # Initialize scalers
        self.scaler = StandardScaler()
        
        print("Trainer initialized.")
    
    def _competition_metric(self, y_true_binary, y_pred_binary, y_true_multi, y_pred_multi):
        """
        Calculate the competition metric (average of binary F1 and multi-class F1)
        
        Args:
            y_true_binary: True binary labels
            y_pred_binary: Predicted binary labels
            y_true_multi: True multi-class labels
            y_pred_multi: Predicted multi-class labels
        
        Returns:
            score: Competition metric score
        """
        binary_f1 = f1_score(y_true_binary, y_pred_binary, average='macro')
        multi_f1 = f1_score(y_true_multi, y_pred_multi, average='macro')
        return (binary_f1 + multi_f1) / 2
    
    def train_models(self, features_df, n_splits=5, imu_only=False, binary_col='sequence_type',
                    multi_col='gesture', group_col='subject', random_state=42):
        """
        Train models using cross-validation
        
        Args:
            features_df: DataFrame with processed features
            n_splits: Number of cross-validation folds
            imu_only: Whether to use only IMU features
            binary_col: Column name for binary labels
            multi_col: Column name for multi-class labels
            group_col: Column to group by for splitting (to prevent data leakage)
            random_state: Random seed
        
        Returns:
            cv_scores: Dictionary with cross-validation scores
        """
        print(f"Training models with {'IMU-only' if imu_only else 'all'} features")
        
        # Check that required columns are present
        if binary_col not in features_df.columns:
            raise ValueError(f"Binary label column '{binary_col}' not found in features DataFrame")
        if multi_col not in features_df.columns:
            raise ValueError(f"Multi-class label column '{multi_col}' not found in features DataFrame")
        if group_col not in features_df.columns:
            print(f"Warning: Group column '{group_col}' not found. Using standard StratifiedKFold instead.")
            use_group_kfold = False
        else:
            use_group_kfold = True
        
        # Prepare features and labels
        X, y_multi, y_binary = self.processor.prepare_features(
            features_df, 
            label_col=multi_col, 
            binary_col=binary_col,
            drop_cols=[group_col] if use_group_kfold else None,
            imu_only=imu_only
        )
        
        # Set up cross-validation
        if use_group_kfold:
            groups = features_df[group_col].values
            cv = GroupKFold(n_splits=n_splits)
            splits = cv.split(X, y_binary, groups=groups)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            splits = cv.split(X, y_binary)
        
        # Initialize arrays to store predictions
        cv_binary_preds = np.zeros_like(y_binary)
        cv_multi_preds = np.zeros_like(y_multi)
        
        # Store fold scores
        fold_scores = []
        
        # Clear previous models
        self.binary_models = []
        self.multi_models = []
        
        # Train models
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\nTraining fold {fold+1}/{n_splits}")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_binary_train, y_binary_val = y_binary[train_idx], y_binary[val_idx]
            y_multi_train, y_multi_val = y_multi[train_idx], y_multi[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            
            # Train binary classifier
            print("Training binary classifier...")
            binary_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                random_state=random_state + fold,
                n_jobs=-1,
                verbose=0
            )
            binary_model.fit(X_train, y_binary_train)
            
            # Train multi-class classifier
            print("Training multi-class classifier...")
            multi_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=7,
                random_state=random_state + fold,
                n_jobs=-1,
                verbose=0
            )
            multi_model.fit(X_train, y_multi_train)
            
            # Make predictions
            binary_preds = binary_model.predict(X_val)
            multi_preds = multi_model.predict(X_val)
            
            # Store predictions
            cv_binary_preds[val_idx] = binary_preds
            cv_multi_preds[val_idx] = multi_preds
            
            # Calculate scores
            binary_f1 = f1_score(y_binary_val, binary_preds, average='macro')
            multi_f1 = f1_score(y_multi_val, multi_preds, average='macro')
            combined_score = self._competition_metric(
                y_binary_val, binary_preds, y_multi_val, multi_preds
            )
            
            fold_scores.append({
                'fold': fold + 1,
                'binary_f1': binary_f1,
                'multi_f1': multi_f1,
                'combined_score': combined_score
            })
            
            print(f"Fold {fold+1} scores:")
            print(f"  Binary F1: {binary_f1:.4f}")
            print(f"  Multi F1: {multi_f1:.4f}")
            print(f"  Combined Score: {combined_score:.4f}")
            
            # Store models
            self.binary_models.append((binary_model, scaler))
            self.multi_models.append((multi_model, scaler))
            
            # Force garbage collection
            gc.collect()
        
        # Calculate overall cross-validation scores
        overall_binary_f1 = f1_score(y_binary, cv_binary_preds, average='macro')
        overall_multi_f1 = f1_score(y_multi, cv_multi_preds, average='macro')
        overall_combined_score = self._competition_metric(
            y_binary, cv_binary_preds, y_multi, cv_multi_preds
        )
        
        print("\nOverall cross-validation scores:")
        print(f"  Binary F1: {overall_binary_f1:.4f}")
        print(f"  Multi F1: {overall_multi_f1:.4f}")
        print(f"  Combined Score: {overall_combined_score:.4f}")
        
        # Save confusion matrices for both tasks
        self._save_confusion_matrices(y_binary, cv_binary_preds, y_multi, cv_multi_preds, imu_only)
        
        # Return scores
        cv_scores = {
            'fold_scores': fold_scores,
            'overall_binary_f1': overall_binary_f1,
            'overall_multi_f1': overall_multi_f1,
            'overall_combined_score': overall_combined_score
        }
        
        return cv_scores
    
    def _save_confusion_matrices(self, y_binary_true, y_binary_pred, y_multi_true, y_multi_pred, imu_only):
        """Save confusion matrices as plots"""
        # Binary confusion matrix
        plt.figure(figsize=(8, 6))
        binary_cm = confusion_matrix(y_binary_true, y_binary_pred)
        sns.heatmap(
            binary_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.processor.binary_classes_,
            yticklabels=self.processor.binary_classes_
        )
        plt.title(f'Binary Classification Confusion Matrix ({"IMU-only" if imu_only else "All Sensors"})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        binary_cm_path = os.path.join(self.cache_dir, f'binary_cm_{"imu" if imu_only else "all"}.png')
        plt.savefig(binary_cm_path)
        plt.close()
        
        # Multi-class confusion matrix
        plt.figure(figsize=(16, 14))
        multi_cm = confusion_matrix(y_multi_true, y_multi_pred)
        sns.heatmap(
            multi_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.processor.gesture_classes_,
            yticklabels=self.processor.gesture_classes_
        )
        plt.title(f'Multi-class Classification Confusion Matrix ({"IMU-only" if imu_only else "All Sensors"})')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.tight_layout()
        multi_cm_path = os.path.join(self.cache_dir, f'multi_cm_{"imu" if imu_only else "all"}.png')
        plt.savefig(multi_cm_path)
        plt.close()
        
        print(f"Confusion matrices saved to {self.cache_dir}")
    
    def save_models(self, prefix='model'):
        """Save trained models to disk"""
        if not self.binary_models or not self.multi_models:
            raise ValueError("No trained models to save. Run train_models() first.")
        
        # Save binary models
        binary_path = os.path.join(self.cache_dir, f'{prefix}_binary.joblib')
        joblib.dump(self.binary_models, binary_path)
        
        # Save multi-class models
        multi_path = os.path.join(self.cache_dir, f'{prefix}_multi.joblib')
        joblib.dump(self.multi_models, multi_path)
        
        print(f"Models saved to {self.cache_dir}")
    
    def load_models(self, prefix='model'):
        """Load trained models from disk"""
        # Load binary models
        binary_path = os.path.join(self.cache_dir, f'{prefix}_binary.joblib')
        self.binary_models = joblib.load(binary_path)
        
        # Load multi-class models
        multi_path = os.path.join(self.cache_dir, f'{prefix}_multi.joblib')
        self.multi_models = joblib.load(multi_path)
        
        print(f"Models loaded from {self.cache_dir}")
    
    def predict(self, X):
        """
        Make predictions using trained models
        
        Args:
            X: Feature matrix
        
        Returns:
            binary_preds: Binary predictions
            multi_preds: Multi-class predictions
        """
        if not self.binary_models or not self.multi_models:
            raise ValueError("No trained models available. Run train_models() first or load saved models.")
        
        # Initialize arrays to store predictions
        binary_preds_proba = np.zeros((X.shape[0], len(self.binary_models[0][0].classes_)))
        multi_preds_proba = np.zeros((X.shape[0], len(self.multi_models[0][0].classes_)))
        
        # Make predictions with each model
        for (binary_model, binary_scaler), (multi_model, multi_scaler) in zip(self.binary_models, self.multi_models):
            # Scale features
            X_scaled = binary_scaler.transform(X)
            
            # Make predictions
            binary_preds_proba += binary_model.predict_proba(X_scaled)
            multi_preds_proba += multi_model.predict_proba(X_scaled)
        
        # Average predictions
        binary_preds_proba /= len(self.binary_models)
        multi_preds_proba /= len(self.multi_models)
        
        # Get class predictions
        binary_preds = np.argmax(binary_preds_proba, axis=1)
        multi_preds = np.argmax(multi_preds_proba, axis=1)
        
        return binary_preds, multi_preds, binary_preds_proba, multi_preds_proba

if __name__ == "__main__":
    from data_processor import MemoryEfficientProcessor
    
    # Test trainer with a small dataset
    processor = MemoryEfficientProcessor('/Users/shail/Downloads/cmi-detect-behavior-with-sensor-data.zip')
    
    # Check if we've already processed features
    if os.path.exists('test_features.csv'):
        print("Loading previously processed features...")
        features_df = pd.read_csv('test_features.csv')
    else:
        # Process a small number of sequences for testing
        print("Processing features...")
        features_df = processor.process_sequences_by_chunk(
            output_file='test_features.csv',
            max_sequences=100,  # Process only 100 sequences for testing
            include_tof=True
        )
    
    # Create trainer
    trainer = MemoryEfficientTrainer(processor)
    
    # Train models
    cv_scores = trainer.train_models(features_df, n_splits=3, imu_only=False)
    
    # Save models
    trainer.save_models(prefix='test_model')
