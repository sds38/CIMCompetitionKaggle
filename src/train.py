import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import joblib
import gc

# Import local modules
from preprocessing import SensorDataProcessor
from models import MultiSensorModel, SimplifiedModel


class SensorFeatureDataset(Dataset):
    """Dataset for statistical features extracted from sensor data."""
    
    def __init__(self, features, binary_labels, multi_labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.binary_labels = torch.tensor(binary_labels, dtype=torch.long)
        
        if multi_labels is not None:
            self.multi_labels = torch.tensor(multi_labels, dtype=torch.long)
        else:
            self.multi_labels = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.multi_labels is not None:
            return self.features[idx], self.binary_labels[idx], self.multi_labels[idx]
        else:
            return self.features[idx], self.binary_labels[idx], -1


def competition_score(binary_preds, binary_true, multi_preds, multi_true):
    """
    Calculate the competition metric according to Kaggle's requirements:
    1. Binary F1: Macro F1 score for target vs non-target classification
    2. Gesture F1: Macro F1 score for gesture classification, where only target sequences are considered
    Final score is the average of these two components.
    """
    # 1. Binary F1 (target vs non-target)
    binary_f1 = f1_score(binary_true, binary_preds, average='macro')
    
    # 2. Multi-class F1 (gesture classification for target sequences only)
    # Only evaluate on target sequences (where binary_true == 1)
    is_target = binary_true == 1
    
    # If there are target sequences in this batch, calculate F1
    if np.sum(is_target) > 0:
        multi_f1 = f1_score(
            multi_true[is_target],  # True gesture labels for target sequences
            multi_preds[is_target],  # Predicted gesture labels for target sequences
            average='macro'
        )
    else:
        # No target sequences in this batch
        multi_f1 = 0.0
    
    # Final score is the average of the two components
    final_score = (binary_f1 + multi_f1) / 2
    
    return final_score, binary_f1, multi_f1


def train_epoch(model, train_loader, optimizer, criterion_binary, criterion_multi, device):
    model.train()
    train_loss = 0.0
    
    for batch_idx, (features, binary_labels, multi_labels) in enumerate(tqdm(train_loader, desc="Training")):
        # Move data to device
        features = features.to(device)
        binary_labels = binary_labels.to(device)
        multi_labels = multi_labels.to(device)
        
        # Forward pass
        binary_logits, multi_logits = model(features)
        
        # Calculate loss
        binary_loss = criterion_binary(binary_logits, binary_labels)
        
        # Only calculate multi-class loss for target sequences
        is_target = binary_labels == 1
        if torch.sum(is_target) > 0:
            multi_loss = criterion_multi(
                multi_logits[is_target], 
                multi_labels[is_target]
            )
            loss = binary_loss + multi_loss
        else:
            loss = binary_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    return train_loss / len(train_loader)


def validate(model, val_loader, criterion_binary, criterion_multi, device):
    model.eval()
    val_loss = 0.0
    
    all_binary_preds = []
    all_binary_labels = []
    all_multi_preds = []
    all_multi_labels = []
    
    with torch.no_grad():
        for batch_idx, (features, binary_labels, multi_labels) in enumerate(tqdm(val_loader, desc="Validation")):
            # Move data to device
            features = features.to(device)
            binary_labels = binary_labels.to(device)
            multi_labels = multi_labels.to(device)
            
            # Forward pass
            binary_logits, multi_logits = model(features)
            
            # Calculate loss
            binary_loss = criterion_binary(binary_logits, binary_labels)
            
            # Only calculate multi-class loss for target sequences
            is_target = binary_labels == 1
            if torch.sum(is_target) > 0:
                multi_loss = criterion_multi(
                    multi_logits[is_target], 
                    multi_labels[is_target]
                )
                loss = binary_loss + multi_loss
            else:
                loss = binary_loss
            
            val_loss += loss.item()
            
            # Store predictions
            binary_preds = torch.argmax(binary_logits, dim=1)
            multi_preds = torch.argmax(multi_logits, dim=1)
            
            all_binary_preds.append(binary_preds.cpu().numpy())
            all_binary_labels.append(binary_labels.cpu().numpy())
            all_multi_preds.append(multi_preds.cpu().numpy())
            all_multi_labels.append(multi_labels.cpu().numpy())
    
    # Concatenate predictions
    binary_preds = np.concatenate(all_binary_preds)
    binary_labels = np.concatenate(all_binary_labels)
    multi_preds = np.concatenate(all_multi_preds)
    multi_labels = np.concatenate(all_multi_labels)
    
    # Calculate metrics
    score, binary_f1, multi_f1 = competition_score(
        binary_preds, binary_labels, multi_preds, multi_labels
    )
    
    return val_loss / len(val_loader), score, binary_f1, multi_f1, binary_preds, multi_preds


def plot_confusion_matrices(binary_true, binary_preds, multi_true, multi_preds, 
                           binary_classes, multi_classes, output_dir, prefix):
    """Plot and save confusion matrices for binary and multi-class predictions."""
    # Binary confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(binary_true, binary_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=binary_classes,
                yticklabels=binary_classes)
    plt.title('Binary Classification Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{prefix}_binary_cm.png'))
    plt.close()
    
    # Multi-class confusion matrix
    is_target = binary_true == 1
    if np.sum(is_target) > 0:
        plt.figure(figsize=(16, 14))
        cm = confusion_matrix(multi_true[is_target], multi_preds[is_target])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=multi_classes,
                    yticklabels=multi_classes)
        plt.title('Multi-class Classification Confusion Matrix (Target Sequences Only)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{prefix}_multi_cm.png'))
        plt.close()


def main(args):
    """Main training function."""
    print(f"Starting training with the following arguments: {args}")
    
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processor
    processor = SensorDataProcessor(args.data_path, cache_dir=args.output_dir)
    
    # Check if features are already processed
    if os.path.exists(args.features_file):
        print(f"Loading pre-processed features from {args.features_file}")
        features_df = pd.read_csv(args.features_file)
    else:
        print("Processing sequences to extract features...")
        features_df = processor.process_sequences_batch(
            file_path='train.csv',
            output_file=args.features_file,
            max_sequences=args.max_sequences
        )
    
    # Prepare for training
    X, y_binary, y_multi = processor.prepare_for_training(
        features_df, 
        imu_only=args.imu_only
    )
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Get class names
    binary_classes = processor.binary_encoder.classes_
    multi_classes = processor.gesture_encoder.classes_
    num_gestures = len(multi_classes)
    
    # Save scaler and encoders
    joblib.dump(scaler, os.path.join(args.output_dir, 'scaler.joblib'))
    joblib.dump(processor.binary_encoder, os.path.join(args.output_dir, 'binary_encoder.joblib'))
    joblib.dump(processor.gesture_encoder, os.path.join(args.output_dir, 'gesture_encoder.joblib'))
    
    # Set up cross-validation
    if args.group_cv:
        # Group by subject to prevent data leakage
        groups = features_df['subject'].values
        cv = GroupKFold(n_splits=args.folds)
        splits = list(cv.split(X, y_binary, groups=groups))
    else:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(cv.split(X, y_binary))
    
    # Train model for each fold
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        print(f"\n--- Training Fold {fold + 1}/{args.folds} ---")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_binary_train, y_binary_val = y_binary[train_idx], y_binary[val_idx]
        y_multi_train, y_multi_val = y_multi[train_idx], y_multi[val_idx]
        
        # Create datasets
        train_dataset = SensorFeatureDataset(X_train, y_binary_train, y_multi_train)
        val_dataset = SensorFeatureDataset(X_val, y_binary_val, y_multi_val)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False
        )
        
        # Create model
        input_dim = X_train.shape[1]
        model = SimplifiedModel(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            num_gestures=num_gestures,
            dropout=args.dropout
        ).to(device)
        
        # Define optimizer and loss functions
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion_binary = nn.CrossEntropyLoss()
        criterion_multi = nn.CrossEntropyLoss()
        
        # Train model
        best_score = 0.0
        best_epoch = 0
        best_model_path = os.path.join(args.output_dir, f'model_fold{fold}.pt')
        
        for epoch in range(args.epochs):
            print(f"Epoch {epoch + 1}/{args.epochs}")
            
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, 
                criterion_binary, criterion_multi, 
                device
            )
            
            # Validate
            val_loss, score, binary_f1, multi_f1, binary_preds, multi_preds = validate(
                model, val_loader, 
                criterion_binary, criterion_multi, 
                device
            )
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Competition Score: {score:.4f} (Binary F1: {binary_f1:.4f}, Multi F1: {multi_f1:.4f})")
            
            # Save best model
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with score: {best_score:.4f}")
            
            # Early stopping
            if epoch - best_epoch >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Force garbage collection
            gc.collect()
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(best_model_path))
        val_loss, score, binary_f1, multi_f1, binary_preds, multi_preds = validate(
            model, val_loader,
            criterion_binary, criterion_multi,
            device
        )
        
        # Plot confusion matrices
        plot_confusion_matrices(
            y_binary_val, binary_preds,
            y_multi_val, multi_preds,
            binary_classes, multi_classes,
            args.output_dir, f'fold{fold}'
        )
        
        # Record fold score
        fold_scores.append({
            'fold': fold + 1,
            'score': score,
            'binary_f1': binary_f1,
            'multi_f1': multi_f1,
            'val_loss': val_loss
        })
    
    # Print and save final results
    print("\n--- Final Results ---")
    df_scores = pd.DataFrame(fold_scores)
    
    print(f"Mean Score: {df_scores['score'].mean():.4f}")
    print(f"Mean Binary F1: {df_scores['binary_f1'].mean():.4f}")
    print(f"Mean Multi F1: {df_scores['multi_f1'].mean():.4f}")
    
    # Save scores to CSV
    df_scores.to_csv(os.path.join(args.output_dir, 'fold_scores.csv'), index=False)
    
    # Save average scores to text file
    with open(os.path.join(args.output_dir, 'results.txt'), 'w') as f:
        f.write(f"Mean Score: {df_scores['score'].mean():.4f}\n")
        f.write(f"Mean Binary F1: {df_scores['binary_f1'].mean():.4f}\n")
        f.write(f"Mean Multi F1: {df_scores['multi_f1'].mean():.4f}\n")
        f.write(f"\nDetailed Fold Results:\n")
        f.write(df_scores.to_string())
    
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BFRB detection model')
    
    # Data arguments
    parser.add_argument('--data-path', type=str, default='/Users/shail/Downloads/cmi-detect-behavior-with-sensor-data.zip',
                        help='Path to the competition data zip file')
    parser.add_argument('--features-file', type=str, default='data/processed_features.csv',
                        help='Path to save/load processed features')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save models and results')
    parser.add_argument('--max-sequences', type=int, default=None,
                        help='Maximum number of sequences to process (for testing)')
    parser.add_argument('--imu-only', action='store_true',
                        help='Use only IMU features')
    
    # Model arguments
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64],
                        help='Hidden dimensions for SimplifiedModel')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--group-cv', action='store_true',
                        help='Use GroupKFold for cross-validation')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    main(args)
