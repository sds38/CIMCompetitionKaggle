import os
import argparse
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
import zipfile
import gc
import json

from preprocessing import SensorDataProcessor
from models import SimplifiedModel


class SequencePredictor:
    """
    Memory-efficient predictor for BFRB sequences that follows the Kaggle API requirements.
    """
    
    def __init__(self, model_dir, zip_path, device='cpu'):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing trained models
            zip_path: Path to competition zip file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_dir = model_dir
        self.zip_path = zip_path
        self.device = device
        
        # Load the processor
        processor_path = os.path.join(model_dir, 'processor.joblib')
        if os.path.exists(processor_path):
            self.processor = joblib.load(processor_path)
        else:
            # Create a new processor if one doesn't exist
            self.processor = SensorDataProcessor(zip_path, cache_dir=model_dir)
        
        # Load encoders
        self.binary_encoder = joblib.load(os.path.join(model_dir, 'binary_encoder.joblib'))
        self.gesture_encoder = joblib.load(os.path.join(model_dir, 'gesture_encoder.joblib'))
        
        # Load scaler
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.joblib'))
        
        # Load models
        self.models = []
        for file in os.listdir(model_dir):
            if file.startswith('model_fold') and file.endswith('.pt'):
                # Create model
                input_dim = len(self.scaler.mean_)
                num_gestures = len(self.gesture_encoder.classes_)
                
                model = SimplifiedModel(
                    input_dim=input_dim,
                    num_gestures=num_gestures
                ).to(device)
                
                # Load weights
                model.load_state_dict(torch.load(
                    os.path.join(model_dir, file),
                    map_location=device
                ))
                
                # Set to eval mode
                model.eval()
                
                self.models.append(model)
        
        print(f"Loaded {len(self.models)} models from {model_dir}")
    
    def predict_sequence(self, sequence_id=None, sequence_df=None, file_path='test.csv'):
        """
        Make a prediction for a single sequence, following Kaggle's API requirements.
        The method can be called either with a sequence_id (which will be extracted from the data)
        or directly with a sequence_df (for the API use case).
        
        Args:
            sequence_id: ID of the sequence to predict
            sequence_df: DataFrame containing the sequence data (alternative to sequence_id)
            file_path: Path within the zip file to the test data
            
        Returns:
            predicted_gesture: The predicted gesture
        """
        # Extract the sequence if only sequence_id is provided
        if sequence_df is None and sequence_id is not None:
            sequence_df = self.processor.extract_sequence(file_path, sequence_id)
        elif sequence_df is None and sequence_id is None:
            raise ValueError("Either sequence_id or sequence_df must be provided")
        
        # Preprocess the sequence
        sequence_data = self.processor.preprocess_sequence(sequence_df)
        
        # Extract statistical features
        features = self.processor.extract_statistical_features(sequence_data)
        
        # Create a feature vector
        feature_vector = pd.DataFrame([features])
        
        # Drop non-feature columns
        feature_vector = feature_vector.drop(columns=['sequence_id', 'subject'], errors='ignore')
        
        # Scale the features
        X = self.scaler.transform(feature_vector)
        
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Initialize arrays for predictions
        binary_preds = np.zeros(2)
        multi_preds = np.zeros(len(self.gesture_encoder.classes_))
        
        # Get predictions from all models
        with torch.no_grad():
            for model in self.models:
                binary_logits, multi_logits = model(X_tensor)
                
                binary_probs = torch.softmax(binary_logits, dim=1).cpu().numpy()
                multi_probs = torch.softmax(multi_logits, dim=1).cpu().numpy()
                
                binary_preds += binary_probs[0]
                multi_preds += multi_probs[0]
        
        # Average predictions
        binary_preds /= len(self.models)
        multi_preds /= len(self.models)
        
        # Get the predicted class
        is_target = np.argmax(binary_preds) == 1  # Check if it's a target sequence
        
        if is_target:
            # If it's a target, predict the specific gesture
            gesture_id = np.argmax(multi_preds)
            predicted_gesture = self.gesture_encoder.classes_[gesture_id]
            
            # Make sure the gesture is in the training set
            if predicted_gesture not in self.gesture_encoder.classes_:
                print(f"Warning: Predicted gesture '{predicted_gesture}' not found in training set. Defaulting to most confident known gesture.")
                # Find the highest confidence for a known gesture
                known_classes_indices = [i for i, c in enumerate(self.gesture_encoder.classes_)]
                gesture_id = np.argmax(multi_preds[known_classes_indices])
                predicted_gesture = self.gesture_encoder.classes_[gesture_id]
        else:
            # If it's not a target, return "non_target" as required by Kaggle
            predicted_gesture = "non_target"
        
        return predicted_gesture
    
    def run_prediction_api(self, output_file='submission.csv'):
        """
        Run predictions for all test sequences and save to a submission file.
        This follows the Kaggle API structure where you predict one sequence at a time.
        
        Args:
            output_file: Path to save the submission file
        """
        # Get all sequence IDs from the test set
        sequence_ids = self.processor.get_sequence_ids('test.csv')
        
        # Initialize results list
        results = []
        
        # Make predictions for each sequence
        for sequence_id in tqdm(sequence_ids, desc="Predicting"):
            predicted_gesture = self.predict_sequence(sequence_id)
            
            results.append({
                'sequence_id': sequence_id,
                'gesture': predicted_gesture
            })
            
            # Force garbage collection
            gc.collect()
        
        # Create submission DataFrame
        submission_df = pd.DataFrame(results)
        
        # Save to CSV
        submission_df.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")
        
        return submission_df


def api_inference_example():
    """
    Example of how to implement the Kaggle API for inference.
    This is how the evaluation server will call your model.
    
    The Kaggle API expects your submission to expose a `predict_gesture` method
    that takes a DataFrame containing a single sequence and returns the predicted
    gesture. This follows the API pattern required for the competition.
    """
    # In a real scenario, this would be imported from the Kaggle API
    class PredictionService:
        def __init__(self, model_dir, zip_path):
            # Initialize your predictor
            self.predictor = SequencePredictor(model_dir, zip_path)
        
        def predict_gesture(self, sequence_data):
            """
            Make a prediction for a single sequence.
            
            Args:
                sequence_data: DataFrame containing the sequence data
                
            Returns:
                predicted_gesture: The predicted gesture
            """
            # This is the expected API interface by Kaggle
            # It should process one sequence at a time as provided directly by the API
            return self.predictor.predict_sequence(sequence_df=sequence_data)
    
    # Usage example
    service = PredictionService('models', '/Users/shail/Downloads/cmi-detect-behavior-with-sensor-data.zip')
    
    # In the real API, the Kaggle server would provide the sequence data
    # For testing, we'll manually load a sequence
    processor = SensorDataProcessor('/Users/shail/Downloads/cmi-detect-behavior-with-sensor-data.zip')
    test_sequence_id = processor.get_sequence_ids('test.csv')[0]
    test_sequence = processor.extract_sequence('test.csv', test_sequence_id)
    
    # Make prediction
    prediction = service.predict_gesture(test_sequence)
    print(f"Predicted gesture for sequence {test_sequence_id}: {prediction}")


def main(args):
    """Main inference function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create predictor
    predictor = SequencePredictor(args.model_dir, args.data_path, device=device)
    
    if args.api_example:
        # Run the API example
        api_inference_example()
    else:
        # Run predictions for all test sequences
        predictor.run_prediction_api(args.output_file)
    
    print("Inference complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference for BFRB detection')
    
    parser.add_argument('--data-path', type=str, default='/Users/shail/Downloads/cmi-detect-behavior-with-sensor-data.zip',
                        help='Path to the competition data zip file')
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--output-file', type=str, default='submission/submission.csv',
                        help='Path to save the submission file')
    parser.add_argument('--api-example', action='store_true',
                        help='Run the API example instead of full inference')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    main(args)
