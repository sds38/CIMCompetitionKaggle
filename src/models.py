import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Union, Optional


class MultiSensorModel(nn.Module):
    """
    Multi-input neural network for BFRB detection that can handle both IMU-only and full sensor data.
    
    Architecture:
    - IMU Branch: Uses 1D CNN + LSTM for time series data from accelerometer and rotation sensors
    - Thermopile Branch: MLP for temperature readings
    - ToF Branch: 2D CNN for each 8x8 frame followed by temporal fusion
    """
    def __init__(
        self,
        imu_input_dim: int = 7,
        thermo_input_dim: int = 5,
        tof_input_dim: int = 5,  # 5 sensors, each with 8x8 grid
        hidden_dim: int = 128,
        num_gestures: int = 18,  # Number of gesture classes
        dropout: float = 0.3,
        seq_length: int = 50  # Typical sequence length
    ):
        super().__init__()
        
        # Define dimensions
        self.hidden_dim = hidden_dim
        self.imu_input_dim = imu_input_dim
        self.thermo_input_dim = thermo_input_dim
        self.tof_input_dim = tof_input_dim
        self.seq_length = seq_length
        
        # --- IMU Branch ---
        # 1D CNN for feature extraction
        self.imu_conv = nn.Sequential(
            nn.Conv1d(imu_input_dim, hidden_dim//2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Conv1d(hidden_dim//2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            nn.Dropout(dropout)
        )
        
        # LSTM for temporal dynamics
        self.imu_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # --- Thermopile Branch ---
        # Smaller network for temperature sensors
        self.thermo_fc = nn.Sequential(
            nn.Linear(thermo_input_dim * seq_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # --- ToF Branch ---
        # CNN for each ToF sensor (treating each as 8x8 image grid)
        self.tof_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal fusion for ToF
        self.tof_temporal_conv = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=(5, 3, 3),  # (time, height, width)
            padding=(2, 1, 1)
        )
        
        # Determine output size after convolutions
        tof_output_size = 32 * 2 * 2 * tof_input_dim  # channels * height * width * num_sensors
        
        # --- Fusion and Classification Layers ---
        # Define dimensions for each branch
        imu_output_dim = hidden_dim * 2  # Bidirectional LSTM
        thermo_output_dim = hidden_dim // 2
        tof_output_dim = hidden_dim
        
        # Fusion layer to combine different modalities
        self.fusion = nn.Sequential(
            nn.Linear(imu_output_dim + thermo_output_dim + tof_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Parallel classification heads
        self.binary_classifier = nn.Linear(hidden_dim // 2, 2)  # Binary: Target vs. Non-Target
        self.gesture_classifier = nn.Linear(hidden_dim // 2, num_gestures)  # Multi-class
        
        # Special projection for ToF data
        self.tof_projection = nn.Linear(tof_output_size, tof_output_dim)
        
        # Flag to track if we're in IMU-only mode
        self.imu_only_mode = False
    
    def forward(
        self,
        imu_data: torch.Tensor,
        thermo_data: Optional[torch.Tensor] = None,
        tof_data: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the multi-sensor network.
        
        Args:
            imu_data: Tensor of shape [batch_size, seq_length, imu_features]
            thermo_data: Tensor of shape [batch_size, seq_length, thermo_features] or None
            tof_data: Tensor of shape [batch_size, seq_length, tof_sensors, 8, 8] or None
            
        Returns:
            binary_logits: Binary classification logits (Target vs Non-target)
            gesture_logits: Multi-class classification logits for specific gestures
        """
        batch_size = imu_data.shape[0]
        
        # Set IMU-only mode if thermopile and ToF data are not provided
        self.imu_only_mode = (thermo_data is None or tof_data is None)
        
        # --- Process IMU Data ---
        # Reshape for Conv1d: [batch, channels, seq_length]
        imu_x = imu_data.transpose(1, 2)
        imu_x = self.imu_conv(imu_x)
        
        # Reshape for LSTM: [batch, seq_length, features]
        imu_x = imu_x.transpose(1, 2)
        _, (h_n, _) = self.imu_lstm(imu_x)
        
        # Combine the last hidden state from both directions and layers
        h_n = h_n.view(2, 2, batch_size, self.hidden_dim)  # [num_layers, num_directions, batch, hidden]
        imu_features = h_n[-1].transpose(0, 1).contiguous().view(batch_size, -1)
        
        # --- Process Other Sensors (if available) ---
        if self.imu_only_mode:
            # If IMU-only, use zeros for the other branches
            thermo_features = torch.zeros(batch_size, self.hidden_dim // 2, device=imu_data.device)
            tof_features = torch.zeros(batch_size, self.hidden_dim, device=imu_data.device)
        else:
            # --- Process Thermopile Data ---
            thermo_x = thermo_data.reshape(batch_size, -1)  # Flatten across time
            thermo_features = self.thermo_fc(thermo_x)
            
            # --- Process ToF Data ---
            # Shape: [batch, seq, sensors, h, w] -> process each frame
            batch_size, seq_len, num_sensors, height, width = tof_data.shape
            
            # Reshape to process each sensor's data through the same CNN
            # From [batch, seq, sensors, h, w] to [batch*seq*sensors, 1, h, w]
            tof_x = tof_data.reshape(-1, 1, height, width)
            tof_x = self.tof_conv(tof_x)
            
            # Reshape back and apply temporal convolution
            # From [batch*seq*sensors, channels, h, w] to [batch, 1, seq, sensors*channels, h*w]
            _, c, h, w = tof_x.shape
            tof_x = tof_x.reshape(batch_size, seq_len, num_sensors, c, h, w)
            tof_x = tof_x.permute(0, 3, 1, 2, 4, 5).contiguous()
            tof_x = tof_x.reshape(batch_size, 1, seq_len, -1)  # Combine sensor and channels
            
            # Project to lower dimension
            tof_x = tof_x.reshape(batch_size, -1)
            tof_features = self.tof_projection(tof_x)
        
        # --- Combine Features from All Branches ---
        combined = torch.cat([imu_features, thermo_features, tof_features], dim=1)
        fused_features = self.fusion(combined)
        
        # --- Generate Predictions ---
        binary_logits = self.binary_classifier(fused_features)
        gesture_logits = self.gesture_classifier(fused_features)
        
        return binary_logits, gesture_logits
    
    def reshape_tof_data(self, data: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Helper method to reshape raw data from the dataset into model inputs.
        
        Args:
            data: Dictionary containing the raw sensor data
                'imu': Array of shape [seq_length, imu_features]
                'thermo': Array of shape [seq_length, thermo_features]
                'tof': Array of shape [seq_length, tof_sensors * 64]
                
        Returns:
            imu_tensor: Tensor of shape [1, seq_length, imu_features]
            thermo_tensor: Tensor of shape [1, seq_length, thermo_features]
            tof_tensor: Tensor of shape [1, seq_length, tof_sensors, 8, 8]
        """
        # Process IMU data
        imu_tensor = torch.tensor(data['imu'], dtype=torch.float32).unsqueeze(0)
        
        # Process thermopile data
        thermo_tensor = torch.tensor(data['thermo'], dtype=torch.float32).unsqueeze(0)
        
        # Process ToF data - reshape from flat to grid
        seq_length, num_values = data['tof'].shape
        num_sensors = self.tof_input_dim
        
        # Check if we have missing ToF data (IMU-only scenario)
        if np.isnan(data['tof']).all():
            # Create a zero tensor instead
            tof_tensor = torch.zeros(1, seq_length, num_sensors, 8, 8, dtype=torch.float32)
        else:
            # Reshape flat ToF data into 8x8 grids for each sensor
            tof_reshaped = np.zeros((seq_length, num_sensors, 8, 8), dtype=np.float32)
            
            for t in range(seq_length):
                for sensor in range(num_sensors):
                    sensor_data = data['tof'][t, sensor*64:(sensor+1)*64]
                    # Replace -1 values (no reflection) with 0
                    sensor_data[sensor_data == -1] = 0
                    tof_reshaped[t, sensor] = sensor_data.reshape(8, 8)
            
            tof_tensor = torch.tensor(tof_reshaped, dtype=torch.float32).unsqueeze(0)
        
        return imu_tensor, thermo_tensor, tof_tensor


class SimplifiedModel(nn.Module):
    """
    A simplified model for faster development and testing.
    This model uses statistical features instead of raw time series.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        num_gestures: int = 18,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Create the hidden layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        self.binary_classifier = nn.Linear(hidden_dims[-1], 2)
        self.gesture_classifier = nn.Linear(hidden_dims[-1], num_gestures)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor with statistical features
            
        Returns:
            binary_logits: Target/Non-target logits
            gesture_logits: Gesture classification logits
        """
        features = self.feature_extractor(x)
        binary_logits = self.binary_classifier(features)
        gesture_logits = self.gesture_classifier(features)
        
        return binary_logits, gesture_logits
