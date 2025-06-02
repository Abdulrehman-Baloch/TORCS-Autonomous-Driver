import numpy as np
import joblib
import os

class TORCSAIDriver:
    def __init__(self, model_path='Dirt2_ToyotaCorolla_model'):
        # Load the model
        self.model = joblib.load(f'{model_path}/best_model.pkl')
        
        # Load the scaler
        self.scaler = joblib.load(f'{model_path}/scaler.pkl')
        
        # Load feature column names
        with open(f'{model_path}/feature_columns.txt', 'r') as f:
            self.feature_columns = f.read().splitlines()
        
        # Load target column names
        with open(f'{model_path}/target_columns.txt', 'r') as f:
            self.target_columns = f.read().splitlines()
        
        # Load column mapping to map between standardized names and dataset column names
        self.column_mapping = {}
        try:
            with open(f'{model_path}/column_mapping.txt', 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        self.column_mapping[key] = value
        except:
            print("Column mapping file not found, using direct column names")
        
        # Create reverse mapping (dataset column to standard name)
        self.reverse_mapping = {v: k for k, v in self.column_mapping.items()}
        
        # Initialize state tracking for smoother control
        self.prev_prediction = None
        self.control_smoothing = 0.3  # Smoothing factor (0 = no smoothing, 1 = complete smoothing)
        
        print(f"Loaded TORCS AI driver model with {len(self.feature_columns)} input features")
        print(f"Model predicts: {', '.join(self.target_columns)}")

    def predict(self, state):
        # Extract features from state in the correct order
        features = []
        
        # Create a dictionary to hold feature values
        feature_dict = {}
        
        # Map state attributes to standardized names first
        state_dict = {
            'angle': state.angle,
            'speedX': state.speedX,
            'speedY': state.speedY,
            'speedZ': state.speedZ,
            'rpm': state.rpm,
            'trackPos': state.trackPos,
            'gear': state.gear,
            'distFromStart': state.distFromStart
        }
        
        # Add optional attributes if they exist
        if hasattr(state, 'damage'):
            state_dict['damage'] = state.damage
        
        if hasattr(state, 'fuel'):
            state_dict['fuel'] = state.fuel
        
        # Add track sensors
        if state.track:
            for i, track_val in enumerate(state.track):
                state_dict[f'track_{i}'] = track_val
        
        # Add wheel spin velocities
        if state.wheelSpinVel:
            for i, wheel_val in enumerate(state.wheelSpinVel):
                state_dict[f'wheelSpin{i}'] = wheel_val
        
        # Map from standardized names to dataset column names using the mapping
        for std_name, value in state_dict.items():
            if std_name in self.column_mapping:
                dataset_col = self.column_mapping[std_name]
                feature_dict[dataset_col] = value
        
        # Extract features in correct order
        for feature_name in self.feature_columns:
            if feature_name in feature_dict:
                features.append(feature_dict[feature_name])
            else:
                # If feature not available, use 0
                features.append(0)
        
        # Reshape features for the model
        features = np.array(features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict control values
        predictions = self.model.predict(features_scaled)[0]
        
        # Create control dictionary mapping from dataset column names to standardized names
        control_dict = {}
        for i, target_col in enumerate(self.target_columns):
            if target_col in self.reverse_mapping:
                std_name = self.reverse_mapping[target_col]
                control_dict[std_name] = float(predictions[i])
            else:
                # Use column name directly if not in mapping
                control_dict[target_col.lower()] = float(predictions[i])
        
        # Apply smoothing if we have a previous prediction
        if self.prev_prediction is not None:
            for key in ['steer', 'accel', 'brake', 'clutch']:
                if key in control_dict and key in self.prev_prediction:
                    control_dict[key] = (1 - self.control_smoothing) * control_dict[key] +                                        self.control_smoothing * self.prev_prediction[key]
        
        # Post-process controls
        
        # Ensure steering is within valid range [-1, 1]
        if 'steer' in control_dict:
            control_dict['steer'] = max(-1.0, min(1.0, control_dict['steer']))
        
        # Ensure acceleration and brake are within [0, 1]
        if 'accel' in control_dict:
            control_dict['accel'] = max(0.0, min(1.0, control_dict['accel']))
        
        if 'brake' in control_dict:
            control_dict['brake'] = max(0.0, min(1.0, control_dict['brake']))
        
        # Post-process gear (should be an integer)
        if 'gear' in control_dict:
            # Round gear to nearest integer and ensure valid range
            gear_value = round(control_dict['gear'])
            control_dict['gear'] = max(0, min(6, gear_value))
        
        # Post-process clutch if it exists (ensure 0-1 range)
        if 'clutch' in control_dict:
            control_dict['clutch'] = max(0.0, min(1.0, control_dict['clutch']))
        
        # Don't accelerate and brake simultaneously
        if 'accel' in control_dict and 'brake' in control_dict:
            if control_dict['accel'] > 0.1 and control_dict['brake'] > 0.1:
                if control_dict['accel'] > control_dict['brake']:
                    control_dict['brake'] = 0
                else:
                    control_dict['accel'] = 0
        
        # Save current prediction for next time
        self.prev_prediction = control_dict.copy()
        
        return control_dict
