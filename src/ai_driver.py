import numpy as np
import joblib
import os

class TORCSAIDriver:
    def __init__(self, model_path):
        """Initialize the AI driver with a trained model"""
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
        self.prev_speed = None
        self.prev_rpm = None
        self.control_smoothing = 0.3  # Smoothing factor (0 = no smoothing, 1 = complete smoothing)
        self.last_moving_time = 0
        self.stuck_count = 0
        self.steady_state_count = 0
        self.target_gear = 1
        
        print(f"Loaded TORCS AI driver model with {len(self.feature_columns)} input features")
        print(f"Model predicts: {', '.join(self.target_columns)}")

    def predict(self, state):
        """
        Predict control commands from car state
        
        Args:
            state: CarState object with sensor readings
            
        Returns:
            dict: Control commands (steering, accel, brake, gear, clutch)
        """
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
        
        # Add race-related attributes if they exist
        for attr, std_name in [('curLapTime', 'curLapTime'), ('lastLapTime', 'lastLapTime'), ('racePos', 'racePos')]:
            if hasattr(state, attr):
                state_dict[std_name] = getattr(state, attr)
        
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
            else:
                # Use standard name directly if no mapping exists
                feature_dict[std_name] = value
        
        # Extract features in correct order
        for feature_name in self.feature_columns:
            if feature_name in feature_dict:
                features.append(feature_dict[feature_name])
            else:
                # Check if removing underscores helps match the column
                alternative_name = feature_name.replace('_', '')
                if alternative_name in feature_dict:
                    features.append(feature_dict[alternative_name])
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
            if i < len(predictions):
                if target_col in self.reverse_mapping:
                    std_name = self.reverse_mapping[target_col]
                    control_dict[std_name] = float(predictions[i])
                else:
                    # Map common target names to standard control names
                    if 'steering' in target_col.lower():
                        control_dict['steer'] = float(predictions[i])
                    elif 'acceleration' in target_col.lower() or 'accel' in target_col.lower():
                        control_dict['accel'] = float(predictions[i])
                    elif 'braking' in target_col.lower() or 'brake' in target_col.lower():
                        control_dict['brake'] = float(predictions[i])
                    elif target_col.lower() == 'gear' or 'gear' in target_col.lower():
                        control_dict['gear'] = float(predictions[i])
                    elif 'clutch' in target_col.lower():
                        control_dict['clutch'] = float(predictions[i])
                    else:
                        # Use column name directly if no mapping available
                        control_dict[target_col.lower()] = float(predictions[i])
        
        # Ensure we have all standard controls
        if 'steer' not in control_dict:
            control_dict['steer'] = -state.angle * 0.5
        
        if 'accel' not in control_dict:
            control_dict['accel'] = 0.8
        
        if 'brake' not in control_dict:
            control_dict['brake'] = 0.0
        
        if 'gear' not in control_dict:
            control_dict['gear'] = state.gear
        
        if 'clutch' not in control_dict:
            control_dict['clutch'] = 0.0
        
        # Apply smoothing if we have a previous prediction
        if self.prev_prediction is not None:
            for key in ['steer', 'accel', 'brake', 'clutch']:
                if key in control_dict and key in self.prev_prediction:
                    control_dict[key] = (1 - self.control_smoothing) * control_dict[key] + \
                                       self.control_smoothing * self.prev_prediction[key]
        
        # Post-process controls
        
        # Ensure steering is within valid range [-1, 1]
        control_dict['steer'] = max(-1.0, min(1.0, control_dict['steer']))
        
        # Ensure acceleration and brake are within [0, 1]
        control_dict['accel'] = max(0.0, min(1.0, control_dict['accel']))
        control_dict['brake'] = max(0.0, min(1.0, control_dict['brake']))
        
        # Post-process clutch (ensure 0-1 range)
        control_dict['clutch'] = max(0.0, min(1.0, control_dict['clutch']))
        
        # COMPLETELY REVISED GEAR HANDLING LOGIC
        # Use direct RPM/speed-based gear selection instead of relying on AI prediction
        current_gear = state.gear
        speed = abs(state.speedX)  # Use absolute value to handle negative speeds
        rpm = state.rpm
        
        # Store RPM from previous iteration for comparison
        if self.prev_rpm is None:
            self.prev_rpm = rpm
        
        # Initialize target gear if needed
        if self.target_gear is None:
            self.target_gear = current_gear
        
        # Start in 1st gear when stopped
        if speed < 5 and current_gear <= 0:
            self.target_gear = 1
        else:
            # Define RPM thresholds for gear changes - more aggressive thresholds
            upshift_rpm = 6800  # Lower threshold to ensure upshifts happen
            downshift_rpm = 3500  # Higher threshold to prevent unnecessary downshifts
            
            # Detect if we're at steady RPM (not changing much)
            steady_state = abs(rpm - self.prev_rpm) < 50
            
            if steady_state:
                self.steady_state_count += 1
            else:
                self.steady_state_count = 0
            
            # If RPM is steady and high for several iterations, force an upshift
            if self.steady_state_count > 20 and rpm > 6000 and current_gear < 6:
                self.target_gear = current_gear + 1
                print(f"Forced upshift from {current_gear} to {self.target_gear} due to steady high RPM")
                self.steady_state_count = 0
            
            # Normal gear shifting based on RPM and current gear
            elif rpm > upshift_rpm and current_gear < 6:
                self.target_gear = current_gear + 1
                print(f"Upshift triggered: RPM={rpm:.0f} > threshold={upshift_rpm}")
            elif rpm < downshift_rpm and current_gear > 1:
                self.target_gear = current_gear - 1
                print(f"Downshift triggered: RPM={rpm:.0f} < threshold={downshift_rpm}")
        
        # Create a speed-to-gear reference table as a fallback
        speed_gear_map = {
            0: 1,    # 0 km/h -> 1st gear
            20: 2,   # 20 km/h -> 2nd gear
            40: 3,   # 40 km/h -> 3rd gear
            70: 4,   # 70 km/h -> 4th gear
            100: 5,  # 100 km/h -> 5th gear
            140: 6   # 140 km/h -> 6th gear
        }
        
        # Check if we're in an appropriate gear for our speed as a safety check
        target_gear_for_speed = 1
        for speed_threshold, gear in sorted(speed_gear_map.items()):
            if speed >= speed_threshold:
                target_gear_for_speed = gear
        
        # If there's a big mismatch between our target gear and what's appropriate for our speed,
        # use the speed-based gear instead
        if abs(self.target_gear - target_gear_for_speed) > 2:
            self.target_gear = target_gear_for_speed
            print(f"Speed-based gear correction: {speed:.1f} km/h → gear {self.target_gear}")
        
        # Special case for starting from standstill
        if speed < 5 and self.target_gear == 1:
            # Manage clutch for smooth starts
            control_dict['clutch'] = max(0.5, 1.0 - (rpm / 3000))
        elif self.target_gear != current_gear:
            # Apply clutch during gear changes
            control_dict['clutch'] = 0.5
        else:
            # Normal driving - reduce clutch slip
            control_dict['clutch'] = max(0.0, control_dict['clutch'] - 0.1)
        
        # Ensure gear is in valid range and set to target
        self.target_gear = max(1, min(6, self.target_gear))
        control_dict['gear'] = self.target_gear
        
        # Apply action constraints
        # Threshold braking (don't brake lightly)
        if control_dict['brake'] < 0.1:
            control_dict['brake'] = 0
            
        # Don't accelerate and brake simultaneously
        if control_dict['accel'] > 0.1 and control_dict['brake'] > 0.1:
            if control_dict['accel'] > control_dict['brake']:
                control_dict['brake'] = 0
            else:
                control_dict['accel'] = 0
        
        # Check for stuck condition and apply recovery if needed
        current_speed = abs(state.speedX)
        
        # Detect if we're stuck
        if current_speed < 3 and abs(control_dict['steer']) > 0.5:
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 1)  # Gradually decrease counter if not stuck
        
        # Apply recovery maneuver if stuck for too long
        if self.stuck_count > 100:  # Stuck for ~5 seconds (assuming 20 fps)
            print("Stuck detected! Attempting recovery maneuver.")
            
            if self.stuck_count % 40 < 20:
                # Reverse with opposite steering
                self.target_gear = -1
                control_dict['gear'] = -1
                control_dict['steer'] = -control_dict['steer']
                control_dict['accel'] = 0.8
                control_dict['brake'] = 0.0
                control_dict['clutch'] = 0.0
            else:
                # Then go forward
                self.target_gear = 1
                control_dict['gear'] = 1
                control_dict['accel'] = 0.8
                control_dict['brake'] = 0.0
                control_dict['clutch'] = 0.0
            
            # Reset stuck counter after a while
            if self.stuck_count > 200:
                self.stuck_count = 0
        
        # Save current prediction for next time
        self.prev_prediction = control_dict.copy()
        self.prev_rpm = rpm
        self.prev_speed = speed
        
        return control_dict