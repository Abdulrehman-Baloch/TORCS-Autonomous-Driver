import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import time

# Set random seed for reproducibility
np.random.seed(42)

class TorcsAIDriver:
    def __init__(self, data_path, model_save_path='Dirt2_ToyotaCorolla_model'):
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.X = None
        self.y = None
        self.model = None
        self.scaler = None
        
        # Create directories if they don't exist
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    
    def load_and_preprocess_data(self):
        print("Loading data from:", self.data_path)
        
        # Try to read the CSV file
        try:
            df = pd.read_csv(self.data_path)
            print(f"Successfully loaded data using standard CSV parser.")
        except Exception as e:
            print(f"Error reading CSV with standard parser: {e}")
            print("Trying alternative parsers...")
            
            # Try different delimiters
            for delimiter in [',', ';', '\t', '|']:
                try:
                    df = pd.read_csv(self.data_path, delimiter=delimiter)
                    print(f"Successfully loaded data using delimiter: '{delimiter}'")
                    break
                except:
                    continue
            else:
                raise ValueError("Could not parse the CSV file with any common delimiter. Please check the file format.")
        
        print(f"Dataset shape: {df.shape}")
        
        # Display some statistics about the dataset
        print("\nData summary:")
        print(df.describe().transpose())
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        print(f"\nMissing values in dataset: {missing_values}")
        
        # Print column names
        print("\nColumns in the dataset:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}: {col}")
        
        # Define input features based on your data format
        # We know the expected column names from the user's input
        
        # Core features essential for driving
        core_cols = [
            'Angle', 'RPM', 'SpeedX', 'SpeedY', 'SpeedZ',
            'TrackPosition', 'Gear', 'DistanceFromStart', 'DistanceCovered'
        ]
        
        # Add optional columns if they exist
        optional_cols = ['Damage', 'FuelLevel', 'CurrentLapTime', 'LastLapTime', 'RacePosition']
        for col in optional_cols:
            if col in df.columns:
                core_cols.append(col)
        
        # Track sensors
        track_cols = [col for col in df.columns if col.startswith('Track_')]
        
        # Wheel spin velocity columns
        wheel_cols = [col for col in df.columns if col.startswith('WheelSpinVelocity_')]
        
        # Ensure all columns exist in the dataframe
        feature_cols = []
        for col_list in [core_cols, track_cols, wheel_cols]:
            feature_cols.extend([col for col in col_list if col in df.columns])
        
        # Target variables - what the AI needs to predict/control
        target_cols = ['Steering', 'Acceleration', 'Braking', 'Gear', 'Clutch']
        
        # Check if target columns exist
        missing_targets = [col for col in target_cols if col not in df.columns]
        existing_targets = [col for col in target_cols if col in df.columns]
        
        if missing_targets:
            print(f"Warning: The following target columns are missing: {missing_targets}")
            print(f"Available target columns: {existing_targets}")
            
            # Create any missing target columns based on existing data
            if 'Steering' not in df.columns and 'Angle' in df.columns:
                print("Creating 'Steering' column based on 'Angle'")
                df['Steering'] = -df['Angle'] * 0.5  # Simple steering based on angle
                existing_targets.append('Steering')
            
            if 'Acceleration' not in df.columns:
                print("Creating 'Acceleration' column with default values")
                df['Acceleration'] = 0.8  # Default acceleration
                existing_targets.append('Acceleration')
            
            if 'Braking' not in df.columns:
                print("Creating 'Braking' column with default values")
                df['Braking'] = 0.0  # Default no braking
                existing_targets.append('Braking')
                
            if 'Clutch' not in df.columns:
                print("Creating 'Clutch' column with default values")
                df['Clutch'] = 0.0  # Default no clutch
                existing_targets.append('Clutch')
            
            # Use existing Gear column as both input and output if available
            if 'Gear' not in existing_targets and 'Gear' in df.columns:
                # We'll use the Gear column as both an input and a target
                print("Using existing 'Gear' column as a target")
                existing_targets.append('Gear')
        
        # Update target_cols to include only available targets
        target_cols = existing_targets
        
        print(f"\nSelected input features ({len(feature_cols)}):")
        print(feature_cols)
        
        print(f"\nSelected target features ({len(target_cols)}):")
        print(target_cols)
        
        # Get inputs and outputs
        X = df[feature_cols].values
        y = df[target_cols].values
        
        print(f"\nInput shape: {X.shape}")
        print(f"Output shape: {y.shape}")
        
        # Handle NaN values
        X = np.nan_to_num(X)
        y = np.nan_to_num(y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Save scaler for future use
        joblib.dump(self.scaler, os.path.join(self.model_save_path, 'scaler.pkl'))
        
        # Store feature column names for reference
        with open(os.path.join(self.model_save_path, 'feature_columns.txt'), 'w') as f:
            for col in feature_cols:
                f.write(f"{col}\n")
        
        # Store target column names
        with open(os.path.join(self.model_save_path, 'target_columns.txt'), 'w') as f:
            for col in target_cols:
                f.write(f"{col}\n")
        
        # Create a mapping between standardized names and dataset column names
        std_to_dataset = {
            'angle': 'Angle',
            'speedX': 'SpeedX',
            'speedY': 'SpeedY',
            'speedZ': 'SpeedZ',
            'rpm': 'RPM',
            'trackPos': 'TrackPosition',
            'gear': 'Gear',
            'distFromStart': 'DistanceFromStart',
            'distCovered': 'DistanceCovered',
            'damage': 'Damage',
            'fuel': 'FuelLevel',
            'curLapTime': 'CurrentLapTime',
            'lastLapTime': 'LastLapTime',
            'racePos': 'RacePosition',
            'steer': 'Steering',
            'accel': 'Acceleration',
            'brake': 'Braking',
            'clutch': 'Clutch'
        }
        
        # Add track and wheel mappings
        for i in range(19):
            std_to_dataset[f'track_{i}'] = f'Track_{i+1}'
        
        for i in range(4):
            std_to_dataset[f'wheelSpin{i}'] = f'WheelSpinVelocity_{i+1}'
        
        # Save the mapping
        with open(os.path.join(self.model_save_path, 'column_mapping.txt'), 'w') as f:
            for std_name, dataset_name in std_to_dataset.items():
                f.write(f"{std_name}={dataset_name}\n")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        self.X = {'train': X_train, 'test': X_test}
        self.y = {'train': y_train, 'test': y_test}
        
        self.input_dim = X_train.shape[1]
        self.output_dim = y_train.shape[1]
        
        print(f"Input dimensions: {self.input_dim}")
        print(f"Output dimensions: {self.output_dim}")
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Testing samples: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self):
        """Creating an improved neural network model for the TORCS driver using scikit-learn"""
        # MLP Regressor - A neural network model from scikit-learn
        # Improved architecture with more neurons and better hyperparameters
        self.model = MLPRegressor(
            hidden_layer_sizes=(512, 256, 128),  # Three hidden layers with more neurons
            activation='relu',                   # ReLU activation function
            solver='adam',                       # Adam optimizer
            alpha=0.0001,                        # L2 regularization
            batch_size=128,                      # Batch size
            learning_rate='adaptive',            # Adaptive learning rate
            learning_rate_init=0.001,            # Initial learning rate
            max_iter=300,                        # Maximum iterations
            tol=1e-4,                            # Tolerance
            early_stopping=True,                 # Enable early stopping
            validation_fraction=0.1,             # Validation fraction
            n_iter_no_change=15,                 # Iterations without improvement for early stopping
            random_state=42,                     # Random seed
            verbose=True                         # Print progress
        )
        
        print("Model created successfully!")
        return self.model
    
    def train_model(self):
        """Train the neural network model"""
        if self.model is None:
            self.build_model()
        
        print("Starting model training...")
        start_time = time.time()
        
        # Train the model
        self.model.fit(self.X['train'], self.y['train'])
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Create model directory if it doesn't exist
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
            
        # Save the model
        joblib.dump(self.model, os.path.join(self.model_save_path, 'best_model.pkl'))
        
        # Calculate training and validation loss
        train_score = self.model.score(self.X['train'], self.y['train'])
        test_score = self.model.score(self.X['test'], self.y['test'])
        
        print(f"Training R² score: {train_score:.4f}")
        print(f"Testing R² score: {test_score:.4f}")
        
        # Plot loss curve if available
        if hasattr(self.model, 'loss_curve_'):
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(self.model.loss_curve_)
            plt.title('Model Loss Curve')
            plt.ylabel('Loss')
            plt.xlabel('Iteration')
            plt.grid(True)
            
            # Plot validation scores if available
            if hasattr(self.model, 'validation_scores_'):
                plt.subplot(1, 2, 2)
                plt.plot(self.model.validation_scores_)
                plt.title('Validation Scores')
                plt.ylabel('Score')
                plt.xlabel('Iteration')
                plt.grid(True)
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_save_path, 'training_history.png'))
            
        return self.model
    
    def evaluate_model(self):
        """Evaluate the model on test data with detailed metrics"""
        if self.model is None:
            print("No model to evaluate. Train the model first.")
            return
        
        # Make predictions
        y_pred = self.model.predict(self.X['test'])
        
        # Get target column names
        try:
            with open(os.path.join(self.model_save_path, 'target_columns.txt'), 'r') as f:
                target_cols = f.read().splitlines()
        except:
            target_cols = [f"Target_{i}" for i in range(y_pred.shape[1])]
        
        # Calculate overall MSE and R²
        mse = mean_squared_error(self.y['test'], y_pred)
        r2 = r2_score(self.y['test'], y_pred)
        
        print(f"Overall Mean Squared Error: {mse:.6f}")
        print(f"Overall R² Score: {r2:.6f}")
        
        # Calculate and print metrics for each target variable
        print("\nPerformance by target variable:")
        for i, col in enumerate(target_cols):
            col_mse = mean_squared_error(self.y['test'][:, i], y_pred[:, i])
            col_r2 = r2_score(self.y['test'][:, i], y_pred[:, i])
            print(f"{col:15s}: MSE = {col_mse:.6f}, R² = {col_r2:.6f}")
        
        # Show some example predictions
        print("\nSample Predictions vs Actual Values:")
        for i in range(5):
            print(f"Prediction {i+1}:")
            for j, col in enumerate(target_cols):
                print(f"  {col:15s}: {y_pred[i][j]:+.4f} (Actual: {self.y['test'][i][j]:+.4f})")
        
        # Visualize predictions vs actual values for each target
        try:
            plt.figure(figsize=(15, 10))
            for i, col in enumerate(target_cols):
                plt.subplot(2, 3, i+1)
                
                # Get a subset of test data for visualization
                sample_indices = np.random.choice(len(y_pred), min(500, len(y_pred)), replace=False)
                plt.scatter(
                    self.y['test'][sample_indices, i], 
                    y_pred[sample_indices, i], 
                    alpha=0.5
                )
                
                # Add perfect prediction line
                min_val = min(np.min(self.y['test'][:, i]), np.min(y_pred[:, i]))
                max_val = max(np.max(self.y['test'][:, i]), np.max(y_pred[:, i]))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                plt.title(f'{col} Predictions')
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_save_path, 'prediction_analysis.png'))
        except Exception as e:
            print(f"Error generating prediction plots: {e}")
    
    def export_for_torcs(self):
        """Export model for use with TORCS Python driver"""
        # Driver code for TORCS
        code = """import numpy as np
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
                    control_dict[key] = (1 - self.control_smoothing) * control_dict[key] + \
                                       self.control_smoothing * self.prev_prediction[key]
        
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
"""
        # Save the driver code
        with open(os.path.join(self.model_save_path, 'Dirt2_ToyotaCorolla_model.py'), 'w') as f:
            f.write(code)
        
        print(f"TORCS driver code exported to {os.path.join(self.model_save_path, 'Dirt2_ToyotaCorolla_model.py')}")

# Main execution
if __name__ == "__main__":
    # Create the driver AI
    data_file = "Dirt2_ToyotaCorolla.csv"  # Your merged data file
    driver_ai = TorcsAIDriver(data_file)
    
    # Load and preprocess data
    driver_ai.load_and_preprocess_data()
    
    # Build and train the model
    driver_ai.build_model()
    driver_ai.train_model()
    
    # Evaluate the model
    driver_ai.evaluate_model()
    
    # Export for TORCS
    driver_ai.export_for_torcs()
    
    print("\nTraining complete! Your TORCS AI driver is ready.")
    print("The model and necessary files are saved in the directory.")
    print("You can now use the TORCSAIDriver class in your TORCS setup.")