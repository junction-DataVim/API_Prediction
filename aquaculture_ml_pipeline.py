"""
Aquaculture IoT ML Pipeline
==========================

This module provides a comprehensive machine learning solution for:
1. Fish farming metrics prediction (weight, survival rate, disease occurrence)
2. Environmental parameter forecasting (temperature, oxygen, pH, turbidity)
3. Real-time monitoring and anomaly detection
4. IoT sensor data processing and trend analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# Time series libraries
from datetime import datetime, timedelta
import joblib
import json
import os

class AquacultureMLPipeline:
    """
    Comprehensive ML Pipeline for Aquaculture IoT Data Analysis and Prediction
    """
    
    def __init__(self, data_path='Data_Model_IoTMLCQ_2024.csv'):
        """Initialize the pipeline with data path"""
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.predictions = {}
        self.model_performance = {}
        
        # Define target columns (metrics we want to predict)
        self.target_columns = [
            'Average Fish Weight (g)',
            'Survival Rate (%)',
            'Disease Occurrence (Cases)',
            'Temperature (°C)',
            'Dissolved Oxygen (mg/L)',
            'pH',
            'Turbidity (NTU)'
        ]
        
        # Environmental features
        self.environmental_features = [
            'Temperature (°C)',
            'Dissolved Oxygen (mg/L)',
            'pH',
            'Turbidity (NTU)',
            'Average Temperature (°C)',
            'High Temperature (°C)',
            'Low Temperature (°C)',
            'Precipitation (inches)',
            'oxigeno_scaled',
            'ph',
            'turbidez'
        ]
        
        # Operational features
        self.operational_features = [
            'Oxygenation Interventions',
            'Corrective Interventions',
            'Oxygenation Automatic',
            'Corrective Measures'
        ]
        
    def load_and_preprocess_data(self):
        """Load and preprocess the IoT sensor data"""
        print("Loading and preprocessing IoT sensor data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        
        # Convert datetime column
        self.data['Datetime'] = pd.to_datetime(self.data['Datetime'])
        self.data = self.data.sort_values('Datetime').reset_index(drop=True)
        
        # Create additional time features
        self.data['Year'] = self.data['Datetime'].dt.year
        self.data['Month_Num'] = self.data['Datetime'].dt.month
        self.data['Day'] = self.data['Datetime'].dt.day
        self.data['Hour'] = self.data['Datetime'].dt.hour
        self.data['DayOfWeek'] = self.data['Datetime'].dt.dayofweek
        self.data['DayOfYear'] = self.data['Datetime'].dt.dayofyear
        self.data['WeekOfYear'] = self.data['Datetime'].dt.isocalendar().week
        self.data['Quarter'] = self.data['Datetime'].dt.quarter
        self.data['IsWeekend'] = self.data['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Handle categorical variables with proper mapping
        # Define specific mappings for binary categorical variables
        binary_mappings = {
            'Oxygenation Automatic': {'Yes': 1, 'No': 0},
            'Corrective Measures': {'Yes': 1, 'No': 0},
            'Thermal Risk Index': {'Normal': 0, 'High': 1, 'Low': -1},
            'Low Oxygen Alert': {'Safe': 0, 'Alert': 1, 'Critical': 2},
            'Health Status': {'Stable': 0, 'Unstable': 1, 'Critical': 2}
        }
        
        # Apply binary mappings
        for col, mapping in binary_mappings.items():
            if col in self.data.columns:
                # Convert to string first to handle any missing values
                self.data[col] = self.data[col].astype(str)
                # Apply mapping with default value for unknown categories
                self.data[f'{col}_encoded'] = self.data[col].map(mapping).fillna(0).astype(int)
                print(f"Mapped {col}: {self.data[col].unique()} -> {self.data[f'{col}_encoded'].unique()}")
        
        # Handle Month column with LabelEncoder
        categorical_cols = ['Month']
        le = LabelEncoder()
        
        for col in categorical_cols:
            if col in self.data.columns:
                self.data[f'{col}_encoded'] = le.fit_transform(self.data[col].astype(str))
                print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Convert any remaining string columns to numeric where possible
        for col in self.data.columns:
            if self.data[col].dtype == 'object' and col != 'Datetime':
                try:
                    # Try to convert to numeric
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    print(f"Converted {col} to numeric")
                except:
                    # If conversion fails, use label encoding
                    if col not in [c + '_encoded' for c in binary_mappings.keys()]:
                        le_temp = LabelEncoder()
                        self.data[f'{col}_encoded'] = le_temp.fit_transform(self.data[col].astype(str))
                        print(f"Label encoded {col}")
        
        # Fill any remaining NaN values with appropriate defaults
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Datetime':
                self.data[col] = self.data[col].fillna(self.data[col].median())
        
        # Ensure all target columns are properly numeric
        for target in self.target_columns:
            if target in self.data.columns:
                self.data[target] = pd.to_numeric(self.data[target], errors='coerce')
                if self.data[target].isna().any():
                    print(f"Warning: {target} contains NaN values, filling with median")
                    self.data[target] = self.data[target].fillna(self.data[target].median())
        
        # Create lag features for key metrics (more conservative)
        for target in self.target_columns:
            if target in self.data.columns:
                # Only create essential lag features to avoid too many NaN values
                for lag in [1, 6, 24]:  # 1h, 6h, 24h lags only
                    self.data[f'{target}_lag_{lag}'] = self.data[target].shift(lag)
        
        # Create rolling averages for key metrics (more conservative)
        for target in self.target_columns:
            if target in self.data.columns:
                # Only create essential rolling windows
                for window in [6, 24]:  # 6h, 24h windows only
                    self.data[f'{target}_rolling_{window}'] = self.data[target].rolling(window=window).mean()
        
        # Create interaction features
        if 'Temperature (°C)' in self.data.columns and 'Dissolved Oxygen (mg/L)' in self.data.columns:
            self.data['Temp_Oxygen_Interaction'] = self.data['Temperature (°C)'] * self.data['Dissolved Oxygen (mg/L)']
        
        if 'pH' in self.data.columns and 'Turbidity (NTU)' in self.data.columns:
            self.data['pH_Turbidity_Interaction'] = self.data['pH'] * self.data['Turbidity (NTU)']
        
        # Drop rows with too many NaN values, but keep records with minimal NaN
        # First, fill NaN values in lag and rolling features with forward fill
        lag_rolling_cols = [col for col in self.data.columns if '_lag_' in col or '_rolling_' in col]
        for col in lag_rolling_cols:
            self.data[col] = self.data[col].ffill()
        
        # Now drop rows where target columns have NaN values
        target_cols_present = [col for col in self.target_columns if col in self.data.columns]
        self.processed_data = self.data.dropna(subset=target_cols_present).reset_index(drop=True)
        
        # Validate data types
        self.validate_data_types()
        
        print(f"Data loaded: {len(self.processed_data)} records")
        print(f"Date range: {self.processed_data['Datetime'].min()} to {self.processed_data['Datetime'].max()}")
        print(f"Target columns: {self.target_columns}")
        
        return self.processed_data
    
    def create_features(self, target_column):
        """Create feature matrix for ML models"""
        if target_column not in self.processed_data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Base time features
        feature_cols = [
            'Year', 'Month_Num', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear', 
            'WeekOfYear', 'Quarter', 'IsWeekend'
        ]
        
        # Add environmental features (excluding the target if it's environmental)
        env_features = [col for col in self.environmental_features 
                       if col in self.processed_data.columns and col != target_column]
        feature_cols.extend(env_features)
        
        # Add operational features
        op_features = [col for col in self.operational_features 
                      if col in self.processed_data.columns]
        feature_cols.extend(op_features)
        
        # Add encoded categorical features
        categorical_encoded = [col for col in self.processed_data.columns 
                             if col.endswith('_encoded')]
        feature_cols.extend(categorical_encoded)
        
        # Add specific encoded features if they exist
        encoded_features = [
            'Oxygenation Automatic_encoded',
            'Corrective Measures_encoded', 
            'Thermal Risk Index_encoded',
            'Low Oxygen Alert_encoded',
            'Health Status_encoded',
            'Month_encoded'
        ]
        
        for col in encoded_features:
            if col in self.processed_data.columns and col not in feature_cols:
                feature_cols.append(col)
        
        # Add lag features
        lag_cols = [col for col in self.processed_data.columns 
                   if '_lag_' in col and not col.startswith(target_column)]
        feature_cols.extend(lag_cols)
        
        # Add rolling features
        rolling_cols = [col for col in self.processed_data.columns 
                       if '_rolling_' in col and not col.startswith(target_column)]
        feature_cols.extend(rolling_cols)
        
        # Add interaction features
        interaction_cols = [col for col in self.processed_data.columns 
                           if '_Interaction' in col]
        feature_cols.extend(interaction_cols)
        
        # Add other target columns as features (for cross-prediction)
        other_targets = [col for col in self.target_columns 
                        if col != target_column and col in self.processed_data.columns]
        feature_cols.extend(other_targets)
        
        # Remove duplicates and ensure all columns exist
        feature_cols = list(set(feature_cols))
        feature_cols = [col for col in feature_cols if col in self.processed_data.columns]
        
        # Ensure all feature columns are numeric
        X = self.processed_data[feature_cols].copy()
        
        # Convert any remaining non-numeric columns and handle NaN values
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    # If conversion fails, use label encoding
                    le_temp = LabelEncoder()
                    X[col] = le_temp.fit_transform(X[col].astype(str))
            
            # Fill NaN values with median for numeric columns
            if X[col].dtype in ['float64', 'int64', 'int32', 'float32']:
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median())
            else:
                # For non-numeric columns, fill with mode or 0
                if X[col].isna().any():
                    mode_val = X[col].mode()
                    if len(mode_val) > 0:
                        X[col] = X[col].fillna(mode_val.iloc[0])
                    else:
                        X[col] = X[col].fillna(0)
        
        # Ensure target is numeric and has no NaN values
        y = pd.to_numeric(self.processed_data[target_column], errors='coerce')
        if y.isna().any():
            print(f"Warning: Target {target_column} has NaN values, filling with median")
            y = y.fillna(y.median())
        
        # Final check for NaN values
        if X.isna().any().any():
            print(f"Warning: Still found NaN values in features, filling with 0")
            X = X.fillna(0)
        
        if y.isna().any():
            print(f"Warning: Still found NaN values in target, filling with median")
            y = y.fillna(y.median())
        
        print(f"Features for {target_column}: {len(feature_cols)} features")
        print(f"Feature data types: {X.dtypes.value_counts().to_dict()}")
        print(f"Features shape: {X.shape}, Target shape: {y.shape}")
        
        return X, y, feature_cols
    
    def train_models(self, target_column, test_size=0.2):
        """Train multiple ML models for a specific target"""
        print(f"Training models for {target_column}...")
        
        X, y, feature_cols = self.create_features(target_column)
        
        # Check if we have enough data
        if len(X) < 10:
            print(f"Warning: Not enough data for {target_column} (only {len(X)} samples)")
            return None
        
        # Time series split to maintain temporal order
        if len(X) >= 50:  # Use time series split for larger datasets
            tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 10))
            split_idx = list(tscv.split(X))[-1]  # Use the last split
            train_idx, test_idx = split_idx
        else:
            # For smaller datasets, use a simple split
            split_point = int(len(X) * 0.8)
            train_idx = list(range(split_point))
            test_idx = list(range(split_point, len(X)))
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_column] = scaler
        
        # Initialize models with better parameters
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'XGBoost': xgb.XGBRegressor(random_state=42, max_depth=6, n_estimators=100),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1, max_depth=6, n_estimators=100),
            'GradientBoosting': GradientBoostingRegressor(random_state=42, max_depth=6, n_estimators=100),
            'Ridge': Ridge(alpha=1.0),
            'ElasticNet': ElasticNet(alpha=1.0, random_state=42, max_iter=2000)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            if name in ['Ridge', 'ElasticNet']:
                # Use scaled features for linear models
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                # Use original features for tree-based models
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'predictions': y_pred,
                'feature_cols': feature_cols
            }
            
            print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
        
        self.models[target_column] = results
        self.model_performance[target_column] = {
            name: {'mae': res['mae'], 'rmse': res['rmse'], 'r2': res['r2']} 
            for name, res in results.items()
        }
        
        # Store test data for visualization
        self.test_data = {
            'y_test': y_test,
            'X_test': X_test,
            'dates': self.processed_data.iloc[test_idx]['Datetime']
        }
        
        return results
    
    def train_all_models(self):
        """Train models for all target columns"""
        print("Training models for all target columns...")
        
        self.best_models = {}
        
        for target in self.target_columns:
            if target in self.processed_data.columns:
                print(f"\n=== Training models for {target} ===")
                self.train_models(target)
                
                # Find best model for this target
                best_model_name = max(
                    self.model_performance[target].keys(),
                    key=lambda k: self.model_performance[target][k]['r2']
                )
                
                self.best_models[target] = {
                    'name': best_model_name,
                    'performance': self.model_performance[target][best_model_name]
                }
                
                print(f"Best model for {target}: {best_model_name} (R² = {self.model_performance[target][best_model_name]['r2']:.4f})")
        
        return self.best_models
    
    def predict_single_metric(self, target_column, hours_ahead=24):
        """Generate predictions for a single metric"""
        if target_column not in self.models:
            raise ValueError(f"No trained model found for {target_column}")
        
        # Get the best model
        best_model_name = self.best_models[target_column]['name']
        best_model = self.models[target_column][best_model_name]['model']
        feature_cols = self.models[target_column][best_model_name]['feature_cols']
        
        # Get recent data for prediction
        recent_data = self.processed_data.tail(hours_ahead * 2).copy()
        
        # Create predictions
        predictions = []
        current_data = self.processed_data.copy()
        
        # Get last timestamp
        last_datetime = current_data['Datetime'].max()
        
        for i in range(hours_ahead):
            # Create next timestamp
            next_datetime = last_datetime + timedelta(hours=i+1)
            
            # Create new row with time features
            new_row = pd.Series({
                'Datetime': next_datetime,
                'Year': next_datetime.year,
                'Month_Num': next_datetime.month,
                'Day': next_datetime.day,
                'Hour': next_datetime.hour,
                'DayOfWeek': next_datetime.dayofweek,
                'DayOfYear': next_datetime.dayofyear,
                'WeekOfYear': next_datetime.isocalendar().week,
                'Quarter': next_datetime.quarter,
                'IsWeekend': int(next_datetime.weekday() in [5, 6])
            })
            
            # Add lag features using most recent data
            for lag in [1, 3, 6, 12, 24]:
                if len(current_data) >= lag:
                    new_row[f'{target_column}_lag_{lag}'] = current_data[target_column].iloc[-lag]
            
            # Add rolling features
            for window in [3, 6, 12, 24, 48]:
                if len(current_data) >= window:
                    new_row[f'{target_column}_rolling_{window}'] = current_data[target_column].tail(window).mean()
            
            # Add other features (use latest values or means)
            for col in feature_cols:
                if col not in new_row and col in current_data.columns:
                    if col in self.target_columns:
                        new_row[col] = current_data[col].iloc[-1]
                    else:
                        new_row[col] = current_data[col].mean()
            
            # Prepare features for prediction
            feature_values = []
            for col in feature_cols:
                if col in new_row:
                    feature_values.append(new_row[col])
                else:
                    feature_values.append(0)  # Default value
            
            X_pred = np.array(feature_values).reshape(1, -1)
            
            # Make prediction
            if best_model_name in ['Ridge', 'ElasticNet']:
                scaler = self.scalers[target_column]
                X_pred_scaled = scaler.transform(X_pred)
                prediction = best_model.predict(X_pred_scaled)[0]
            else:
                prediction = best_model.predict(X_pred)[0]
            
            predictions.append({
                'datetime': next_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'predicted_value': round(float(prediction), 4),
                'target': target_column,
                'model_used': best_model_name
            })
            
            # Add prediction to current data for next iteration
            new_row[target_column] = prediction
            current_data = pd.concat([current_data, new_row.to_frame().T], ignore_index=True)
        
        return predictions
    
    def detect_anomalies(self, target_column, threshold_std=2):
        """Detect anomalies in the data"""
        if target_column not in self.processed_data.columns:
            return []
        
        data = self.processed_data[target_column]
        mean_val = data.mean()
        std_val = data.std()
        
        # Find anomalies
        anomalies = []
        for idx, value in enumerate(data):
            if abs(value - mean_val) > threshold_std * std_val:
                anomalies.append({
                    'datetime': self.processed_data.iloc[idx]['Datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                    'value': float(value),
                    'mean': float(mean_val),
                    'std': float(std_val),
                    'deviation': float(abs(value - mean_val) / std_val),
                    'target': target_column
                })
        
        return anomalies
    
    def save_all_models(self):
        """Save all trained models"""
        print("Saving all trained models...")
        
        os.makedirs('models', exist_ok=True)
        
        for target in self.target_columns:
            if target in self.models and target in self.best_models:
                print(f"Saving models for {target}...")
                
                # Clean target name for filename
                clean_target = target.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('/', '_')
                
                # Save the complete model data
                model_data = {
                    'models': self.models[target],
                    'scaler': self.scalers[target],
                    'best_model_info': self.best_models[target],
                    'model_performance': self.model_performance[target],
                    'target': target,
                    'saved_at': datetime.now().isoformat()
                }
                
                # Save to file
                joblib.dump(model_data, f'models/best_model_{clean_target}.joblib')
                
                # Also save the best model separately
                best_model_name = self.best_models[target]['name']
                best_model = self.models[target][best_model_name]['model']
                joblib.dump(best_model, f'models/{clean_target}_best.joblib')
        
        print("All models saved to 'models/' directory")
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        report = f"""
        Aquaculture IoT ML Pipeline Report
        =================================
        
        Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Data Overview:
        - Total Records: {len(self.processed_data)}
        - Date Range: {self.processed_data['Datetime'].min()} to {self.processed_data['Datetime'].max()}
        - Targets Analyzed: {len(self.target_columns)}
        
        Target Metrics:
        """
        
        for target in self.target_columns:
            if target in self.model_performance:
                report += f"\n        {target}:"
                best_model = self.best_models[target]['name']
                performance = self.best_models[target]['performance']
                report += f"""
          - Best Model: {best_model}
          - MAE: {performance['mae']:.4f}
          - RMSE: {performance['rmse']:.4f}
          - R²: {performance['r2']:.4f}"""
        
        # Save report
        with open('aquaculture_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print("Analysis report generated!")
        print(report)
        
        return report
    
    def validate_data_types(self):
        """Validate that all data types are properly converted"""
        print("Validating data types...")
        
        # Check for any remaining object columns (strings)
        object_cols = self.processed_data.select_dtypes(include=['object']).columns
        object_cols = [col for col in object_cols if col != 'Datetime']
        
        if len(object_cols) > 0:
            print(f"Warning: Found object columns that may need conversion: {object_cols}")
            
            # Try to convert these columns
            for col in object_cols:
                unique_vals = self.processed_data[col].unique()
                print(f"Column {col} unique values: {unique_vals}")
                
                # If it's a simple binary column, convert it
                if len(unique_vals) <= 2:
                    try:
                        # Create a simple binary mapping
                        mapping = {val: i for i, val in enumerate(unique_vals)}
                        self.processed_data[f'{col}_encoded'] = self.processed_data[col].map(mapping)
                        print(f"Created binary mapping for {col}: {mapping}")
                    except Exception as e:
                        print(f"Failed to create binary mapping for {col}: {e}")
        
        # Check for any columns with NaN values
        nan_cols = self.processed_data.columns[self.processed_data.isna().any()].tolist()
        if len(nan_cols) > 0:
            print(f"Warning: Found columns with NaN values: {nan_cols}")
            
            # Fill NaN values
            for col in nan_cols:
                if col != 'Datetime':
                    if self.processed_data[col].dtype in ['int64', 'float64']:
                        self.processed_data[col] = self.processed_data[col].fillna(self.processed_data[col].median())
                    else:
                        self.processed_data[col] = self.processed_data[col].fillna('Unknown')
        
        # Print data type summary
        print("\nData type summary:")
        print(self.processed_data.dtypes.value_counts())
        
        # Check target columns specifically
        print("\nTarget columns data types:")
        for target in self.target_columns:
            if target in self.processed_data.columns:
                print(f"{target}: {self.processed_data[target].dtype}")
                if self.processed_data[target].dtype == 'object':
                    print(f"  Warning: {target} is still object type!")
                    print(f"  Sample values: {self.processed_data[target].head()}")
        
        print("Data validation complete!")
        return True

def main():
    """Main execution function"""
    print("=== Aquaculture IoT ML Pipeline Demo ===")
    
    # Initialize pipeline
    pipeline = AquacultureMLPipeline('Data_Model_IoTMLCQ_2024.csv')
    
    # Load and preprocess data
    data = pipeline.load_and_preprocess_data()
    
    # Train models for all targets
    print("\n=== Training Models for All Targets ===")
    pipeline.train_all_models()
    
    # Save models
    print("\n=== Saving Models ===")
    pipeline.save_all_models()
    
    # Generate report
    print("\n=== Generating Report ===")
    pipeline.generate_report()
    
    # Example predictions
    print("\n=== Example Predictions ===")
    try:
        predictions = pipeline.predict_single_metric('Average Fish Weight (g)', hours_ahead=24)
        print(f"Generated {len(predictions)} predictions for Average Fish Weight")
    except Exception as e:
        print(f"Error generating predictions: {e}")
    
    print("\n=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
