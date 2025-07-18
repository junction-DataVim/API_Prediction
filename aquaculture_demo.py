#!/usr/bin/env python3
"""
Aquaculture IoT System Demo
==========================

This script demonstrates the complete aquaculture IoT system:
1. Train ML models for all target metrics
2. Start the real-time data pipeline
3. Launch the REST API server
4. Test the system with sample data
"""

import sys
import time
import requests
import subprocess
import threading
from datetime import datetime
import json

def train_models():
    """Train ML models for all target metrics"""
    print("=" * 60)
    print("STEP 1: Training ML Models")
    print("=" * 60)
    
    try:
        from aquaculture_ml_pipeline import AquacultureMLPipeline
        
        # Initialize pipeline
        pipeline = AquacultureMLPipeline('Data_Model_IoTMLCQ_2024.csv')
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        data = pipeline.load_and_preprocess_data()
        print(f"✓ Data loaded: {len(data)} records")
        
        # Train models for all targets
        print("\nTraining models for all target metrics...")
        best_models = pipeline.train_all_models()
        
        # Save models
        print("\nSaving trained models...")
        pipeline.save_all_models()
        
        # Generate report
        print("\nGenerating analysis report...")
        pipeline.generate_report()
        
        print("\n✓ Model training completed successfully!")
        print(f"✓ Trained models for {len(best_models)} metrics")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during model training: {e}")
        return False

def start_api_server():
    """Start the API server in a separate thread"""
    print("\n" + "=" * 60)
    print("STEP 2: Starting API Server")
    print("=" * 60)
    
    try:
        # Start the API server
        print("Starting Aquaculture IoT API server...")
        subprocess.Popen([
            sys.executable, 'aquaculture_api.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for server to start
        print("Waiting for server to start...")
        time.sleep(5)
        
        # Test if server is running
        try:
            response = requests.get('http://localhost:5000/api/status', timeout=5)
            if response.status_code == 200:
                print("✓ API server started successfully!")
                print("✓ Server available at: http://localhost:5000")
                return True
            else:
                print(f"✗ Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Could not connect to server: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error starting API server: {e}")
        return False

def test_system():
    """Test the system with sample data"""
    print("\n" + "=" * 60)
    print("STEP 3: Testing System")
    print("=" * 60)
    
    base_url = 'http://localhost:5000'
    
    # Test 1: Check API status
    print("Test 1: Checking API status...")
    try:
        response = requests.get(f'{base_url}/api/status')
        if response.status_code == 200:
            status = response.json()
            print(f"✓ API Status: {status['status']}")
            print(f"✓ Models loaded: {status['models_loaded']}")
            print(f"✓ Target metrics: {len(status['target_metrics'])}")
        else:
            print(f"✗ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error checking status: {e}")
        return False
    
    # Test 2: Ingest sample data
    print("\nTest 2: Ingesting sample sensor data...")
    sample_data = {
        "avg_fish_weight": 275.5,
        "survival_rate": 94.2,
        "disease_occurrence": 1,
        "temperature": 26.8,
        "dissolved_oxygen": 7.2,
        "ph": 7.8,
        "turbidity": 3.1,
        "month": "July",
        "oxygenation_interventions": 0,
        "corrective_interventions": 0,
        "oxigeno_scaled": 7.0,
        "ph_scaled": 0.4,
        "turbidez_scaled": 0.3,
        "oxygenation_automatic": "Yes",
        "corrective_measures": "No",
        "thermal_risk_index": "Normal",
        "low_oxygen_alert": "Safe",
        "health_status": "Stable"
    }
    
    try:
        response = requests.post(f'{base_url}/api/data/ingest', json=sample_data)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Data ingested successfully")
            print(f"✓ Predictions generated: {len(result['predictions'])}")
            print(f"✓ Anomalies detected: {len(result['anomalies'])}")
            
            # Show some predictions
            if result['predictions']:
                print("\nSample predictions:")
                for pred in result['predictions'][:3]:
                    print(f"  - {pred['target']}: {pred['predicted_value']:.3f} (model: {pred['model_used']})")
        else:
            print(f"✗ Data ingestion failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error ingesting data: {e}")
        return False
    
    # Test 3: Get latest predictions
    print("\nTest 3: Getting latest predictions...")
    try:
        response = requests.get(f'{base_url}/api/predictions/latest?limit=10')
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Retrieved {result['count']} predictions")
            
            if result['predictions']:
                print("\nLatest predictions:")
                for pred in result['predictions'][:3]:
                    print(f"  - {pred['target_metric']}: {pred['predicted_value']:.3f}")
        else:
            print(f"✗ Failed to get predictions: {response.status_code}")
    except Exception as e:
        print(f"✗ Error getting predictions: {e}")
    
    # Test 4: Get predictions for specific metric
    print("\nTest 4: Getting predictions for Average Fish Weight...")
    try:
        response = requests.get(f'{base_url}/api/predictions/metric/Average Fish Weight (g)')
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Retrieved {result['count']} predictions for {result['metric']}")
        else:
            print(f"✗ Failed to get metric predictions: {response.status_code}")
    except Exception as e:
        print(f"✗ Error getting metric predictions: {e}")
    
    # Test 5: Generate forecast
    print("\nTest 5: Generating forecast...")
    try:
        response = requests.get(f'{base_url}/api/predictions/forecast/Temperature (°C)?hours=12')
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Generated {len(result['forecast'])} hour forecast for {result['metric']}")
            
            if result['forecast']:
                print("\nSample forecast:")
                for forecast in result['forecast'][:3]:
                    print(f"  - {forecast['datetime']}: {forecast['predicted_value']:.3f}")
        else:
            print(f"✗ Failed to generate forecast: {response.status_code}")
    except Exception as e:
        print(f"✗ Error generating forecast: {e}")
    
    # Test 6: Start simulation
    print("\nTest 6: Starting data simulation...")
    try:
        response = requests.post(f'{base_url}/api/simulation/start')
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Data simulation started: {result['status']}")
            
            # Wait a bit and check for new data
            print("Waiting for simulated data...")
            time.sleep(15)
            
            # Check for new predictions
            response = requests.get(f'{base_url}/api/predictions/latest?limit=5')
            if response.status_code == 200:
                result = response.json()
                print(f"✓ Found {result['count']} new predictions from simulation")
            
        else:
            print(f"✗ Failed to start simulation: {response.status_code}")
    except Exception as e:
        print(f"✗ Error starting simulation: {e}")
    
    # Test 7: Check for anomalies
    print("\nTest 7: Checking for anomalies...")
    try:
        response = requests.get(f'{base_url}/api/anomalies?hours=1')
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Found {result['count']} anomalies in last hour")
            
            if result['anomalies']:
                print("\nRecent anomalies:")
                for anomaly in result['anomalies'][:3]:
                    print(f"  - {anomaly['metric']}: {anomaly['value']:.3f} (score: {anomaly['anomaly_score']:.3f})")
        else:
            print(f"✗ Failed to get anomalies: {response.status_code}")
    except Exception as e:
        print(f"✗ Error getting anomalies: {e}")
    
    # Test 8: Health check
    print("\nTest 8: Health check...")
    try:
        response = requests.get(f'{base_url}/api/monitoring/health')
        if response.status_code == 200:
            result = response.json()
            print(f"✓ System health: {result['status']}")
            print(f"✓ Components: {result['components']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"✗ Error in health check: {e}")
    
    print("\n✓ System testing completed!")
    return True

def main():
    """Main demo function"""
    print("🐟 Aquaculture IoT System Demo")
    print("=" * 60)
    print(f"Started at: {datetime.now()}")
    print("=" * 60)
    
    # Step 1: Train models
    if not train_models():
        print("✗ Model training failed. Exiting.")
        return False
    
    # Step 2: Start API server
    if not start_api_server():
        print("✗ API server failed to start. Exiting.")
        return False
    
    # Step 3: Test system
    if not test_system():
        print("✗ System testing failed.")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("The aquaculture IoT system is now running!")
    print()
    print("📊 Dashboard: http://localhost:5000")
    print("📋 API Docs: http://localhost:5000/api/docs")
    print("🔍 Status: http://localhost:5000/api/status")
    print()
    print("Target metrics being monitored:")
    for i, metric in enumerate([
        "Average Fish Weight (g)",
        "Survival Rate (%)",
        "Disease Occurrence (Cases)",
        "Temperature (°C)",
        "Dissolved Oxygen (mg/L)",
        "pH",
        "Turbidity (NTU)"
    ], 1):
        print(f"  {i}. {metric}")
    print()
    print("Press Ctrl+C to stop the system")
    print("=" * 60)
    
    # Keep the demo running
    try:
        while True:
            time.sleep(60)
            print(f"System running... {datetime.now().strftime('%H:%M:%S')}")
    except KeyboardInterrupt:
        print("\n\n👋 Stopping the system...")
        
        # Stop simulation
        try:
            requests.post('http://localhost:5000/api/simulation/stop')
            print("✓ Simulation stopped")
        except:
            pass
        
        print("✓ Demo ended")
        return True

if __name__ == "__main__":
    main()
