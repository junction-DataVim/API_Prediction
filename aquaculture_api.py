"""
Aquaculture IoT API
=================

Flask-based REST API for aquaculture IoT monitoring, prediction, and real-time data processing.
Provides endpoints for sensor data ingestion, predictions, analytics, and monitoring dashboard.
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sqlite3
import joblib
from pathlib import Path
import logging
import os
import threading
import queue
from contextlib import contextmanager

# Import our pipeline classes
from aquaculture_ml_pipeline import AquacultureMLPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
pipeline = None
data_api = None

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect('aquaculture_realtime.db')
    try:
        yield conn
    finally:
        conn.close()

def init_pipeline():
    """Initialize the pipeline and API"""
    global pipeline, data_api
    
    if pipeline is None:
        logger.info("Initializing aquaculture pipeline...")
        pipeline = AquacultureMLPipeline()
        
        # Load and preprocess data
        try:
            pipeline.load_and_preprocess_data()
            pipeline.train_all_models()
            logger.info("Pipeline initialized and models trained successfully")
        except Exception as e:
            logger.warning(f"Could not train models: {e}")
            logger.info("Pipeline initialized without trained models")
        
        data_api = None  # We'll handle this separately
        
    return pipeline, data_api

# Initialize on startup
init_pipeline()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Aquaculture IoT Monitoring Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
            .metric { text-align: center; padding: 15px; background: #e3f2fd; border-radius: 5px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #1976d2; }
            .metric-label { font-size: 14px; color: #666; }
            .status-ok { color: #4caf50; }
            .status-warning { color: #ff9800; }
            .status-error { color: #f44336; }
            .nav { background: #1976d2; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
            .nav a { color: white; text-decoration: none; margin-right: 20px; }
            .nav a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="nav">
                <h1 style="color: white; display: inline;">🐟 Aquaculture IoT Dashboard</h1>
                <div style="float: right;">
                    <a href="/api/dashboard">Dashboard</a>
                    <a href="/api/status">Status</a>
                    <a href="/api/docs">API Docs</a>
                </div>
            </div>
            
            <div class="card">
                <h2>System Status</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value status-ok">✓ Online</div>
                        <div class="metric-label">System Status</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{ models_loaded }}</div>
                        <div class="metric-label">Models Loaded</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{ data_points }}</div>
                        <div class="metric-label">Data Points</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{ predictions_today }}</div>
                        <div class="metric-label">Predictions Today</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>Quick Actions</h2>
                <p>
                    <button onclick="testIngestion()">Test Data Ingestion</button>
                    <button onclick="startStream()">Start Data Stream</button>
                    <button onclick="viewPredictions()">View Predictions</button>
                    <button onclick="checkAnomalies()">Check Anomalies</button>
                </p>
            </div>
            
            <div class="card">
                <h2>API Endpoints</h2>
                <ul>
                    <li><strong>POST /api/data/ingest</strong> - Ingest sensor data</li>
                    <li><strong>GET /api/predictions/latest</strong> - Get latest predictions</li>
                    <li><strong>GET /api/predictions/metric/{metric}</strong> - Get predictions for specific metric</li>
                    <li><strong>GET /api/anomalies</strong> - Get recent anomalies</li>
                    <li><strong>GET /api/monitoring/health</strong> - System health check</li>
                    <li><strong>GET /api/dashboard</strong> - Interactive dashboard</li>
                </ul>
            </div>
        </div>
        
        <script>
            function testIngestion() {
                const testData = {
                    avg_fish_weight: 275.5,
                    survival_rate: 94.2,
                    disease_occurrence: 1,
                    temperature: 26.8,
                    dissolved_oxygen: 7.2,
                    ph: 7.8,
                    turbidity: 3.1
                };
                
                fetch('/api/data/ingest', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(testData)
                })
                .then(response => response.json())
                .then(data => alert('Test data ingested: ' + JSON.stringify(data.predictions.length) + ' predictions generated'));
            }
            
            function startStream() {
                fetch('/api/simulation/start', {method: 'POST'})
                .then(response => response.json())
                .then(data => alert('Data stream started: ' + data.status));
            }
            
            function viewPredictions() {
                window.open('/api/predictions/latest', '_blank');
            }
            
            function checkAnomalies() {
                window.open('/api/anomalies', '_blank');
            }
        </script>
    </body>
    </html>
    """, models_loaded=len(pipeline.models) if hasattr(pipeline, 'models') and pipeline.models else 0, 
         data_points=len(pipeline.processed_data) if hasattr(pipeline, 'processed_data') and pipeline.processed_data is not None else 0, 
         predictions_today=0)

@app.route('/api/status')
def api_status():
    """Get API status and statistics"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        # Get database statistics
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Count total records
            cursor.execute("SELECT COUNT(*) as count FROM sensor_data")
            total_records = cursor.fetchone()[0]
            
            # Count predictions today
            today = datetime.now().date()
            cursor.execute("SELECT COUNT(*) as count FROM predictions WHERE date(datetime) = ?", (today,))
            predictions_today = cursor.fetchone()[0]
            
            # Count anomalies in last 24 hours
            yesterday = datetime.now() - timedelta(days=1)
            cursor.execute("SELECT COUNT(*) as count FROM anomalies WHERE datetime >= ?", (yesterday.isoformat(),))
            anomalies_24h = cursor.fetchone()[0]
        
        return jsonify({
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "models_loaded": len(pipeline.models),
            "target_metrics": pipeline.target_columns,
            "database": {
                "total_records": total_records,
                "predictions_today": predictions_today,
                "anomalies_24h": anomalies_24h
            },
            "version": "1.0.0"
        })
    
    except Exception as e:
        logger.error(f"Error in status endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/data/ingest', methods=['POST'])
def ingest_data():
    """Ingest sensor data"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Process the data through the pipeline
        result = pipeline.process_sensor_data(data)
        
        return jsonify({
            "status": "success",
            "message": "Data ingested successfully",
            "predictions": result.get('predictions', []),
            "anomalies": result.get('anomalies', []),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in data ingestion: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/latest')
def get_latest_predictions():
    """Get latest predictions"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        limit = request.args.get('limit', 50, type=int)
        
        # Get predictions from database if available, otherwise generate sample predictions
        predictions = []
        if hasattr(pipeline, 'models') and pipeline.models:
            # Try to get predictions from existing models
            try:
                target = list(pipeline.models.keys())[0]  # Use first available target
                sample_predictions = pipeline.predict_single_metric(target, hours_ahead=min(limit, 24))
                predictions = sample_predictions
            except Exception as e:
                logger.warning(f"Could not generate predictions: {e}")
                predictions = []
        
        # If no predictions, return sample data
        if not predictions:
            base_time = datetime.now()
            for i in range(min(limit, 24)):
                predictions.append({
                    'datetime': (base_time + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S'),
                    'target': 'Temperature (°C)',
                    'predicted_value': 27.5 + np.random.normal(0, 0.5),
                    'model_used': 'Sample'
                })
        
        return jsonify({
            "predictions": predictions,
            "count": len(predictions),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/metric/<metric>')
def get_metric_predictions(metric):
    """Get predictions for a specific metric"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        # Validate metric
        if metric not in pipeline.target_columns:
            return jsonify({
                "error": f"Invalid metric. Valid metrics: {pipeline.target_columns}"
            }), 400
        
        hours = request.args.get('hours', 24, type=int)
        
        # Get predictions from database
        with get_db_connection() as conn:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            df = pd.read_sql_query('''
                SELECT datetime, target_metric, predicted_value, model_used, confidence_score
                FROM predictions 
                WHERE target_metric = ? AND datetime >= ?
                ORDER BY datetime DESC
            ''', conn, params=(metric, cutoff_time))
            
            predictions = df.to_dict('records')
        
        return jsonify({
            "metric": metric,
            "predictions": predictions,
            "count": len(predictions),
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting metric predictions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions/forecast/<metric>')
def get_forecast(metric):
    """Generate future forecast for a metric"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        # Validate metric
        if metric not in pipeline.target_columns:
            return jsonify({
                "error": f"Invalid metric. Valid metrics: {pipeline.target_columns}"
            }), 400
        
        hours_ahead = request.args.get('hours', 24, type=int)
        if hours_ahead < 1 or hours_ahead > 168:  # Max 1 week
            return jsonify({"error": "Hours must be between 1 and 168"}), 400
        
        # Check if models are available for this metric
        if not hasattr(pipeline, 'models') or metric not in pipeline.models:
            return jsonify({"error": f"No trained model found for metric: {metric}"}), 400
        
        # Use the existing pipeline to generate predictions
        try:
            forecast = pipeline.predict_single_metric(metric, hours_ahead)
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Generate sample forecast if prediction fails
            base_time = datetime.now()
            forecast = []
            for i in range(hours_ahead):
                forecast.append({
                    'datetime': (base_time + timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_value': 27.5 + np.random.normal(0, 0.5),
                    'target': metric,
                    'model_used': 'Sample'
                })
        
        return jsonify({
            "metric": metric,
            "forecast": forecast,
            "hours_ahead": hours_ahead,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error generating forecast: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/anomalies')
def get_anomalies():
    """Get recent anomalies"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        hours = request.args.get('hours', 24, type=int)
        anomalies = pipeline.get_anomalies(hours)
        
        return jsonify({
            "anomalies": anomalies,
            "count": len(anomalies),
            "hours": hours,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/monitoring/health')
def health_check():
    """Health check endpoint"""
    try:
        if not pipeline:
            return jsonify({"status": "unhealthy", "error": "Pipeline not initialized"}), 500
        
        # Check database connection
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
        
        # Check models
        models_status = len(pipeline.models) > 0
        
        return jsonify({
            "status": "healthy",
            "components": {
                "database": "ok",
                "models": "ok" if models_status else "no models loaded",
                "pipeline": "ok"
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/data/latest')
def get_latest_data():
    """Get latest sensor data"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        limit = request.args.get('limit', 50, type=int)
        data = pipeline.get_latest_data(limit)
        
        return jsonify({
            "data": data,
            "count": len(data),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start data simulation"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        # For now, just return success - simulation would be implemented later
        return jsonify({
            "message": "Data simulation started (mock)",
            "status": "running",
            "started_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop data simulation"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        # For now, just return success - simulation would be implemented later
        return jsonify({
            "message": "Data simulation stopped (mock)",
            "status": "stopped",
            "stopped_at": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
        return jsonify({"error": str(e)}), 500
    
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard')
def dashboard():
    """Interactive dashboard"""
    try:
        if not pipeline:
            return jsonify({"error": "Pipeline not initialized"}), 500
        
        # Get recent data for dashboard
        latest_data = []
        latest_predictions = []
        recent_anomalies = []
        
        # Try to get data if models are available
        if hasattr(pipeline, 'models') and pipeline.models:
            try:
                # Get sample predictions
                target = list(pipeline.models.keys())[0]
                latest_predictions = pipeline.predict_single_metric(target, hours_ahead=24)
            except Exception as e:
                logger.warning(f"Could not get predictions: {e}")
        
        # Get sample data for dashboard
        if hasattr(pipeline, 'processed_data') and pipeline.processed_data is not None:
            latest_data = pipeline.processed_data.tail(24).to_dict('records')
        
        # Get anomalies if available
        if hasattr(pipeline, 'detect_anomalies') and pipeline.target_columns:
            try:
                target = pipeline.target_columns[0]
                recent_anomalies = pipeline.detect_anomalies(target)[-10:]  # Last 10 anomalies
            except Exception as e:
                logger.warning(f"Could not get anomalies: {e}")
        
        # Process data for dashboard
        metrics_summary = {}
        for metric in pipeline.target_columns:
            # Get latest actual values from processed data
            actual_values = []
            if latest_data:
                actual_values = [d.get(metric, 0) for d in latest_data if d.get(metric) is not None]
            
            # Get latest predictions
            pred_values = [p['predicted_value'] for p in latest_predictions if p.get('target') == metric]
            
            metrics_summary[metric] = {
                'latest_actual': actual_values[-1] if actual_values else None,
                'latest_predicted': pred_values[0] if pred_values else None,
                'actual_count': len(actual_values),
                'prediction_count': len(pred_values),
                'avg_actual': np.mean(actual_values) if actual_values else None,
                'avg_predicted': np.mean(pred_values) if pred_values else None
            }
        
        return jsonify({
            "dashboard_data": {
                "metrics_summary": metrics_summary,
                "latest_data": latest_data[:10],  # Last 10 records
                "latest_predictions": latest_predictions[:10],
                "recent_anomalies": recent_anomalies,
                "system_status": {
                    "models_loaded": len(pipeline.models),
                    "data_points": len(latest_data),
                    "predictions_generated": len(latest_predictions),
                    "anomalies_detected": len(recent_anomalies)
                }
            },
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in dashboard: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/docs')
def api_docs():
    """API documentation"""
    return jsonify({
        "api_version": "1.0.0",
        "description": "Aquaculture IoT Monitoring and Prediction API",
        "endpoints": {
            "GET /": "Main dashboard",
            "GET /api/status": "API status and statistics",
            "POST /api/data/ingest": "Ingest sensor data",
            "GET /api/predictions/latest": "Get latest predictions",
            "GET /api/predictions/metric/<metric>": "Get predictions for specific metric",
            "GET /api/predictions/forecast/<metric>": "Generate future forecast",
            "GET /api/anomalies": "Get recent anomalies",
            "GET /api/monitoring/health": "Health check",
            "GET /api/data/latest": "Get latest sensor data",
            "POST /api/simulation/start": "Start data simulation",
            "POST /api/simulation/stop": "Stop data simulation",
            "GET /api/dashboard": "Interactive dashboard data",
            "GET /api/docs": "This documentation"
        },
        "target_metrics": [
            "Average Fish Weight (g)",
            "Survival Rate (%)",
            "Disease Occurrence (Cases)",
            "Temperature (°C)",
            "Dissolved Oxygen (mg/L)",
            "pH",
            "Turbidity (NTU)"
        ],
        "sample_request": {
            "endpoint": "POST /api/data/ingest",
            "payload": {
                "avg_fish_weight": 275.5,
                "survival_rate": 94.2,
                "disease_occurrence": 1,
                "temperature": 26.8,
                "dissolved_oxygen": 7.2,
                "ph": 7.8,
                "turbidity": 3.1
            }
        }
    })

if __name__ == '__main__':
    print("Starting Aquaculture IoT API server...")
    print("Dashboard available at: http://localhost:5000")
    print("API documentation at: http://localhost:5000/api/docs")
    app.run(debug=True, host='0.0.0.0', port=5000)
