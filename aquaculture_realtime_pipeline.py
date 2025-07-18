"""
Aquaculture Real-Time Pipeline
=============================

Real-time data processing pipeline for aquaculture IoT monitoring.
"""

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from aquaculture_ml_pipeline import AquacultureMLPipeline

class AquacultureRealTimePipeline:
    """
    Real-time processing pipeline for aquaculture data
    """
    
    def __init__(self, db_path='aquaculture_realtime.db'):
        self.db_path = db_path
        self.ml_pipeline = None
        self.init_database()
        
    def init_database(self):
        """Initialize the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sensor data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                avg_fish_weight REAL,
                survival_rate REAL,
                disease_occurrence INTEGER,
                temperature REAL,
                dissolved_oxygen REAL,
                ph REAL,
                turbidity REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                target_metric TEXT,
                predicted_value REAL,
                model_used TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_ml_pipeline(self):
        """Load the ML pipeline"""
        if self.ml_pipeline is None:
            self.ml_pipeline = AquacultureMLPipeline()
            # Load pre-trained models if they exist
            try:
                self.ml_pipeline.load_and_preprocess_data()
                print("ML pipeline loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load ML pipeline: {e}")
        
        return self.ml_pipeline
    
    def ingest_data(self, data):
        """Ingest sensor data into the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sensor_data 
            (avg_fish_weight, survival_rate, disease_occurrence, temperature, 
             dissolved_oxygen, ph, turbidity) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data.get('avg_fish_weight'),
            data.get('survival_rate'),
            data.get('disease_occurrence'),
            data.get('temperature'),
            data.get('dissolved_oxygen'),
            data.get('ph'),
            data.get('turbidity')
        ))
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_latest_data(self, limit=100):
        """Get the latest sensor data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM sensor_data 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        data = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return data
    
    def get_system_status(self):
        """Get current system status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get data count
        cursor.execute('SELECT COUNT(*) FROM sensor_data')
        total_records = cursor.fetchone()[0]
        
        # Get latest data
        cursor.execute('SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1')
        latest_data = cursor.fetchone()
        
        conn.close()
        
        return {
            'status': 'online',
            'total_records': total_records,
            'latest_data': latest_data,
            'last_update': datetime.now().isoformat()
        }

class AquacultureDataStreamAPI:
    """
    API wrapper for real-time data streaming
    """
    
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def process_data_stream(self, data_stream):
        """Process a stream of data"""
        results = []
        
        for data_point in data_stream:
            try:
                # Ingest data
                self.pipeline.ingest_data(data_point)
                
                # Process and return result
                results.append({
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                    'data': data_point
                })
                
            except Exception as e:
                results.append({
                    'status': 'error',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e),
                    'data': data_point
                })
        
        return results
    
    def get_real_time_metrics(self):
        """Get real-time metrics"""
        latest_data = self.pipeline.get_latest_data(limit=1)
        
        if not latest_data:
            return {
                'status': 'no_data',
                'metrics': {}
            }
        
        data = latest_data[0]
        
        # Calculate basic metrics
        metrics = {
            'avg_fish_weight': data.get('avg_fish_weight', 0),
            'survival_rate': data.get('survival_rate', 0),
            'disease_occurrence': data.get('disease_occurrence', 0),
            'temperature': data.get('temperature', 0),
            'dissolved_oxygen': data.get('dissolved_oxygen', 0),
            'ph': data.get('ph', 0),
            'turbidity': data.get('turbidity', 0),
            'timestamp': data.get('timestamp', ''),
            'status': 'active'
        }
        
        return {
            'status': 'success',
            'metrics': metrics,
            'last_update': datetime.now().isoformat()
        }
