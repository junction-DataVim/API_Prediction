# Aquaculture IoT Monitoring & Prediction API

A comprehensive Flask-based REST API for aquaculture IoT monitoring, machine learning predictions, and real-time data processing. This system provides advanced analytics for fish farming operations including environmental monitoring, fish health predictions, and anomaly detection.

## 🐟 Features

- **Real-time Data Ingestion**: Process sensor data from IoT devices
- **Machine Learning Predictions**: Predict fish weight, survival rates, disease occurrence, and environmental parameters
- **Anomaly Detection**: Identify unusual patterns in aquaculture data
- **Interactive Dashboard**: Web-based monitoring interface
- **RESTful API**: Complete API for integration with external systems
- **Data Processing Pipeline**: Automated data preprocessing and feature engineering
- **Multiple ML Models**: Support for RandomForest, XGBoost, LightGBM, and more

## 📊 Supported Metrics

The API can predict and monitor the following aquaculture metrics:

- **Average Fish Weight (g)** - Fish growth tracking
- **Survival Rate (%)** - Fish mortality monitoring
- **Disease Occurrence (Cases)** - Health status prediction
- **Temperature (°C)** - Water temperature monitoring
- **Dissolved Oxygen (mg/L)** - Oxygen level tracking
- **pH** - Water acidity monitoring
- **Turbidity (NTU)** - Water clarity measurement

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd aquaculture-iot-api
```

2. **Create and activate virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install flask flask-cors pandas numpy scikit-learn xgboost lightgbm joblib sqlite3
```

4. **Prepare your data:**
   - Place your IoT sensor data CSV file in the project directory
   - Update the `data_path` in the pipeline initialization if needed

5. **Start the API server:**
```bash
python aquaculture_api.py
```

The API will be available at `http://localhost:5000`

## 🔧 Configuration

### Environment Variables

You can configure the following environment variables:

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
export API_HOST=0.0.0.0
export API_PORT=5000
```

### Data File Format

The API expects CSV data with the following columns:

```csv
Datetime,Month,Average Fish Weight (g),Survival Rate (%),Disease Occurrence (Cases),Temperature (°C),Dissolved Oxygen (mg/L),pH,Turbidity (NTU),Oxygenation Automatic,Corrective Measures,Thermal Risk Index,Low Oxygen Alert,Health Status
```

## 📚 API Documentation

### Base URL
```
http://localhost:5000
```

### Main Endpoints

#### 🏠 Dashboard
- **GET** `/` - Main dashboard interface
- **GET** `/api/dashboard` - Dashboard data (JSON)

#### 📊 System Status
- **GET** `/api/status` - API status and statistics
- **GET** `/api/monitoring/health` - Health check endpoint
- **GET** `/api/docs` - API documentation

#### 📥 Data Ingestion
- **POST** `/api/data/ingest` - Ingest new sensor data
- **GET** `/api/data/latest` - Get latest sensor readings

#### 🔮 Predictions
- **GET** `/api/predictions/latest` - Get recent predictions
- **GET** `/api/predictions/metric/<metric>` - Get predictions for specific metric
- **GET** `/api/predictions/forecast/<metric>` - Generate future forecasts

#### 🚨 Anomaly Detection
- **GET** `/api/anomalies` - Get detected anomalies

#### 🎮 Simulation
- **POST** `/api/simulation/start` - Start data simulation
- **POST** `/api/simulation/stop` - Stop data simulation

### Example Requests

#### Data Ingestion
```bash
curl -X POST http://localhost:5000/api/data/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "avg_fish_weight": 275.5,
    "survival_rate": 94.2,
    "disease_occurrence": 1,
    "temperature": 26.8,
    "dissolved_oxygen": 7.2,
    "ph": 7.8,
    "turbidity": 3.1
  }'
```

#### Get Predictions
```bash
curl http://localhost:5000/api/predictions/latest?limit=24
```

#### Generate Forecast
```bash
curl http://localhost:5000/api/predictions/forecast/Temperature%20%28°C%29?hours=48
```

#### Check System Status
```bash
curl http://localhost:5000/api/status
```

## 🤖 Machine Learning Pipeline

### Data Processing Features

- **Categorical Encoding**: Automatic conversion of text categories to numerical values
- **Time Series Features**: Lag features, rolling averages, and temporal patterns
- **Feature Engineering**: Interaction features and domain-specific transformations
- **Data Validation**: Comprehensive type checking and NaN handling

### Supported Models

- **RandomForest**: Ensemble method for robust predictions
- **XGBoost**: Gradient boosting for high performance
- **LightGBM**: Fast gradient boosting framework
- **GradientBoosting**: Scikit-learn gradient boosting
- **Ridge Regression**: Linear model with L2 regularization
- **ElasticNet**: Linear model with L1 and L2 regularization

### Model Selection

The system automatically selects the best performing model based on R² score for each target metric.

## 🔍 Monitoring & Analytics

### Dashboard Features

- **Real-time Metrics**: Live display of key aquaculture indicators
- **Historical Trends**: Time series visualization of sensor data
- **Prediction Accuracy**: Model performance monitoring
- **Anomaly Alerts**: Real-time anomaly detection and alerting

### Anomaly Detection

The system uses statistical methods to detect anomalies:
- **Z-score Analysis**: Identifies values beyond 2 standard deviations
- **Time Series Patterns**: Detects unusual temporal patterns
- **Multi-variate Analysis**: Considers relationships between metrics

## 📊 Data Schema

### Sensor Data Structure
```json
{
  "timestamp": "2024-07-18T10:30:00Z",
  "avg_fish_weight": 275.5,
  "survival_rate": 94.2,
  "disease_occurrence": 1,
  "temperature": 26.8,
  "dissolved_oxygen": 7.2,
  "ph": 7.8,
  "turbidity": 3.1,
  "oxygenation_automatic": "Yes",
  "corrective_measures": "No",
  "thermal_risk_index": "Normal",
  "low_oxygen_alert": "Safe",
  "health_status": "Stable"
}
```

### Prediction Response Structure
```json
{
  "predictions": [
    {
      "datetime": "2024-07-18T11:00:00",
      "predicted_value": 27.2,
      "target": "Temperature (°C)",
      "model_used": "RandomForest"
    }
  ],
  "count": 1,
  "timestamp": "2024-07-18T10:30:00Z"
}
```

## 🛠️ Development

### Project Structure
```
aquaculture-iot-api/
├── aquaculture_api.py              # Main API application
├── aquaculture_ml_pipeline.py      # ML pipeline and models
├── aquaculture_realtime_pipeline.py # Real-time data processing
├── Data_Model_IoTMLCQ_2024.csv     # Sample dataset
├── models/                         # Trained model files
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

### Adding New Features

1. **New Endpoints**: Add routes in `aquaculture_api.py`
2. **New Models**: Extend the ML pipeline in `aquaculture_ml_pipeline.py`
3. **New Metrics**: Add to `target_columns` in the pipeline class
4. **New Visualizations**: Extend the dashboard templates

### Testing

Test the API endpoints using curl, Postman, or the built-in dashboard:

```bash
# Test health check
curl http://localhost:5000/api/monitoring/health

# Test data ingestion
curl -X POST http://localhost:5000/api/data/ingest \
  -H "Content-Type: application/json" \
  -d '{"temperature": 27.5, "ph": 7.8}'

# Test predictions
curl http://localhost:5000/api/predictions/latest
```

## 🔒 Security Considerations

- **Input Validation**: All API inputs are validated
- **Error Handling**: Comprehensive error handling and logging
- **CORS Support**: Cross-origin resource sharing enabled
- **Rate Limiting**: Consider implementing rate limiting for production

## 🚀 Production Deployment

### Docker Deployment (Recommended)

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "aquaculture_api.py"]
```

### Environment Configuration

For production, use environment variables:
```bash
export FLASK_ENV=production
export FLASK_DEBUG=0
export DATABASE_URL=postgresql://user:pass@localhost/aquaculture
```

### Scaling Considerations

- **Load Balancing**: Use nginx or similar for multiple instances
- **Database**: Consider PostgreSQL for production data storage
- **Caching**: Implement Redis for caching predictions
- **Monitoring**: Add application monitoring and logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- **Documentation**: Check the API docs at `/api/docs`
- **Issues**: Create an issue in the repository
- **Email**: Contact the development team

## 🙏 Acknowledgments

- **Scikit-learn**: For machine learning algorithms
- **Flask**: For the web framework
- **Pandas**: For data manipulation
- **XGBoost & LightGBM**: For advanced ML models

---

**Happy Fish Farming! 🐟📊**
