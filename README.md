# End-to-End ML Project for Gemstone Price Prediction

A complete machine learning pipeline for predicting gemstone prices using CatBoost regression model. This project demonstrates best practices in ML engineering including DVC for pipeline orchestration, MLflow for experiment tracking, and FastAPI for model serving.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [DVC Pipeline](#dvc-pipeline)
  - [FastAPI Server](#fastapi-server)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Configuration](#configuration)
- [Results](#results)

## 🎯 Project Overview

This project builds an end-to-end machine learning system for predicting gemstone prices based on physical characteristics (carat weight, cut quality, color grade, clarity, dimensions). The pipeline includes:

- **Data Ingestion**: Download and extract gemstone dataset
- **Data Validation**: Validate data schema and quality
- **Data Transformation**: Feature engineering, encoding, and scaling
- **Model Training**: Train CatBoost regressor with hyperparameter tuning
- **Model Evaluation**: Evaluate performance and track metrics
- **Prediction API**: FastAPI server for real-time predictions

## ✨ Features

- ✅ **DVC Data Pipeline**: Reproducible ML workflow with dependency tracking
- ✅ **Parameter Management**: YAML-based configuration for easy experimentation
- ✅ **MLflow Integration**: Experiment tracking and model registry
- ✅ **FastAPI Server**: RESTful API for single and batch predictions
- ✅ **Preprocessing Pipeline**: Automatic feature encoding and scaling
- ✅ **Error Handling**: Comprehensive validation and error messages
- ✅ **Logging**: Detailed logging for debugging and monitoring

## 📁 Project Structure

```
├── artifacts/                          # Generated artifacts
│   ├── data_ingestion/                # Raw data
│   ├── data_transformation/           # Processed train/test data & preprocessing objects
│   ├── data_validation/               # Validation status
│   ├── model_trainer/                 # Trained model
│   └── model_evaluation/              # Evaluation metrics
├── config/
│   └── config.yaml                    # Configuration for all stages
├── src/
│   └── mlProject/
│       ├── components/                # Pipeline components
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       │   ├── model_evaluation.py
│       │   └── predict.py             # Prediction component
│       ├── config/
│       │   └── configuration.py       # Configuration manager
│       ├── constants/
│       ├── entity/
│       │   └── config_entity.py       # Data classes for configs
│       ├── pipeline/
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   ├── stage_05_model_evaluation.py
│       │   └── stage_06_prediction.py # Prediction pipeline
│       └── utils/
│           └── common.py              # Utility functions
├── templates/
│   └── index.html                     # Web UI (optional)
├── app.py                             # FastAPI application
├── dvc.yaml                           # DVC pipeline definition
├── params.yaml                        # Model hyperparameters
├── schema.yaml                        # Data schema
├── config.yaml                        # Project configuration
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker configuration
└── README.md                          # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Git
- pip/conda

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/ajaychaudhary8104/End_to_End_ML_project_for_Gems_Price_Prediction
cd End_to_End_ML_project_for_Gems_Price_Prediction
```

2. **Create and activate virtual environment**

**Using Conda:**
```bash
conda create -n gems python=3.12 -y
conda activate gems
```

**Using venv:**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Initialize DVC (if not already initialized)**
```bash
dvc init
```

## 📊 Usage

### DVC Pipeline

The project uses DVC to manage the ML pipeline with reproducible stages.

#### Run the entire pipeline
```bash
dvc repro
```

#### Run a specific stage
```bash
dvc repro stage_name
```

#### View the pipeline DAG
```bash
dvc dag
```

#### Check pipeline status
```bash
dvc status
```

#### Pipeline Stages:

1. **Data Ingestion** - Downloads gemstone dataset and extracts ZIP file
2. **Data Validation** - Validates data schema against schema.yaml
3. **Data Transformation** - Handles missing values, encodes categories, scales features
4. **Model Training** - Trains CatBoost regression model
5. **Model Evaluation** - Evaluates model and calculates metrics
6. **Prediction** - Available via API

### FastAPI Server

#### Start the API server
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

#### Access documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Documentation

### Endpoints

#### 1. Root Endpoint
```
GET /
```
Returns API information and available endpoints.

#### 2. Health Check
```
GET /health
```
Returns API health status.

```json
{
  "status": "healthy",
  "message": "API is running"
}
```

#### 3. Single Prediction
```
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "carat": 0.5,
  "cut": "Ideal",
  "color": "E",
  "clarity": "SI1",
  "depth": 61.5,
  "table": 55.0,
  "x": 5.1,
  "y": 5.2,
  "z": 3.2
}
```

**Response:**
```json
{
  "predicted_price": 2245.50,
  "message": "Prediction successful"
}
```

#### 4. Batch Predictions
```
POST /batch-predict
Content-Type: application/json
```

**Request Body:**
```json
[
  {
    "carat": 0.5,
    "cut": "Ideal",
    "color": "E",
    "clarity": "SI1",
    "depth": 61.5,
    "table": 55.0,
    "x": 5.1,
    "y": 5.2,
    "z": 3.2
  },
  {
    "carat": 1.0,
    "cut": "Premium",
    "color": "F",
    "clarity": "VS1",
    "depth": 62.0,
    "table": 56.0,
    "x": 6.5,
    "y": 6.6,
    "z": 4.1
  }
]
```

**Response:**
```json
{
  "total_predictions": 2,
  "predictions": [
    {
      "input": {...},
      "predicted_price": 2245.50
    },
    {
      "input": {...},
      "predicted_price": 5432.75
    }
  ],
  "message": "Batch prediction successful"
}
```

#### 5. Model Information
```
GET /info
```
Returns model type, input features, and valid values.

### Input Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| **carat** | float | > 0 | Weight of gemstone in carats |
| **cut** | string | Fair, Good, Very Good, Premium, Ideal | Cut quality |
| **color** | string | D-J | Color grade (D is best) |
| **clarity** | string | I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF | Clarity level |
| **depth** | float | 0-100 | Depth percentage |
| **table** | float | 0-100 | Table percentage |
| **x** | float | ≥ 0 | Length in mm |
| **y** | float | ≥ 0 | Width in mm |
| **z** | float | ≥ 0 | Height in mm |

### Example Requests (cURL)

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "carat": 0.5,
    "cut": "Ideal",
    "color": "E",
    "clarity": "SI1",
    "depth": 61.5,
    "table": 55.0,
    "x": 5.1,
    "y": 5.2,
    "z": 3.2
  }'
```

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Model Info:**
```bash
curl http://localhost:8000/info
```

## 🤖 Model Details

### Algorithm
**CatBoost Regressor** - A gradient boosting algorithm optimized for categorical features.

### Hyperparameters
All hyperparameters are defined in `params.yaml`:
```yaml
CatBoostRegressor:
  depth: 6
  learning_rate: 0.03
  iterations: 600
  verbose: False
  random_state: 42
```

### Training Data
- **Dataset**: Gemstone price dataset
- **Samples**: ~193,500 gemstone records
- **Features**: 9 input features + 1 target (price)
- **Train/Test Split**: 80/20

### Preprocessing
1. **Handling Missing Values**: Median for numerical, mode for categorical
2. **Encoding**: LabelEncoder for categorical features
3. **Scaling**: StandardScaler for numerical features (carat, depth, table, x, y, z)
4. **Feature Selection**: All 9 features used based on domain knowledge

### Evaluation Metrics
Located in `artifacts/model_evaluation/metrics.json`:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Percentage Error (MAPE)

## ⚙️ Configuration

### config.yaml
Main configuration file specifying paths and hyperparameters for each stage.

### params.yaml
Model hyperparameters used during training:
```yaml
CatBoostRegressor:
  depth: 6
  learning_rate: 0.03
  iterations: 600
  verbose: False
  random_state: 42
```

### schema.yaml
Data schema for validation with column names and types.

## 🐳 Docker Deployment

Build and run with Docker:
```bash
# Build image
docker build -t gemstone-predictor:latest .

# Run container
docker run -p 8000:8000 gemstone-predictor:latest
```

## 🔄 MLflow Experiment Tracking

View MLflow dashboard:
```bash
mlflow ui
```

Access at `http://localhost:5000`

## 📝 Logging

Logs are saved in the `logs/` directory. Check for detailed error messages and execution traces.

## 🐛 Troubleshooting

### Model Not Found
```
FileNotFoundError: Model not found at artifacts/model_trainer/model.joblib
```
**Solution**: Run `dvc repro` to train the model first.

### Invalid Feature Error
```
The feature names should match those that were passed during fit
```
**Solution**: Ensure all required features are provided in the correct format.

### Module Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Run from project root directory.

### Preprocessing Object Not Found
```
Warning: Label encoders/scaler not found
```
**Solution**: Run `dvc repro` to generate preprocessing objects.

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [CatBoost Documentation](https://catboost.ai/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

Created as a demonstration of end-to-end ML pipeline best practices.

---

**Last Updated**: February 2026

For questions or issues, please open an issue on the repository.