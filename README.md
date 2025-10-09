# DPM (Default Prediction Model) for Existing Clients

A comprehensive machine learning pipeline for predicting default risk of existing clients who have already been lent money, using XGBoost, LightGBM, and CatBoost models.

## 📋 Project Overview

This project implements an enterprise-grade default prediction system for **existing clients** that:
- Predicts default probability for clients with existing loans
- Classifies clients into risk categories (Excellent to Loss)
- Provides portfolio management recommendations
- Monitors early warning signals for risk deterioration
- Calculates required provisions for loan loss reserves
- Supports model interpretability through SHAP analysis

## 🏗️ Project Structure

```
DPM/
├── .env                        # Environment variables (W&B API key, DB config)
├── config.py                   # Configuration and scoring standards
├── config/
│   └── data_config.py          # Data processing and temporal configuration
├── db_logger.py                # Database logging for experiments and predictions
├── example_time_aware_training.py # Time-aware training example
├── Train/
│   └── main.py                 # Main training pipeline with time-aware features
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── models/                     # Saved model artifacts
│   ├── XGBoost.pkl
│   ├── LightGBM.pkl
│   ├── CatBoost.pkl
│   ├── ensemble_model.pkl
│   └── preprocessing_pipeline.pkl
└── Train/Source/               # Data files
    ├── raw/                    # Raw input data
    ├── processed/              # Processed datasets
    └── *.csv, *.xlsx           # Original datasets
```

## 🚀 Features

### Machine Learning Models
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast and efficient gradient boosting
- **CatBoost**: Handles categorical features automatically
- **Ensemble**: Voting classifier combining all models

### Feature Engineering
- **Basic Features**: Age, income, debt, credit history, payment behavior
- **Behavioral Features**: Payment patterns, account activity, utilization rates
- **Risk Indicators**: Payment risk score, financial stability score, early warning signals
- **Temporal Features**: Application date, economic periods, time-based trends, concept drift indicators
- **Portfolio Features**: Loan characteristics, account aging, relationship depth

### New Time-Aware Features
- **Temporal Splitting**: Chronological train/val/test splits to prevent data leakage
- **Economic Indicators**: GDP trends, unemployment rates, inflation periods
- **Time-based Metrics**: Application year/month, days since application, seasonal patterns
- **Concept Drift Detection**: Performance monitoring across different time periods
- **Data Freshness**: Recent data weighting and staleness detection

### Risk Classification System
- **7-Tier Classification**: Excellent → Good → Satisfactory → Watch → Substandard → Doubtful → Loss
- **Default Probability**: 0-100% risk assessment with precision monitoring
- **Portfolio Actions**: Specific recommendations for each risk tier
- **Provision Calculations**: Automated loan loss reserve calculations

## 📊 Risk Classification System

| Classification | Default Risk | Portfolio Action | Review Frequency | Provision Rate |
|---------------|--------------|------------------|------------------|----------------|
| Excellent     | 0-2%         | Maintain         | Annual          | 0.1%           |
| Good          | 2-5%         | Maintain         | Annual          | 0.5%           |
| Satisfactory  | 5-10%        | Monitor          | Semi-Annual     | 1.0%           |
| Watch         | 10-20%       | Enhanced Monitor | Quarterly       | 3.0%           |
| Substandard   | 20-35%       | Restructure      | Monthly         | 10.0%          |
| Doubtful      | 35-60%       | Workout          | Weekly          | 50.0%          |
| Loss          | 60%+         | Write-off        | Daily           | 100.0%         |

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd DPM
```

2. **Setup environment variables:**
```bash
# Create .env file and set your W&B API key
# Get your W&B API key from: https://wandb.ai/settings

# Edit .env file with your actual values:
# WANDB_API_KEY=your_actual_wandb_api_key
# WANDB_ENTITY=your_wandb_username
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Required packages:**
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
catboost>=1.0.0
shap>=0.40.0
matplotlib>=3.5.0
seaborn>=0.11.0
joblib>=1.1.0
imbalanced-learn>=0.8.0
wandb>=0.15.0          # Experiment tracking
python-dotenv>=1.0.0    # Environment variables
```

## 📖 Usage

### Quick Start

```python
from Train.main import AdvancedDefaultPredictionPipeline
import config

# Initialize the pipeline
pipeline = AdvancedDefaultPredictionPipeline(random_state=42)

# Generate or load data
data = pipeline.generate_dataset(n_samples=10000)

# Feature engineering
processed_data = pipeline.advanced_feature_engineering(data)

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = pipeline.advanced_train_test_split(processed_data)

# Create preprocessing pipeline
preprocessing = pipeline.create_preprocessing_pipeline()

# Transform data
X_train_processed = preprocessing.fit_transform(X_train)
X_val_processed = preprocessing.transform(X_val)
X_test_processed = preprocessing.transform(X_test)

# Initialize and train models
models = pipeline.initialize_models()
results = pipeline.train_and_evaluate_models(
    X_train_processed, y_train,
    X_val_processed, y_val,
    X_test_processed, y_test
)

# Build ensemble
ensemble = pipeline.build_ensemble()

# Save models
pipeline.save_model('models/')
```

### Predict Existing Client Default Risk

```python
# Example existing client data
existing_client = {
    'Client_ID': 'CLT_12345',
    'Age': 35,
    'Annual_Income': 60000,
    'Loan_Amount': 15000,
    'Outstanding_Balance': 12000,
    'Credit_History_Months': 120,
    'On_Time_Payment_Rate': 0.95,
    'Late_Payments_Count': 1,
    'Credit_Utilization_Rate': 0.45,
    'Days_Past_Due': 0,
    'Employment_Status': 1,  # Full-time
    'Education_Level': 3     # Bachelor's
}

# Get default risk prediction
prediction = pipeline.predict_client_default(existing_client)
risk_assessment = prediction['risk_assessment']

print(f"Default Probability: {risk_assessment['default_probability']:.2%}")
print(f"Risk Classification: {risk_assessment['risk_classification']}")
print(f"Portfolio Action: {risk_assessment['portfolio_action']}")
print(f"Provision Required: ${risk_assessment['provision_amount']:,.2f}")
print(f"Review Frequency: {risk_assessment['review_frequency']}")
```

## ⏰ Time-Aware Training (NEW)

### Overview
The system now supports **time-aware training** with concept drift detection and temporal data splitting. This addresses the challenge of changing economic conditions and customer behavior over time.

### Key Features
- **Temporal Data Splitting**: Ensures training data comes before test data chronologically
- **W&B Integration**: Complete experiment tracking and visualization
- **Concept Drift Detection**: Monitors model performance changes over time
- **Recent Data Focus**: Filter data to recent years (default: 5 years)

### Quick Start with Time-Aware Training

```python
from Train.main import AdvancedDefaultPredictionPipeline

# Initialize with W&B tracking
pipeline = AdvancedDefaultPredictionPipeline(
    random_state=42,
    use_wandb=True,  # Enable W&B tracking
    wandb_project="dpm_credit_risk"
)

# Load and filter recent data
data_path = 'Train/Source/客戶基本資料 10201-11305.csv'
recent_data = pipeline.load_and_filter_recent_data(data_path, years=5)

# Feature engineering
processed_data = pipeline.advanced_feature_engineering(recent_data)

# Time-aware data splitting
X_train, X_val, X_test, y_train, y_val, y_test = pipeline.temporal_train_test_split(
    processed_data,
    train_ratio=0.7,
    val_ratio=0.15
)

# Train models
preprocessing = pipeline.create_preprocessing_pipeline()
X_train_processed = preprocessing.fit_transform(X_train)

models = pipeline.initialize_models()
results = pipeline.train_and_evaluate_models(
    X_train_processed, y_train,
    X_val_processed, y_val,
    X_test_processed, y_test
)
```

### Environment Configuration
Create a `.env` file in the project root with your settings:

```bash
# W&B Configuration
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT_NAME=dpm_credit_risk
WANDB_ENTITY=your_wandb_username

# Database Configuration
DATABASE_PATH=experiments.db
DATABASE_BACKUP_PATH=backups/experiments_backup.db

# Data Configuration
USE_RECENT_YEARS=5
DATA_VERSION=v2.0_time_aware
```

### W&B Dashboard
- **View experiments**: https://wandb.ai/your_username/dpm_credit_risk
- **Compare models**: Visual comparison of different time periods
- **Track metrics**: Real-time monitoring of model performance
- **Share results**: Collaborate with team members

### Data Splitting Strategy
The time-aware splitting ensures:
- **Training data**: Earliest time period (e.g., 2019-2022)
- **Validation data**: Middle period (e.g., 2023 Q1-Q3)
- **Test data**: Most recent period (e.g., 2023 Q4)

This prevents data leakage and provides realistic performance estimates.

## 📊 Model Performance

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate for defaults
- **Recall**: Coverage of actual defaults
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve
- **Log Loss**: Probability calibration quality

### Expected Performance
- **Accuracy**: ~85-90%
- **AUC-ROC**: ~0.85-0.92
- **Precision**: ~75-85%
- **Recall**: ~70-80%

## 🔧 Configuration

The `config.py` file contains all configurable parameters:

- **Scoring thresholds**: Risk grade boundaries
- **Model parameters**: Default hyperparameters
- **Feature engineering**: Feature selection criteria
- **Business rules**: Approval/rejection logic

## 📈 Model Interpretability

### SHAP Analysis
- Global feature importance
- Individual prediction explanations
- Feature interaction effects
- Model comparison insights

### Feature Importance
- Tree-based feature importance scores
- Permutation importance analysis
- Correlation analysis with target variable

## 🔄 Model Monitoring

### Performance Tracking
- Prediction accuracy over time
- Feature drift detection
- Model degradation alerts
- Retraining recommendations

### Business Metrics
- Approval rates by risk grade
- Default rates by score range
- Portfolio performance analysis
- ROI calculations

## 🚨 Risk Management

### Model Validation
- Cross-validation with temporal splits
- Out-of-time testing
- Stress testing scenarios
- Model comparison analysis

### Regulatory Compliance
- Fair lending practices
- Model documentation
- Audit trail maintenance
- Bias detection and mitigation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🏷️ Version History

- **v1.0.0**: Initial release with basic scoring functionality
- **v1.1.0**: Added ensemble methods and SHAP analysis
- **v1.2.0**: Enhanced feature engineering and risk grading
- **v2.0.0**: Enterprise-grade pipeline with advanced monitoring
- **v2.1.0**: Added time-aware training with concept drift detection and W&B integration

## 📄 Copyright Notice

© 2023-2025. All rights reserved. This project is proprietary and confidential.
This software and its documentation are protected by copyright law. Unauthorized copying, modification, distribution, or use is strictly prohibited.

**Note**: This system is for educational, research and internal business purposes only. For production use in financial services, ensure compliance with relevant regulations and conduct thorough validation.