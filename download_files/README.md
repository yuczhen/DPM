# A-B-C Credit Card Model for Loan Risk Evaluation

## Overview
This project implements a comprehensive credit evaluation system for loan companies to assess borrower risk and classify clients into A (Low Risk), B (Medium Risk), and C (High Risk) categories.

## Features
- **Multi-model approach**: Supports Logistic Regression, Random Forest, XGBoost, and LightGBM
- **A-B-C Classification**: Automatically groups clients into risk categories
- **Risk Scoring**: Provides numerical risk scores for detailed evaluation
- **Feature Engineering**: Automated feature creation and selection
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Production Ready**: Scalable architecture for real-world deployment

## Models Supported
1. **Logistic Regression** - Interpretable baseline model
2. **Random Forest** - Robust ensemble method
3. **XGBoost** - High-performance gradient boosting
4. **LightGBM** - Fast and memory-efficient boosting

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
from credit_evaluator import CreditEvaluator
from data_processor import DataProcessor

# Load and process data
processor = DataProcessor()
X, y = processor.load_sample_data()

# Initialize and train model
evaluator = CreditEvaluator(model_type='xgboost')
evaluator.fit(X, y)

# Make predictions
risk_scores, abc_categories = evaluator.predict_risk(X_new)
```

## Project Structure
```
├── credit_evaluator.py    # Main credit evaluation class
├── data_processor.py      # Data preprocessing and feature engineering
├── model_trainer.py       # Model training and optimization
├── risk_classifier.py     # A-B-C classification logic
├── evaluation_metrics.py  # Model evaluation and reporting
├── config.py             # Configuration and parameters
└── sample_data/          # Sample datasets for testing
```

## Risk Categories
- **A (Low Risk)**: Score 0-300 - High creditworthiness, low default probability
- **B (Medium Risk)**: Score 301-600 - Moderate risk, requires standard monitoring
- **C (High Risk)**: Score 601-1000 - High default risk, requires careful evaluation

## License
MIT License - Free for commercial use