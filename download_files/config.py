"""
Configuration file for Credit Evaluation System
"""

# Risk Score Thresholds for A-B-C Classification
RISK_THRESHOLDS = {
    'A': (0, 300),      # Low Risk
    'B': (301, 600),    # Medium Risk  
    'C': (601, 1000)    # High Risk
}

# Model Parameters
MODEL_CONFIGS = {
    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'objective': 'binary:logistic'
    },
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'objective': 'binary',
        'verbose': -1
    }
}

# Feature Engineering Settings
FEATURE_SETTINGS = {
    'debt_to_income_threshold': 0.4,
    'credit_utilization_threshold': 0.8,
    'payment_history_months': 12,
    'missing_value_threshold': 0.3
}

# Evaluation Metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision', 
    'recall',
    'f1_score',
    'roc_auc',
    'confusion_matrix'
]

# Data Processing
DATA_SETTINGS = {
    'test_size': 0.2,
    'validation_size': 0.2,
    'random_state': 42,
    'stratify': True
}