 # -*- coding: utf-8 -*-
"""
DPM Configuration
=================
Central configuration file for DPM system including:
- Model parameter configuration
- Feature engineering configuration
- Validation rules
- Scoring configuration
"""

import numpy as np

# =============================================================================
# RISK CLASSIFICATION SYSTEM
# =============================================================================

# Risk score ranges for client classification
RISK_SCORE_RANGES = {
    'EXCELLENT': {'min_score': 80, 'max_score': 100, 'label': 'Excellent Client'},
    'GOOD': {'min_score': 60, 'max_score': 79, 'label': 'Good Client'},
    'FAIR': {'min_score': 40, 'max_score': 59, 'label': 'Fair Client'},
    'POOR': {'min_score': 20, 'max_score': 39, 'label': 'Poor Client'},
    'BAD': {'min_score': 0, 'max_score': 19, 'label': 'High Risk Client'}
}

# Good/Bad client classification threshold
DEFAULT_PROBABILITY_THRESHOLD = 0.5  # Default probability threshold

# =============================================================================
# CREDIT SCORING PARAMETERS
# =============================================================================

# Score range (FICO-like scale)
SCORE_RANGE = {
    'min_score': 300,
    'max_score': 850,
    'base_score': 300
}

# Feature weights for basic credit scoring
BASIC_SCORE_WEIGHTS = {
    'payment_history': 0.35,      # 35% - Most important factor
    'amounts_owed': 0.30,         # 30% - Debt-to-income ratio
    'credit_history_length': 0.15, # 15% - Length of credit history
    'new_credit': 0.10,           # 10% - Recent credit inquiries
    'credit_mix': 0.10            # 10% - Types of credit used
}

# Age group risk adjustments
AGE_RISK_ADJUSTMENTS = {
    '18-25': 0.15,   # Higher risk for young adults
    '26-35': 0.05,   # Slightly higher risk
    '36-50': 0.00,   # Baseline risk
    '51-65': -0.05,  # Lower risk
    '65+': 0.10      # Higher risk due to fixed income
}

# Employment status risk multipliers
EMPLOYMENT_RISK_MULTIPLIERS = {
    1: 1.0,  # Full-time employed - baseline
    2: 1.2,  # Part-time employed - 20% higher risk
    3: 1.5,  # Self-employed - 50% higher risk
    4: 2.0   # Unemployed - 100% higher risk
}

# Education level risk adjustments
EDUCATION_RISK_ADJUSTMENTS = {
    1: 0.20,   # Below high school - higher risk
    2: 0.10,   # High school - moderate risk
    3: 0.00,   # College/Bachelor - baseline
    4: -0.05,  # University - lower risk
    5: -0.10   # Graduate degree - lowest risk
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Default model hyperparameters
# OPTIMIZED via W&B Sweep (ethereal-sweep-27, val_AUC=0.79814)
# Sweep date: 2025-10-21
# Features: use_geo_risk=True, use_smote=False, use_target_encoding=False
MODEL_PARAMS = {
    'XGBoost': {
        'n_estimators': 200,           # Optimized by sweep
        'max_depth': 5,                # Optimized: balanced tree depth
        'learning_rate': 0.024591,     # Optimized: fine-tuned learning rate
        'subsample': 0.8852,           # Optimized: ~89% row sampling
        'colsample_bytree': 0.7098,    # Optimized: ~71% feature sampling
        'min_child_weight': 1,         # Optimized: lower regularization
        'reg_alpha': 0.1,              # L1 regularization
        'reg_lambda': 1.0,             # L2 regularization
        'scale_pos_weight': 14.106,    # Optimized: class imbalance handling
        'gamma': 0.1,                  # Minimum loss reduction
        'random_state': 42,
        'eval_metric': 'auc',          # Optimize for AUC
        'verbosity': 0
    },
    'LightGBM': {
        'n_estimators': 300,           # Optimized by sweep
        'max_depth': 5,                # Optimized: balanced tree depth
        'learning_rate': 0.0903,       # Optimized: fine-tuned learning rate
        'subsample': 0.8,              # Bagging fraction
        'colsample_bytree': 0.8,       # Feature fraction
        'min_child_samples': 10,       # Allow smaller leaves
        'reg_alpha': 0.1,              # L1 regularization
        'reg_lambda': 1.0,             # L2 regularization
        'num_leaves': 127,             # Optimized: higher complexity
        'min_split_gain': 0.01,        # Minimum gain to split
        'is_unbalance': True,          # Handle class imbalance
        'random_state': 42,
        'verbosity': -1
    },
    'CatBoost': {
        'iterations': 300,             # More iterations
        'depth': 6,                    # Deeper trees
        'learning_rate': 0.05,         # Lower LR
        'l2_leaf_reg': 3,              # L2 regularization
        'border_count': 128,           # More borders for numerical features
        'auto_class_weights': 'Balanced',  # Auto balance classes
        'min_data_in_leaf': 10,        # Allow smaller leaves
        'random_state': 42,
        'verbose': False
    }
}

# Hyperparameter search spaces for optimization
PARAM_SEARCH_SPACES = {
    'XGBoost': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    },
    'LightGBM': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_samples': [10, 20, 30, 50],
        'reg_alpha': [0, 0.1, 0.5, 1.0],
        'reg_lambda': [0, 0.1, 0.5, 1.0]
    },
    'CatBoost': {
        'iterations': [100, 200, 300, 500],
        'depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7, 10],
        'border_count': [32, 64, 128, 255]
    }
}

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Feature selection criteria
FEATURE_SELECTION = {
    'correlation_threshold': 0.95,  # Remove highly correlated features
    'variance_threshold': 0.01,     # Remove low variance features
    'importance_threshold': 0.001,  # Minimum feature importance
    'max_features': 50              # Maximum number of features to keep
}

# Outlier detection parameters
OUTLIER_DETECTION = {
    'method': 'IQR',  # 'IQR' or 'zscore'
    'iqr_multiplier': 1.5,
    'zscore_threshold': 3.0,
    'cap_outliers': True  # Whether to cap outliers or remove them
}

# Missing value handling
MISSING_VALUE_HANDLING = {
    'numerical_strategy': 'knn',  # 'mean', 'median', 'knn'
    'categorical_strategy': 'mode',  # 'mode', 'constant'
    'knn_neighbors': 5,
    'missing_threshold': 0.5  # Drop features with >50% missing values
}

# Feature scaling parameters
SCALING_CONFIG = {
    'numerical_scaler': 'robust',  # 'standard', 'robust', 'minmax'
    'categorical_encoding': 'onehot'  # 'onehot', 'label', 'target'
}

# =============================================================================
# BUSINESS RULES
# =============================================================================

# Minimum requirements for loan approval
MINIMUM_REQUIREMENTS = {
    'min_age': 18,
    'max_age': 75,
    'min_income': 20000,
    'max_debt_to_income': 0.5,
    'min_credit_history_months': 6,
    'max_late_payments': 3,
    'max_missed_payments': 1
}

# Conditional approval criteria
CONDITIONAL_CRITERIA = {
    'high_income_threshold': 100000,  # Higher income can offset some risks
    'excellent_payment_history': 0.95,  # 95%+ on-time payments
    'low_utilization': 0.3,  # <30% credit utilization
    'stable_employment_months': 24  # 2+ years in current job
}

# Automatic rejection criteria
AUTO_REJECT_CRITERIA = {
    'bankruptcy_in_years': 2,
    'foreclosure_in_years': 3,
    'max_consecutive_late_payments': 6,
    'debt_to_income_threshold': 0.6,
    'no_credit_history': False  # Can't have zero credit history
}

# =============================================================================
# MONITORING AND ALERTS
# =============================================================================

# Model performance monitoring thresholds
MONITORING_THRESHOLDS = {
    'min_auc_roc': 0.75,        # Alert if AUC drops below 75%
    'min_accuracy': 0.80,       # Alert if accuracy drops below 80%
    'max_log_loss': 0.5,        # Alert if log loss exceeds 0.5
    'default_rate_deviation': 0.05,  # Alert if default rate deviates by 5%
    'prediction_drift_threshold': 0.1  # Statistical drift threshold
}

# Data quality checks
DATA_QUALITY_CHECKS = {
    'max_missing_rate': 0.1,     # Alert if >10% missing values
    'min_sample_size': 1000,     # Minimum samples for model training
    'feature_drift_threshold': 0.1,  # Statistical feature drift
    'target_imbalance_threshold': 0.05  # Min proportion of positive class
}

# =============================================================================
# REPORTING CONFIGURATION
# =============================================================================

# Performance metrics to track
PERFORMANCE_METRICS = [
    'accuracy', 'precision', 'recall', 'f1_score',
    'auc_roc', 'auc_pr', 'log_loss', 'brier_score'
]

# Business metrics to monitor
BUSINESS_METRICS = [
    'approval_rate', 'default_rate', 'profit_per_client',
    'portfolio_risk', 'grade_migration', 'concentration_risk'
]

# Reporting frequency
REPORTING_CONFIG = {
    'daily_metrics': ['approval_rate', 'application_volume'],
    'weekly_metrics': ['default_rate', 'grade_distribution'],
    'monthly_metrics': ['model_performance', 'portfolio_analysis'],
    'quarterly_metrics': ['model_validation', 'stress_testing']
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_risk_score_grade(risk_score):
    """
    Get risk grade based on risk score

    Args:
        risk_score (int): Risk score (0-100)

    Returns:
        str: Risk grade
    """
    for grade, params in RISK_SCORE_RANGES.items():
        if params['min_score'] <= risk_score <= params['max_score']:
            return grade
    return 'BAD'  # Default to highest risk

def classify_client(default_probability):
    """
    Classify client as Good/Bad based on default probability

    Args:
        default_probability (float): Default probability (0-1)

    Returns:
        str: Client classification ('Good' or 'Bad')
    """
    return 'Bad' if default_probability >= DEFAULT_PROBABILITY_THRESHOLD else 'Good'

def calculate_risk_adjusted_score(base_score, age, employment_status, education_level):
    """
    Calculate risk-adjusted credit score

    Args:
        base_score (float): Base credit score
        age (int): Client age
        employment_status (int): Employment status code
        education_level (int): Education level code

    Returns:
        float: Risk-adjusted score
    """
    # Age adjustment
    if age <= 25:
        age_adj = AGE_RISK_ADJUSTMENTS['18-25']
    elif age <= 35:
        age_adj = AGE_RISK_ADJUSTMENTS['26-35']
    elif age <= 50:
        age_adj = AGE_RISK_ADJUSTMENTS['36-50']
    elif age <= 65:
        age_adj = AGE_RISK_ADJUSTMENTS['51-65']
    else:
        age_adj = AGE_RISK_ADJUSTMENTS['65+']

    # Employment adjustment
    emp_multiplier = EMPLOYMENT_RISK_MULTIPLIERS.get(employment_status, 1.0)

    # Education adjustment
    edu_adj = EDUCATION_RISK_ADJUSTMENTS.get(education_level, 0.0)

    # Calculate adjusted score
    adjusted_score = base_score * emp_multiplier + (age_adj + edu_adj) * 100

    # Ensure score stays within valid range
    return max(SCORE_RANGE['min_score'],
              min(SCORE_RANGE['max_score'], adjusted_score))

def validate_client_data(client_data):
    """
    Validate client data against minimum requirements

    Args:
        client_data (dict): Client information

    Returns:
        tuple: (is_valid, error_messages)
    """
    errors = []

    # Check minimum requirements
    if client_data.get('Age', 0) < MINIMUM_REQUIREMENTS['min_age']:
        errors.append(f"Age must be at least {MINIMUM_REQUIREMENTS['min_age']}")

    if client_data.get('Age', 0) > MINIMUM_REQUIREMENTS['max_age']:
        errors.append(f"Age must be at most {MINIMUM_REQUIREMENTS['max_age']}")

    if client_data.get('Annual_Income', 0) < MINIMUM_REQUIREMENTS['min_income']:
        errors.append(f"Income must be at least {MINIMUM_REQUIREMENTS['min_income']}")

    debt_to_income = client_data.get('Debt_to_Income_Ratio', 0)
    if debt_to_income > MINIMUM_REQUIREMENTS['max_debt_to_income']:
        errors.append(f"Debt-to-income ratio must be at most {MINIMUM_REQUIREMENTS['max_debt_to_income']}")

    credit_months = client_data.get('Credit_History_Months', 0)
    if credit_months < MINIMUM_REQUIREMENTS['min_credit_history_months']:
        errors.append(f"Credit history must be at least {MINIMUM_REQUIREMENTS['min_credit_history_months']} months")

    return len(errors) == 0, errors

# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

__all__ = [
    'RISK_SCORE_RANGES',
    'DEFAULT_PROBABILITY_THRESHOLD',
    'SCORE_RANGE',
    'BASIC_SCORE_WEIGHTS',
    'MODEL_PARAMS',
    'PARAM_SEARCH_SPACES',
    'FEATURE_SELECTION',
    'MINIMUM_REQUIREMENTS',
    'get_risk_score_grade',
    'classify_client',
    'calculate_risk_adjusted_score',
    'validate_client_data'
]