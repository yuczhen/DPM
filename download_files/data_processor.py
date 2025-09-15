"""
Data Processing Module for Credit Evaluation System
Handles data loading, preprocessing, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        
    def load_sample_data(self, n_samples=1000):
        """
        Generate sample credit data for demonstration
        In production, replace this with your actual data loading logic
        """
        np.random.seed(42)
        
        # Generate synthetic credit data
        data = {
            'age': np.random.normal(40, 12, n_samples).clip(18, 80),
            'annual_income': np.random.lognormal(10.5, 0.5, n_samples).clip(20000, 200000),
            'credit_score': np.random.normal(650, 100, n_samples).clip(300, 850),
            'debt_to_income': np.random.beta(2, 5, n_samples),
            'credit_history_length': np.random.exponential(8, n_samples).clip(0, 30),
            'num_credit_accounts': np.random.poisson(5, n_samples).clip(0, 20),
            'credit_utilization': np.random.beta(2, 3, n_samples),
            'payment_history_score': np.random.normal(0.85, 0.15, n_samples).clip(0, 1),
            'num_late_payments': np.random.poisson(2, n_samples).clip(0, 20),
            'employment_length': np.random.exponential(5, n_samples).clip(0, 40),
            'home_ownership': np.random.choice(['rent', 'own', 'mortgage'], n_samples, p=[0.3, 0.4, 0.3]),
            'loan_purpose': np.random.choice(['debt_consolidation', 'home_improvement', 'major_purchase', 'other'], 
                                           n_samples, p=[0.4, 0.2, 0.2, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable (default risk)
        # Higher risk based on lower credit score, higher debt-to-income, etc.
        risk_score = (
            (850 - df['credit_score']) / 550 * 0.3 +
            df['debt_to_income'] * 0.25 +
            df['credit_utilization'] * 0.2 +
            (1 - df['payment_history_score']) * 0.15 +
            df['num_late_payments'] / 20 * 0.1
        )
        
        # Convert to binary classification (1 = high risk, 0 = low risk)
        df['default_risk'] = (risk_score > 0.4).astype(int)
        
        return df
    
    def engineer_features(self, df):
        """
        Create additional features for better model performance
        """
        df = df.copy()
        
        # Financial ratios
        df['income_to_credit_ratio'] = df['annual_income'] / (df['credit_score'] + 1)
        df['debt_burden'] = df['debt_to_income'] * df['annual_income']
        df['credit_per_account'] = df['credit_score'] / (df['num_credit_accounts'] + 1)
        
        # Risk indicators
        df['high_utilization'] = (df['credit_utilization'] > 0.8).astype(int)
        df['young_borrower'] = (df['age'] < 25).astype(int)
        df['experienced_borrower'] = (df['credit_history_length'] > 10).astype(int)
        
        # Interaction features
        df['age_income_interaction'] = df['age'] * df['annual_income'] / 100000
        df['credit_utilization_score'] = df['credit_utilization'] * df['credit_score']
        
        # Stability indicators
        df['employment_stability'] = (df['employment_length'] > 2).astype(int)
        df['payment_reliability'] = (df['payment_history_score'] > 0.9).astype(int)
        
        return df
    
    def preprocess_data(self, df, target_column='default_risk', fit_transformers=True):
        """
        Preprocess the data for model training
        """
        df = df.copy()
        
        # Separate features and target
        if target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df
            y = None
        
        # Engineer features
        X = self.engineer_features(X)
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if fit_transformers:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    le = self.label_encoders[col]
                    X[col] = X[col].astype(str)
                    mask = X[col].isin(le.classes_)
                    X[col] = X[col].where(mask, le.classes_[0])  # Replace unseen with first class
                    X[col] = le.transform(X[col])
        
        # Handle missing values
        if fit_transformers:
            X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns, index=X.index)
        else:
            X = pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)
        
        # Scale features
        if fit_transformers:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = list(X.columns)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    def get_feature_importance_names(self):
        """
        Return feature names for importance analysis
        """
        return self.feature_names