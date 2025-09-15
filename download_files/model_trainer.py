"""
Model Training Module
Handles training and optimization of different ML models for credit evaluation
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIGS

class ModelTrainer:
    def __init__(self, model_type='xgboost'):
        """
        Initialize model trainer with specified model type
        
        Args:
            model_type: Type of model ('logistic_regression', 'random_forest', 'xgboost', 'lightgbm')
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.feature_importance_ = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model with default parameters"""
        config = MODEL_CONFIGS.get(self.model_type, MODEL_CONFIGS['xgboost'])
        
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(**config)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(**config)
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(**config)
        elif self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(**config)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on training data
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        """
        print(f"Training {self.model_type} model...")
        
        # Special handling for gradient boosting models with validation
        if self.model_type in ['xgboost', 'lightgbm'] and X_val is not None:
            if self.model_type == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:  # lightgbm
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
                )
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        self._extract_feature_importance(X_train)
        print("Model training completed!")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Features to predict on
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)[:, 1]  # Return probability of positive class
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Get predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'roc_auc': roc_auc,
            'accuracy': class_report['accuracy'],
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score'],
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        if not self.is_trained:
            # Use a fresh model for cross-validation
            temp_model = self._get_fresh_model()
        else:
            temp_model = self.model
        
        cv_scores = {
            'accuracy': cross_val_score(temp_model, X, y, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(temp_model, X, y, cv=cv, scoring='precision'),
            'recall': cross_val_score(temp_model, X, y, cv=cv, scoring='recall'),
            'f1': cross_val_score(temp_model, X, y, cv=cv, scoring='f1'),
            'roc_auc': cross_val_score(temp_model, X, y, cv=cv, scoring='roc_auc')
        }
        
        # Calculate mean and std for each metric
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[f'{metric}_mean'] = np.mean(scores)
            cv_results[f'{metric}_std'] = np.std(scores)
        
        return cv_results
    
    def optimize_hyperparameters(self, X_train, y_train, param_grid=None, cv=3):
        """
        Optimize model hyperparameters using grid search
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for grid search
            cv: Number of cross-validation folds
            
        Returns:
            Best parameters found
        """
        if param_grid is None:
            param_grid = self._get_default_param_grid()
        
        print(f"Optimizing hyperparameters for {self.model_type}...")
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        self._extract_feature_importance(X_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from trained model
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained or self.feature_importance_ is None:
            return None
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importance_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'feature_importance': self.feature_importance_
        }, filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_importance_ = model_data.get('feature_importance')
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def _extract_feature_importance(self, X):
        """Extract feature importance from trained model"""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            self.feature_importance_ = np.abs(self.model.coef_[0])
        else:
            self.feature_importance_ = None
    
    def _get_fresh_model(self):
        """Get a fresh instance of the current model type"""
        config = MODEL_CONFIGS.get(self.model_type, MODEL_CONFIGS['xgboost'])
        
        if self.model_type == 'logistic_regression':
            return LogisticRegression(**config)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(**config)
        elif self.model_type == 'xgboost':
            return xgb.XGBClassifier(**config)
        elif self.model_type == 'lightgbm':
            return lgb.LGBMClassifier(**config)
    
    def _get_default_param_grid(self):
        """Get default parameter grid for hyperparameter optimization"""
        if self.model_type == 'logistic_regression':
            return {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [500, 1000]
            }
        elif self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.model_type == 'xgboost':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        elif self.model_type == 'lightgbm':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        return {}