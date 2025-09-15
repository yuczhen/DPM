"""
Main Credit Evaluator Class
Integrates all components for comprehensive credit risk evaluation
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from risk_classifier import RiskClassifier
from evaluation_metrics import EvaluationMetrics

class CreditEvaluator:
    def __init__(self, model_type='xgboost'):
        """
        Initialize Credit Evaluator
        
        Args:
            model_type: Type of ML model to use ('logistic_regression', 'random_forest', 'xgboost', 'lightgbm')
        """
        self.model_type = model_type
        self.data_processor = DataProcessor()
        self.model_trainer = ModelTrainer(model_type)
        self.risk_classifier = RiskClassifier()
        self.evaluator = EvaluationMetrics()
        
        self.is_fitted = False
        self.training_metrics = None
        
    def fit(self, X, y, validation_split=0.2, optimize_hyperparameters=False):
        """
        Train the credit evaluation model
        
        Args:
            X: Training features (DataFrame or array)
            y: Training labels
            validation_split: Proportion of data to use for validation
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            
        Returns:
            Training metrics
        """
        print("Starting credit evaluation model training...")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Preprocess data
        X_processed, y_processed = self.data_processor.preprocess_data(
            pd.concat([X, pd.Series(y, name='default_risk')], axis=1),
            target_column='default_risk',
            fit_transformers=True
        )
        
        # Split data
        if validation_split > 0:
            X_train, X_val, y_train, y_val = self.data_processor.split_data(
                X_processed, y_processed, test_size=validation_split
            )
        else:
            X_train, y_train = X_processed, y_processed
            X_val, y_val = None, None
        
        # Optimize hyperparameters if requested
        if optimize_hyperparameters:
            print("Optimizing hyperparameters...")
            self.model_trainer.optimize_hyperparameters(X_train, y_train)
        else:
            # Train model
            self.model_trainer.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on training data
        if X_val is not None:
            y_pred = self.model_trainer.predict(X_val)
            y_proba = self.model_trainer.predict_proba(X_val)
            
            self.training_metrics = self.evaluator.evaluate_model(
                y_val, y_pred, y_proba, f"{self.model_type}_validation"
            )
            
            print(f"Validation ROC AUC: {self.training_metrics['roc_auc']:.4f}")
            print(f"Validation Accuracy: {self.training_metrics['accuracy']:.4f}")
        
        self.is_fitted = True
        print("Model training completed successfully!")
        
        return self.training_metrics
    
    def predict_risk(self, X, return_probabilities=False):
        """
        Predict credit risk for new clients
        
        Args:
            X: Client features (DataFrame or array)
            return_probabilities: Whether to return raw probabilities
            
        Returns:
            Tuple of (risk_scores, risk_categories) or (risk_scores, risk_categories, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Preprocess data
        X_processed, _ = self.data_processor.preprocess_data(X, fit_transformers=False)
        
        # Get predictions
        probabilities = self.model_trainer.predict_proba(X_processed)
        
        # Convert to risk scores (0-1000)
        risk_scores = self.risk_classifier.probability_to_score(probabilities)
        
        # Get A-B-C categories
        risk_categories = self.risk_classifier.score_to_category(risk_scores)
        
        if return_probabilities:
            return risk_scores, risk_categories, probabilities
        else:
            return risk_scores, risk_categories
    
    def evaluate_portfolio(self, X, y=None, client_ids=None):
        """
        Comprehensive portfolio evaluation
        
        Args:
            X: Client features
            y: True labels (optional, for model evaluation)
            client_ids: Client identifiers (optional)
            
        Returns:
            Dictionary with portfolio analysis results
        """
        # Get risk predictions
        risk_scores, risk_categories, probabilities = self.predict_risk(X, return_probabilities=True)
        
        # Generate risk report
        risk_report = self.risk_classifier.generate_risk_report(risk_scores, client_ids)
        
        # Get portfolio summary
        portfolio_summary = self.risk_classifier.get_portfolio_summary(risk_scores)
        
        results = {
            'risk_report': risk_report,
            'portfolio_summary': portfolio_summary,
            'risk_scores': risk_scores,
            'risk_categories': risk_categories,
            'probabilities': probabilities
        }
        
        # If true labels provided, evaluate model performance
        if y is not None:
            y_pred = (probabilities > 0.5).astype(int)
            evaluation_metrics = self.evaluator.evaluate_model(
                y, y_pred, probabilities, f"{self.model_type}_portfolio"
            )
            results['evaluation_metrics'] = evaluation_metrics
        
        return results
    
    def get_feature_importance(self):
        """
        Get feature importance from trained model
        
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        feature_names = self.data_processor.get_feature_importance_names()
        return self.model_trainer.get_feature_importance(feature_names)
    
    def generate_client_report(self, client_data, client_id="Client"):
        """
        Generate detailed report for a single client
        
        Args:
            client_data: Client features (Series or single-row DataFrame)
            client_id: Client identifier
            
        Returns:
            Dictionary with detailed client analysis
        """
        # Ensure client_data is DataFrame
        if isinstance(client_data, pd.Series):
            client_data = client_data.to_frame().T
        elif len(client_data.shape) == 1:
            client_data = pd.DataFrame([client_data])
        
        # Get predictions
        risk_scores, risk_categories, probabilities = self.predict_risk(
            client_data, return_probabilities=True
        )
        
        risk_score = risk_scores[0]
        risk_category = risk_categories[0]
        probability = probabilities[0]
        
        # Get risk description
        risk_info = self.risk_classifier.get_risk_description(risk_category)
        
        # Analyze key risk factors
        feature_importance = self.get_feature_importance()
        
        client_report = {
            'client_id': client_id,
            'risk_score': risk_score,
            'risk_category': risk_category,
            'default_probability': probability,
            'risk_level': risk_info['risk_level'],
            'description': risk_info['description'],
            'recommendation': risk_info['recommendation'],
            'interest_rate_modifier': risk_info['interest_rate_modifier'],
            'monitoring_frequency': risk_info['monitoring_frequency'],
            'key_risk_factors': feature_importance.head(5) if feature_importance is not None else None
        }
        
        return client_report
    
    def save_model(self, filepath):
        """Save the complete model pipeline"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        import joblib
        
        model_data = {
            'model_trainer': self.model_trainer,
            'data_processor': self.data_processor,
            'risk_classifier': self.risk_classifier,
            'model_type': self.model_type,
            'training_metrics': self.training_metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"Complete model pipeline saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a complete model pipeline"""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model_trainer = model_data['model_trainer']
        self.data_processor = model_data['data_processor']
        self.risk_classifier = model_data['risk_classifier']
        self.model_type = model_data['model_type']
        self.training_metrics = model_data.get('training_metrics')
        self.is_fitted = True
        
        print(f"Complete model pipeline loaded from {filepath}")
    
    def create_evaluation_dashboard(self, X_test=None, y_test=None):
        """
        Create interactive evaluation dashboard
        
        Args:
            X_test: Test features (optional)
            y_test: Test labels (optional)
            
        Returns:
            Plotly figure with interactive dashboard
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before creating dashboard")
        
        # Use training metrics if no test data provided
        if X_test is not None and y_test is not None:
            risk_scores, risk_categories, probabilities = self.predict_risk(X_test, return_probabilities=True)
            y_pred = (probabilities > 0.5).astype(int)
            
            metrics = self.evaluator.evaluate_model(
                y_test, y_pred, probabilities, f"{self.model_type}_test"
            )
        else:
            metrics = self.training_metrics
        
        if metrics is None:
            raise ValueError("No evaluation metrics available")
        
        # Get feature importance
        feature_importance = self.get_feature_importance()
        
        # Create dashboard
        dashboard = self.evaluator.create_interactive_dashboard(metrics, feature_importance)
        
        return dashboard