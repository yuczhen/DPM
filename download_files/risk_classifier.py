"""
Risk Classification Module
Converts risk scores to A-B-C categories and provides risk assessment
"""

import numpy as np
import pandas as pd
from config import RISK_THRESHOLDS

class RiskClassifier:
    def __init__(self, thresholds=None):
        """
        Initialize risk classifier with custom thresholds if provided
        """
        self.thresholds = thresholds or RISK_THRESHOLDS
        
    def score_to_category(self, scores):
        """
        Convert risk scores to A-B-C categories
        
        Args:
            scores: Array-like of risk scores (0-1000)
            
        Returns:
            Array of risk categories ('A', 'B', 'C')
        """
        scores = np.array(scores)
        categories = np.full(scores.shape, 'C', dtype='U1')
        
        categories[(scores >= self.thresholds['A'][0]) & (scores <= self.thresholds['A'][1])] = 'A'
        categories[(scores >= self.thresholds['B'][0]) & (scores <= self.thresholds['B'][1])] = 'B'
        categories[scores >= self.thresholds['C'][0]] = 'C'
        
        return categories
    
    def probability_to_score(self, probabilities):
        """
        Convert default probabilities to risk scores (0-1000)
        
        Args:
            probabilities: Array-like of default probabilities (0-1)
            
        Returns:
            Array of risk scores (0-1000)
        """
        probabilities = np.array(probabilities)
        # Scale probabilities to 0-1000 range
        scores = probabilities * 1000
        return np.clip(scores, 0, 1000)
    
    def get_risk_description(self, category):
        """
        Get detailed risk description for each category
        
        Args:
            category: Risk category ('A', 'B', or 'C')
            
        Returns:
            Dictionary with risk description details
        """
        descriptions = {
            'A': {
                'risk_level': 'Low Risk',
                'description': 'Excellent creditworthiness with very low default probability',
                'recommendation': 'Approve loan with standard terms',
                'interest_rate_modifier': 0.0,  # No additional risk premium
                'monitoring_frequency': 'Annual',
                'default_probability': '< 5%'
            },
            'B': {
                'risk_level': 'Medium Risk', 
                'description': 'Good creditworthiness with moderate risk profile',
                'recommendation': 'Approve with standard monitoring',
                'interest_rate_modifier': 1.5,  # Additional 1.5% risk premium
                'monitoring_frequency': 'Semi-annual',
                'default_probability': '5-15%'
            },
            'C': {
                'risk_level': 'High Risk',
                'description': 'Elevated default risk requiring careful evaluation',
                'recommendation': 'Requires additional documentation and higher interest rates',
                'interest_rate_modifier': 3.0,  # Additional 3% risk premium
                'monitoring_frequency': 'Quarterly',
                'default_probability': '> 15%'
            }
        }
        
        return descriptions.get(category, descriptions['C'])
    
    def generate_risk_report(self, scores, client_ids=None):
        """
        Generate comprehensive risk report for multiple clients
        
        Args:
            scores: Array-like of risk scores
            client_ids: Optional array of client identifiers
            
        Returns:
            DataFrame with detailed risk assessment
        """
        scores = np.array(scores)
        categories = self.score_to_category(scores)
        
        if client_ids is None:
            client_ids = [f"Client_{i+1}" for i in range(len(scores))]
        
        report_data = []
        for i, (client_id, score, category) in enumerate(zip(client_ids, scores, categories)):
            risk_info = self.get_risk_description(category)
            
            report_data.append({
                'client_id': client_id,
                'risk_score': score,
                'risk_category': category,
                'risk_level': risk_info['risk_level'],
                'description': risk_info['description'],
                'recommendation': risk_info['recommendation'],
                'interest_rate_modifier': risk_info['interest_rate_modifier'],
                'monitoring_frequency': risk_info['monitoring_frequency'],
                'default_probability': risk_info['default_probability']
            })
        
        return pd.DataFrame(report_data)
    
    def get_portfolio_summary(self, scores):
        """
        Get summary statistics for a portfolio of clients
        
        Args:
            scores: Array-like of risk scores
            
        Returns:
            Dictionary with portfolio statistics
        """
        scores = np.array(scores)
        categories = self.score_to_category(scores)
        
        category_counts = pd.Series(categories).value_counts()
        total_clients = len(scores)
        
        summary = {
            'total_clients': total_clients,
            'average_risk_score': np.mean(scores),
            'median_risk_score': np.median(scores),
            'risk_distribution': {
                'A_count': category_counts.get('A', 0),
                'B_count': category_counts.get('B', 0),
                'C_count': category_counts.get('C', 0),
                'A_percentage': (category_counts.get('A', 0) / total_clients) * 100,
                'B_percentage': (category_counts.get('B', 0) / total_clients) * 100,
                'C_percentage': (category_counts.get('C', 0) / total_clients) * 100
            },
            'portfolio_risk_level': self._assess_portfolio_risk(scores)
        }
        
        return summary
    
    def _assess_portfolio_risk(self, scores):
        """
        Assess overall portfolio risk level
        """
        avg_score = np.mean(scores)
        
        if avg_score <= 300:
            return "Low Risk Portfolio"
        elif avg_score <= 600:
            return "Moderate Risk Portfolio"
        else:
            return "High Risk Portfolio"