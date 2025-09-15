"""
Evaluation Metrics Module
Provides comprehensive model evaluation and visualization tools
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class EvaluationMetrics:
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_model(self, y_true, y_pred, y_proba, model_name="Model"):
        """
        Comprehensive model evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_proba: Prediction probabilities
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with all evaluation metrics
        """
        # Basic metrics
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # ROC metrics
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall metrics
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        
        # Calculate additional metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        
        metrics = {
            'model_name': model_name,
            'accuracy': class_report['accuracy'],
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score'],
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': conf_matrix,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision,
            'recall_curve': recall
        }
        
        # Store evaluation history
        self.evaluation_history.append(metrics)
        
        return metrics
    
    def plot_confusion_matrix(self, conf_matrix, model_name="Model", save_path=None):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Low Risk', 'High Risk'],
                   yticklabels=['Low Risk', 'High Risk'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, model_name="Model", save_path=None):
        """
        Plot ROC curve
        """
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curve(self, precision, recall, pr_auc, model_name="Model", save_path=None):
        """
        Plot Precision-Recall curve
        """
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, importance_df, top_n=15, save_path=None):
        """
        Plot feature importance
        """
        if importance_df is None or len(importance_df) == 0:
            print("No feature importance data available")
            return
        
        # Get top N features
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, metrics, importance_df=None):
        """
        Create interactive dashboard with Plotly
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve', 'Feature Importance'),
            specs=[[{"type": "heatmap"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Confusion Matrix
        conf_matrix = metrics['confusion_matrix']
        fig.add_trace(
            go.Heatmap(
                z=conf_matrix,
                x=['Low Risk', 'High Risk'],
                y=['Low Risk', 'High Risk'],
                colorscale='Blues',
                showscale=False,
                text=conf_matrix,
                texttemplate="%{text}",
                textfont={"size": 20}
            ),
            row=1, col=1
        )
        
        # ROC Curve
        fig.add_trace(
            go.Scatter(
                x=metrics['fpr'],
                y=metrics['tpr'],
                mode='lines',
                name=f"ROC (AUC={metrics['roc_auc']:.3f})",
                line=dict(color='orange', width=3)
            ),
            row=1, col=2
        )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(dash='dash', color='gray'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Precision-Recall Curve
        fig.add_trace(
            go.Scatter(
                x=metrics['recall_curve'],
                y=metrics['precision_curve'],
                mode='lines',
                name=f"PR (AUC={metrics['pr_auc']:.3f})",
                line=dict(color='blue', width=3)
            ),
            row=2, col=1
        )
        
        # Feature Importance
        if importance_df is not None and len(importance_df) > 0:
            top_features = importance_df.head(10)
            fig.add_trace(
                go.Bar(
                    x=top_features['importance'],
                    y=top_features['feature'],
                    orientation='h',
                    name='Feature Importance'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=f"Model Evaluation Dashboard - {metrics['model_name']}",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def generate_evaluation_report(self, metrics, importance_df=None):
        """
        Generate comprehensive evaluation report
        """
        report = f"""
# Model Evaluation Report: {metrics['model_name']}

## Performance Metrics
- **Accuracy**: {metrics['accuracy']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1-Score**: {metrics['f1_score']:.4f}
- **Specificity**: {metrics['specificity']:.4f}
- **ROC AUC**: {metrics['roc_auc']:.4f}
- **PR AUC**: {metrics['pr_auc']:.4f}

## Confusion Matrix Analysis
- **True Positives**: {metrics['true_positives']}
- **False Positives**: {metrics['false_positives']}
- **True Negatives**: {metrics['true_negatives']}
- **False Negatives**: {metrics['false_negatives']}

## Business Impact Analysis
- **Type I Error Rate** (False Positives): {metrics['false_positives'] / (metrics['false_positives'] + metrics['true_negatives']):.4f}
  - Impact: Rejecting good customers (lost revenue)
- **Type II Error Rate** (False Negatives): {metrics['false_negatives'] / (metrics['false_negatives'] + metrics['true_positives']):.4f}
  - Impact: Approving bad customers (potential losses)

## Model Interpretation
"""
        
        if importance_df is not None and len(importance_df) > 0:
            report += f"""
### Top 5 Most Important Features:
"""
            for i, row in importance_df.head(5).iterrows():
                report += f"- **{row['feature']}**: {row['importance']:.4f}\n"
        
        report += f"""
## Recommendations
- Model shows {'good' if metrics['roc_auc'] > 0.8 else 'moderate' if metrics['roc_auc'] > 0.7 else 'poor'} discriminative ability (ROC AUC = {metrics['roc_auc']:.3f})
- {'Consider deployment' if metrics['roc_auc'] > 0.75 else 'Requires improvement before deployment'}
- Monitor performance regularly and retrain as needed
"""
        
        return report
    
    def compare_models(self, save_path=None):
        """
        Compare multiple models from evaluation history
        """
        if len(self.evaluation_history) < 2:
            print("Need at least 2 models to compare")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        for metrics in self.evaluation_history:
            comparison_data.append({
                'Model': metrics['model_name'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC AUC': metrics['roc_auc'],
                'PR AUC': metrics['pr_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'PR AUC']
        
        for i, metric in enumerate(metrics_to_plot):
            sns.barplot(data=comparison_df, x='Model', y=metric, ax=axes[i])
            axes[i].set_title(f'{metric} Comparison')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df