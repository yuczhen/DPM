"""
Script to create a Jupyter notebook for credit evaluation analysis
Run this after setting up your local environment
"""

import json

def create_jupyter_notebook():
    """Create a comprehensive Jupyter notebook for credit analysis"""
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# A-B-C Credit Evaluation System - Analysis Notebook\n",
                    "\n",
                    "This notebook demonstrates how to use the credit evaluation system for loan risk assessment.\n",
                    "\n",
                    "## Overview\n",
                    "- **A (Low Risk)**: Score 0-300 - Excellent creditworthiness\n",
                    "- **B (Medium Risk)**: Score 301-600 - Moderate risk profile\n",
                    "- **C (High Risk)**: Score 601-1000 - Elevated default risk"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Import required libraries\n",
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Import our credit evaluation system\n",
                    "from credit_evaluator import CreditEvaluator\n",
                    "from data_processor import DataProcessor\n",
                    "from risk_classifier import RiskClassifier\n",
                    "\n",
                    "# Set up plotting\n",
                    "plt.style.use('default')\n",
                    "sns.set_palette(\"husl\")\n",
                    "%matplotlib inline\n",
                    "\n",
                    "print(\"✅ Libraries imported successfully!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load and Explore Data"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Initialize data processor\n",
                    "processor = DataProcessor()\n",
                    "\n",
                    "# Load sample data (replace with your actual data loading)\n",
                    "df = processor.load_sample_data(n_samples=2000)\n",
                    "\n",
                    "print(f\"Dataset shape: {df.shape}\")\n",
                    "print(f\"\\nColumns: {list(df.columns)}\")\n",
                    "print(f\"\\nDefault rate: {df['default_risk'].mean():.2%}\")\n",
                    "\n",
                    "# Display first few rows\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Data exploration\n",
                    "fig, axes = plt.subplots(2, 3, figsize=(15, 10))\n",
                    "axes = axes.ravel()\n",
                    "\n",
                    "# Plot key features\n",
                    "features_to_plot = ['age', 'annual_income', 'credit_score', 'debt_to_income', 'credit_utilization', 'payment_history_score']\n",
                    "\n",
                    "for i, feature in enumerate(features_to_plot):\n",
                    "    axes[i].hist(df[feature], bins=30, alpha=0.7)\n",
                    "    axes[i].set_title(f'Distribution of {feature}')\n",
                    "    axes[i].set_xlabel(feature)\n",
                    "    axes[i].set_ylabel('Frequency')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Train Credit Evaluation Models"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Prepare data\n",
                    "X = df.drop('default_risk', axis=1)\n",
                    "y = df['default_risk']\n",
                    "\n",
                    "print(f\"Features shape: {X.shape}\")\n",
                    "print(f\"Target shape: {y.shape}\")\n",
                    "print(f\"Target distribution:\")\n",
                    "print(y.value_counts(normalize=True))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Train different models and compare performance\n",
                    "models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']\n",
                    "results = {}\n",
                    "\n",
                    "for model_type in models:\n",
                    "    print(f\"\\nTraining {model_type}...\")\n",
                    "    \n",
                    "    # Initialize evaluator\n",
                    "    evaluator = CreditEvaluator(model_type=model_type)\n",
                    "    \n",
                    "    # Train model\n",
                    "    metrics = evaluator.fit(X, y, validation_split=0.2)\n",
                    "    \n",
                    "    if metrics:\n",
                    "        results[model_type] = {\n",
                    "            'ROC AUC': metrics['roc_auc'],\n",
                    "            'Accuracy': metrics['accuracy'],\n",
                    "            'Precision': metrics['precision'],\n",
                    "            'Recall': metrics['recall'],\n",
                    "            'F1-Score': metrics['f1_score']\n",
                    "        }\n",
                    "        print(f\"✅ {model_type}: ROC AUC = {metrics['roc_auc']:.4f}\")\n",
                    "\n",
                    "# Display results\n",
                    "results_df = pd.DataFrame(results).T\n",
                    "print(\"\\n📊 Model Comparison:\")\n",
                    "print(results_df.round(4))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Plot model comparison\n",
                    "fig, ax = plt.subplots(figsize=(12, 6))\n",
                    "results_df.plot(kind='bar', ax=ax)\n",
                    "plt.title('Model Performance Comparison')\n",
                    "plt.xlabel('Models')\n",
                    "plt.ylabel('Score')\n",
                    "plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')\n",
                    "plt.xticks(rotation=45)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                    "\n",
                    "# Find best model\n",
                    "best_model = results_df['ROC AUC'].idxmax()\n",
                    "print(f\"🏆 Best model: {best_model} (ROC AUC: {results_df.loc[best_model, 'ROC AUC']:.4f})\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Use Best Model for Risk Assessment"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Initialize with best performing model\n",
                    "final_evaluator = CreditEvaluator(model_type=best_model)\n",
                    "final_evaluator.fit(X, y, validation_split=0.2)\n",
                    "\n",
                    "print(f\"Final model trained: {best_model}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Create test clients for demonstration\n",
                    "test_clients = pd.DataFrame({\n",
                    "    'age': [25, 45, 35, 55, 30],\n",
                    "    'annual_income': [35000, 85000, 60000, 120000, 45000],\n",
                    "    'credit_score': [580, 750, 680, 820, 620],\n",
                    "    'debt_to_income': [0.6, 0.2, 0.4, 0.15, 0.5],\n",
                    "    'credit_history_length': [2, 15, 8, 25, 5],\n",
                    "    'num_credit_accounts': [3, 8, 5, 12, 4],\n",
                    "    'credit_utilization': [0.9, 0.3, 0.6, 0.1, 0.8],\n",
                    "    'payment_history_score': [0.7, 0.95, 0.85, 0.98, 0.75],\n",
                    "    'num_late_payments': [5, 0, 2, 0, 4],\n",
                    "    'employment_length': [1, 10, 6, 20, 3],\n",
                    "    'home_ownership': ['rent', 'own', 'mortgage', 'own', 'rent'],\n",
                    "    'loan_purpose': ['debt_consolidation', 'home_improvement', 'major_purchase', 'other', 'debt_consolidation']\n",
                    "})\n",
                    "\n",
                    "client_ids = [f\"CLIENT_{i+1:03d}\" for i in range(len(test_clients))]\n",
                    "\n",
                    "print(\"Test clients created:\")\n",
                    "test_clients"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Get risk predictions\n",
                    "risk_scores, risk_categories = final_evaluator.predict_risk(test_clients)\n",
                    "\n",
                    "# Create results dataframe\n",
                    "results_df = test_clients.copy()\n",
                    "results_df['client_id'] = client_ids\n",
                    "results_df['risk_score'] = risk_scores\n",
                    "results_df['risk_category'] = risk_categories\n",
                    "\n",
                    "# Display results\n",
                    "print(\"🎯 Risk Assessment Results:\")\n",
                    "display_cols = ['client_id', 'age', 'credit_score', 'debt_to_income', 'risk_score', 'risk_category']\n",
                    "results_df[display_cols]"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Generate detailed client reports\n",
                    "print(\"📋 Detailed Client Reports:\\n\")\n",
                    "\n",
                    "for i, client_id in enumerate(client_ids):\n",
                    "    client_report = final_evaluator.generate_client_report(\n",
                    "        test_clients.iloc[i], client_id\n",
                    "    )\n",
                    "    \n",
                    "    print(f\"--- {client_report['client_id']} ---\")\n",
                    "    print(f\"Risk Score: {client_report['risk_score']:.0f}\")\n",
                    "    print(f\"Risk Category: {client_report['risk_category']} ({client_report['risk_level']})\")\n",
                    "    print(f\"Default Probability: {client_report['default_probability']}\")\n",
                    "    print(f\"Recommendation: {client_report['recommendation']}\")\n",
                    "    print(f\"Interest Rate Modifier: +{client_report['interest_rate_modifier']:.1f}%\")\n",
                    "    print(f\"Monitoring: {client_report['monitoring_frequency']}\")\n",
                    "    print()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Portfolio Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Portfolio analysis\n",
                    "portfolio_results = final_evaluator.evaluate_portfolio(test_clients, client_ids=client_ids)\n",
                    "portfolio_summary = portfolio_results['portfolio_summary']\n",
                    "\n",
                    "print(\"📊 Portfolio Analysis:\")\n",
                    "print(f\"Total Clients: {portfolio_summary['total_clients']}\")\n",
                    "print(f\"Average Risk Score: {portfolio_summary['average_risk_score']:.0f}\")\n",
                    "print(f\"Portfolio Risk Level: {portfolio_summary['portfolio_risk_level']}\")\n",
                    "print(f\"\\nRisk Distribution:\")\n",
                    "print(f\"  - A (Low Risk): {portfolio_summary['risk_distribution']['A_count']} clients ({portfolio_summary['risk_distribution']['A_percentage']:.1f}%)\")\n",
                    "print(f\"  - B (Medium Risk): {portfolio_summary['risk_distribution']['B_count']} clients ({portfolio_summary['risk_distribution']['B_percentage']:.1f}%)\")\n",
                    "print(f\"  - C (High Risk): {portfolio_summary['risk_distribution']['C_count']} clients ({portfolio_summary['risk_distribution']['C_percentage']:.1f}%)\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Visualize portfolio\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n",
                    "\n",
                    "# Risk score distribution\n",
                    "axes[0].hist(risk_scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')\n",
                    "axes[0].set_title('Risk Score Distribution')\n",
                    "axes[0].set_xlabel('Risk Score')\n",
                    "axes[0].set_ylabel('Frequency')\n",
                    "axes[0].axvline(x=300, color='green', linestyle='--', alpha=0.7, label='A-B Threshold')\n",
                    "axes[0].axvline(x=600, color='orange', linestyle='--', alpha=0.7, label='B-C Threshold')\n",
                    "axes[0].legend()\n",
                    "\n",
                    "# Risk category pie chart\n",
                    "category_counts = pd.Series(risk_categories).value_counts()\n",
                    "colors = {'A': 'lightgreen', 'B': 'orange', 'C': 'lightcoral'}\n",
                    "pie_colors = [colors.get(cat, 'gray') for cat in category_counts.index]\n",
                    "axes[1].pie(category_counts.values, labels=[f\"{cat} ({count})\" for cat, count in category_counts.items()], \n",
                    "           autopct='%1.1f%%', colors=pie_colors)\n",
                    "axes[1].set_title('Risk Category Distribution')\n",
                    "\n",
                    "# Risk score vs client\n",
                    "colors_scatter = [colors.get(cat, 'gray') for cat in risk_categories]\n",
                    "axes[2].scatter(range(len(risk_scores)), risk_scores, c=colors_scatter, alpha=0.7, s=100)\n",
                    "axes[2].set_title('Risk Scores by Client')\n",
                    "axes[2].set_xlabel('Client Index')\n",
                    "axes[2].set_ylabel('Risk Score')\n",
                    "axes[2].axhline(y=300, color='green', linestyle='--', alpha=0.7, label='A-B Threshold')\n",
                    "axes[2].axhline(y=600, color='orange', linestyle='--', alpha=0.7, label='B-C Threshold')\n",
                    "axes[2].legend()\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Feature Importance Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Get feature importance\n",
                    "feature_importance = final_evaluator.get_feature_importance()\n",
                    "\n",
                    "if feature_importance is not None:\n",
                    "    print(\"🔍 Top 10 Most Important Features:\")\n",
                    "    print(feature_importance.head(10))\n",
                    "    \n",
                    "    # Plot feature importance\n",
                    "    plt.figure(figsize=(10, 8))\n",
                    "    top_features = feature_importance.head(15)\n",
                    "    plt.barh(range(len(top_features)), top_features['importance'])\n",
                    "    plt.yticks(range(len(top_features)), top_features['feature'])\n",
                    "    plt.xlabel('Feature Importance')\n",
                    "    plt.title('Top 15 Feature Importance')\n",
                    "    plt.gca().invert_yaxis()\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print(\"Feature importance not available for this model type\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Model Persistence"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Save the trained model\n",
                    "model_path = 'trained_credit_model.pkl'\n",
                    "final_evaluator.save_model(model_path)\n",
                    "\n",
                    "print(f\"✅ Model saved to {model_path}\")\n",
                    "print(\"\\nYou can now load this model in production using:\")\n",
                    "print(\"evaluator = CreditEvaluator()\")\n",
                    "print(\"evaluator.load_model('trained_credit_model.pkl')\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Next Steps for Production\n",
                    "\n",
                    "### Immediate Actions:\n",
                    "1. **Replace sample data** with your actual credit data\n",
                    "2. **Adjust risk thresholds** in `config.py` based on your business requirements\n",
                    "3. **Fine-tune model hyperparameters** for better performance\n",
                    "\n",
                    "### Integration:\n",
                    "1. **API Integration**: Create REST API endpoints for real-time scoring\n",
                    "2. **Database Integration**: Connect to your customer database\n",
                    "3. **Monitoring**: Set up model performance monitoring\n",
                    "4. **Retraining**: Implement automated model retraining pipeline\n",
                    "\n",
                    "### Compliance:\n",
                    "1. **Regulatory Review**: Ensure model meets regulatory requirements\n",
                    "2. **Bias Testing**: Test for fairness across different demographic groups\n",
                    "3. **Documentation**: Document model decisions for audit trails\n",
                    "4. **Validation**: Implement ongoing model validation procedures"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.5"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Write notebook to file
    with open('/workspace/credit_evaluation_analysis.ipynb', 'w') as f:
        json.dump(notebook_content, f, indent=2)
    
    print("✅ Jupyter notebook created: credit_evaluation_analysis.ipynb")
    print("\nTo use this notebook:")
    print("1. Copy it to your local project directory")
    print("2. Start Jupyter: jupyter notebook")
    print("3. Open credit_evaluation_analysis.ipynb")
    print("4. Run all cells to see the complete analysis")

if __name__ == "__main__":
    create_jupyter_notebook()