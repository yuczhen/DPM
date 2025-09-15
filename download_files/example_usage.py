"""
Example Usage of Credit Evaluation System
Demonstrates how to use the A-B-C credit card model for loan risk evaluation
"""

import pandas as pd
import numpy as np
from credit_evaluator import CreditEvaluator
from data_processor import DataProcessor
import matplotlib.pyplot as plt

def main():
    print("=== A-B-C Credit Card Model Demo ===\n")
    
    # 1. Initialize the system
    print("1. Initializing Credit Evaluation System...")
    evaluator = CreditEvaluator(model_type='xgboost')  # You can use 'random_forest', 'lightgbm', 'logistic_regression'
    
    # 2. Load sample data
    print("2. Loading sample credit data...")
    processor = DataProcessor()
    df = processor.load_sample_data(n_samples=2000)
    
    print(f"   Loaded {len(df)} client records")
    print(f"   Features: {list(df.columns[:-1])}")
    print(f"   Default rate: {df['default_risk'].mean():.2%}\n")
    
    # 3. Prepare data
    X = df.drop('default_risk', axis=1)
    y = df['default_risk']
    
    # 4. Train the model
    print("3. Training the credit evaluation model...")
    training_metrics = evaluator.fit(X, y, validation_split=0.2, optimize_hyperparameters=False)
    
    if training_metrics:
        print(f"   Training completed!")
        print(f"   Validation ROC AUC: {training_metrics['roc_auc']:.4f}")
        print(f"   Validation Accuracy: {training_metrics['accuracy']:.4f}\n")
    
    # 5. Test on new data
    print("4. Testing on new client data...")
    
    # Create some test clients
    test_clients = pd.DataFrame({
        'age': [25, 45, 35, 55, 30],
        'annual_income': [35000, 85000, 60000, 120000, 45000],
        'credit_score': [580, 750, 680, 820, 620],
        'debt_to_income': [0.6, 0.2, 0.4, 0.15, 0.5],
        'credit_history_length': [2, 15, 8, 25, 5],
        'num_credit_accounts': [3, 8, 5, 12, 4],
        'credit_utilization': [0.9, 0.3, 0.6, 0.1, 0.8],
        'payment_history_score': [0.7, 0.95, 0.85, 0.98, 0.75],
        'num_late_payments': [5, 0, 2, 0, 4],
        'employment_length': [1, 10, 6, 20, 3],
        'home_ownership': ['rent', 'own', 'mortgage', 'own', 'rent'],
        'loan_purpose': ['debt_consolidation', 'home_improvement', 'major_purchase', 'other', 'debt_consolidation']
    })
    
    client_ids = [f"CLIENT_{i+1:03d}" for i in range(len(test_clients))]
    
    # Get predictions
    risk_scores, risk_categories = evaluator.predict_risk(test_clients)
    
    # 6. Generate individual client reports
    print("5. Generating client risk reports...\n")
    
    for i, client_id in enumerate(client_ids):
        client_report = evaluator.generate_client_report(
            test_clients.iloc[i], client_id
        )
        
        print(f"--- {client_report['client_id']} ---")
        print(f"Risk Score: {client_report['risk_score']:.0f}")
        print(f"Risk Category: {client_report['risk_category']} ({client_report['risk_level']})")
        print(f"Default Probability: {client_report['default_probability']:.2%}")
        print(f"Recommendation: {client_report['recommendation']}")
        print(f"Interest Rate Modifier: +{client_report['interest_rate_modifier']:.1f}%")
        print(f"Monitoring: {client_report['monitoring_frequency']}")
        print()
    
    # 7. Portfolio analysis
    print("6. Portfolio Analysis...")
    portfolio_results = evaluator.evaluate_portfolio(test_clients, client_ids=client_ids)
    
    portfolio_summary = portfolio_results['portfolio_summary']
    print(f"   Total Clients: {portfolio_summary['total_clients']}")
    print(f"   Average Risk Score: {portfolio_summary['average_risk_score']:.0f}")
    print(f"   Portfolio Risk Level: {portfolio_summary['portfolio_risk_level']}")
    print(f"   Risk Distribution:")
    print(f"     - A (Low Risk): {portfolio_summary['risk_distribution']['A_count']} clients ({portfolio_summary['risk_distribution']['A_percentage']:.1f}%)")
    print(f"     - B (Medium Risk): {portfolio_summary['risk_distribution']['B_count']} clients ({portfolio_summary['risk_distribution']['B_percentage']:.1f}%)")
    print(f"     - C (High Risk): {portfolio_summary['risk_distribution']['C_count']} clients ({portfolio_summary['risk_distribution']['C_percentage']:.1f}%)")
    print()
    
    # 8. Feature importance analysis
    print("7. Feature Importance Analysis...")
    feature_importance = evaluator.get_feature_importance()
    
    if feature_importance is not None:
        print("   Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"     {i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
    print()
    
    # 9. Create visualizations
    print("8. Creating visualizations...")
    
    # Plot risk distribution
    plt.figure(figsize=(12, 8))
    
    # Risk score distribution
    plt.subplot(2, 2, 1)
    plt.hist(risk_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Risk Score Distribution')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    
    # Risk category distribution
    plt.subplot(2, 2, 2)
    category_counts = pd.Series(risk_categories).value_counts()
    plt.pie(category_counts.values, labels=[f"{cat} ({count})" for cat, count in category_counts.items()], 
            autopct='%1.1f%%', colors=['lightgreen', 'orange', 'lightcoral'])
    plt.title('Risk Category Distribution')
    
    # Feature importance (top 10)
    if feature_importance is not None:
        plt.subplot(2, 2, 3)
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.title('Top 10 Feature Importance')
        plt.xlabel('Importance')
    
    # Risk score vs category
    plt.subplot(2, 2, 4)
    colors = {'A': 'green', 'B': 'orange', 'C': 'red'}
    for category in ['A', 'B', 'C']:
        mask = risk_categories == category
        if mask.any():
            plt.scatter(range(len(risk_scores)), risk_scores, 
                       c=[colors[cat] for cat in risk_categories], alpha=0.7)
    plt.title('Risk Scores by Category')
    plt.xlabel('Client Index')
    plt.ylabel('Risk Score')
    plt.axhline(y=300, color='green', linestyle='--', alpha=0.7, label='A-B Threshold')
    plt.axhline(y=600, color='orange', linestyle='--', alpha=0.7, label='B-C Threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/workspace/credit_evaluation_results.png', dpi=300, bbox_inches='tight')
    print("   Visualization saved as 'credit_evaluation_results.png'")
    
    # 10. Save the trained model
    print("9. Saving the trained model...")
    evaluator.save_model('/workspace/trained_credit_model.pkl')
    
    print("\n=== Demo completed successfully! ===")
    print("\nNext steps:")
    print("1. Replace sample data with your actual credit data")
    print("2. Adjust risk thresholds in config.py based on your business requirements")
    print("3. Fine-tune model hyperparameters for better performance")
    print("4. Implement the model in your production environment")
    print("5. Set up regular model retraining and monitoring")

def demonstrate_model_comparison():
    """
    Demonstrate comparison between different models
    """
    print("\n=== Model Comparison Demo ===\n")
    
    # Load data
    processor = DataProcessor()
    df = processor.load_sample_data(n_samples=1000)
    X = df.drop('default_risk', axis=1)
    y = df['default_risk']
    
    # Test different models
    models = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    results = {}
    
    for model_type in models:
        print(f"Training {model_type}...")
        evaluator = CreditEvaluator(model_type=model_type)
        metrics = evaluator.fit(X, y, validation_split=0.2)
        
        if metrics:
            results[model_type] = {
                'ROC AUC': metrics['roc_auc'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            }
    
    # Display comparison
    comparison_df = pd.DataFrame(results).T
    print("\nModel Comparison Results:")
    print(comparison_df.round(4))
    
    # Find best model
    best_model = comparison_df['ROC AUC'].idxmax()
    print(f"\nBest performing model: {best_model} (ROC AUC: {comparison_df.loc[best_model, 'ROC AUC']:.4f})")

if __name__ == "__main__":
    # Run main demo
    main()
    
    # Run model comparison demo
    demonstrate_model_comparison()