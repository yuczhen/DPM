"""
Simple Demo of A-B-C Credit Card Model (No External Dependencies)
This demonstrates the core concepts without requiring ML libraries
"""

import random
import json
from typing import Dict, List, Tuple

class SimpleCreditEvaluator:
    """
    Simplified credit evaluator for demonstration purposes
    Uses rule-based approach instead of ML models
    """
    
    def __init__(self):
        self.risk_thresholds = {
            'A': (0, 300),      # Low Risk
            'B': (301, 600),    # Medium Risk  
            'C': (601, 1000)    # High Risk
        }
        
    def calculate_risk_score(self, client_data: Dict) -> float:
        """
        Calculate risk score based on client data using simple rules
        Returns score between 0-1000
        """
        score = 0
        
        # Credit score factor (30% weight)
        credit_score = client_data.get('credit_score', 650)
        credit_factor = max(0, (850 - credit_score) / 550) * 300
        score += credit_factor
        
        # Debt to income ratio (25% weight)
        debt_to_income = client_data.get('debt_to_income', 0.3)
        debt_factor = debt_to_income * 250
        score += debt_factor
        
        # Payment history (20% weight)
        payment_history = client_data.get('payment_history_score', 0.85)
        payment_factor = (1 - payment_history) * 200
        score += payment_factor
        
        # Age factor (15% weight)
        age = client_data.get('age', 35)
        if age < 25:
            age_factor = 50  # Young borrowers higher risk
        elif age > 65:
            age_factor = 30  # Older borrowers moderate risk
        else:
            age_factor = 0
        score += age_factor
        
        # Employment length (10% weight)
        employment_length = client_data.get('employment_length', 5)
        if employment_length < 2:
            employment_factor = 100
        else:
            employment_factor = max(0, (5 - employment_length) * 20)
        score += employment_factor
        
        return min(1000, max(0, score))
    
    def get_risk_category(self, score: float) -> str:
        """Convert risk score to A-B-C category"""
        if score <= self.risk_thresholds['A'][1]:
            return 'A'
        elif score <= self.risk_thresholds['B'][1]:
            return 'B'
        else:
            return 'C'
    
    def get_risk_description(self, category: str) -> Dict:
        """Get detailed risk description"""
        descriptions = {
            'A': {
                'risk_level': 'Low Risk',
                'description': 'Excellent creditworthiness with very low default probability',
                'recommendation': 'Approve loan with standard terms',
                'interest_rate_modifier': 0.0,
                'monitoring_frequency': 'Annual',
                'default_probability': '< 5%'
            },
            'B': {
                'risk_level': 'Medium Risk', 
                'description': 'Good creditworthiness with moderate risk profile',
                'recommendation': 'Approve with standard monitoring',
                'interest_rate_modifier': 1.5,
                'monitoring_frequency': 'Semi-annual',
                'default_probability': '5-15%'
            },
            'C': {
                'risk_level': 'High Risk',
                'description': 'Elevated default risk requiring careful evaluation',
                'recommendation': 'Requires additional documentation and higher interest rates',
                'interest_rate_modifier': 3.0,
                'monitoring_frequency': 'Quarterly',
                'default_probability': '> 15%'
            }
        }
        return descriptions.get(category, descriptions['C'])
    
    def evaluate_client(self, client_data: Dict, client_id: str = "Client") -> Dict:
        """Complete client evaluation"""
        risk_score = self.calculate_risk_score(client_data)
        risk_category = self.get_risk_category(risk_score)
        risk_info = self.get_risk_description(risk_category)
        
        return {
            'client_id': client_id,
            'risk_score': round(risk_score, 0),
            'risk_category': risk_category,
            'risk_level': risk_info['risk_level'],
            'description': risk_info['description'],
            'recommendation': risk_info['recommendation'],
            'interest_rate_modifier': risk_info['interest_rate_modifier'],
            'monitoring_frequency': risk_info['monitoring_frequency'],
            'default_probability': risk_info['default_probability']
        }
    
    def evaluate_portfolio(self, clients: List[Dict]) -> Dict:
        """Evaluate multiple clients and provide portfolio summary"""
        results = []
        risk_scores = []
        categories = {'A': 0, 'B': 0, 'C': 0}
        
        for i, client in enumerate(clients):
            client_id = client.get('client_id', f"CLIENT_{i+1:03d}")
            evaluation = self.evaluate_client(client, client_id)
            results.append(evaluation)
            risk_scores.append(evaluation['risk_score'])
            categories[evaluation['risk_category']] += 1
        
        total_clients = len(clients)
        avg_risk_score = sum(risk_scores) / total_clients if total_clients > 0 else 0
        
        # Determine portfolio risk level
        if avg_risk_score <= 300:
            portfolio_risk = "Low Risk Portfolio"
        elif avg_risk_score <= 600:
            portfolio_risk = "Moderate Risk Portfolio"
        else:
            portfolio_risk = "High Risk Portfolio"
        
        portfolio_summary = {
            'total_clients': total_clients,
            'average_risk_score': round(avg_risk_score, 0),
            'portfolio_risk_level': portfolio_risk,
            'risk_distribution': {
                'A_count': categories['A'],
                'B_count': categories['B'],
                'C_count': categories['C'],
                'A_percentage': (categories['A'] / total_clients) * 100,
                'B_percentage': (categories['B'] / total_clients) * 100,
                'C_percentage': (categories['C'] / total_clients) * 100
            }
        }
        
        return {
            'client_evaluations': results,
            'portfolio_summary': portfolio_summary
        }

def generate_sample_clients() -> List[Dict]:
    """Generate sample client data for testing"""
    random.seed(42)
    
    clients = []
    for i in range(10):
        client = {
            'client_id': f"CLIENT_{i+1:03d}",
            'age': random.randint(22, 70),
            'annual_income': random.randint(25000, 150000),
            'credit_score': random.randint(300, 850),
            'debt_to_income': random.uniform(0.1, 0.8),
            'payment_history_score': random.uniform(0.5, 1.0),
            'employment_length': random.uniform(0, 20),
            'home_ownership': random.choice(['rent', 'own', 'mortgage']),
            'loan_purpose': random.choice(['debt_consolidation', 'home_improvement', 'major_purchase', 'other'])
        }
        clients.append(client)
    
    return clients

def main():
    print("=== A-B-C Credit Card Model Demo ===\n")
    
    # Initialize evaluator
    evaluator = SimpleCreditEvaluator()
    
    # Generate sample clients
    clients = generate_sample_clients()
    
    print("1. Sample Client Data Generated:")
    print(f"   Number of clients: {len(clients)}")
    print()
    
    # Evaluate individual clients
    print("2. Individual Client Evaluations:\n")
    
    for client in clients[:5]:  # Show first 5 clients
        evaluation = evaluator.evaluate_client(client, client['client_id'])
        
        print(f"--- {evaluation['client_id']} ---")
        print(f"Age: {client['age']}, Income: ${client['annual_income']:,}")
        print(f"Credit Score: {client['credit_score']}, Debt-to-Income: {client['debt_to_income']:.2f}")
        print(f"Risk Score: {evaluation['risk_score']}")
        print(f"Risk Category: {evaluation['risk_category']} ({evaluation['risk_level']})")
        print(f"Recommendation: {evaluation['recommendation']}")
        print(f"Interest Rate Modifier: +{evaluation['interest_rate_modifier']:.1f}%")
        print(f"Monitoring: {evaluation['monitoring_frequency']}")
        print()
    
    # Portfolio analysis
    print("3. Portfolio Analysis:")
    portfolio_results = evaluator.evaluate_portfolio(clients)
    portfolio_summary = portfolio_results['portfolio_summary']
    
    print(f"   Total Clients: {portfolio_summary['total_clients']}")
    print(f"   Average Risk Score: {portfolio_summary['average_risk_score']}")
    print(f"   Portfolio Risk Level: {portfolio_summary['portfolio_risk_level']}")
    print(f"   Risk Distribution:")
    print(f"     - A (Low Risk): {portfolio_summary['risk_distribution']['A_count']} clients ({portfolio_summary['risk_distribution']['A_percentage']:.1f}%)")
    print(f"     - B (Medium Risk): {portfolio_summary['risk_distribution']['B_count']} clients ({portfolio_summary['risk_distribution']['B_percentage']:.1f}%)")
    print(f"     - C (High Risk): {portfolio_summary['risk_distribution']['C_count']} clients ({portfolio_summary['risk_distribution']['C_percentage']:.1f}%)")
    print()
    
    # Risk factor analysis
    print("4. Key Risk Factors Analysis:")
    print("   The model considers the following factors (in order of importance):")
    print("   1. Credit Score (30% weight) - Lower scores increase risk")
    print("   2. Debt-to-Income Ratio (25% weight) - Higher ratios increase risk")
    print("   3. Payment History (20% weight) - Poor history increases risk")
    print("   4. Age (15% weight) - Very young or old borrowers have higher risk")
    print("   5. Employment Length (10% weight) - Short employment increases risk")
    print()
    
    # Business recommendations
    print("5. Business Recommendations:")
    print("   Category A (Low Risk):")
    print("   - Approve loans quickly with standard interest rates")
    print("   - Minimal documentation required")
    print("   - Annual review sufficient")
    print()
    print("   Category B (Medium Risk):")
    print("   - Approve with moderate interest rate premium (+1.5%)")
    print("   - Standard documentation and verification")
    print("   - Semi-annual monitoring recommended")
    print()
    print("   Category C (High Risk):")
    print("   - Requires careful evaluation and additional documentation")
    print("   - Higher interest rate premium (+3.0%) to compensate for risk")
    print("   - Quarterly monitoring and follow-up required")
    print("   - Consider requiring collateral or co-signer")
    print()
    
    # Save results
    with open('/workspace/portfolio_analysis.json', 'w') as f:
        # Convert to JSON-serializable format
        json_results = {
            'portfolio_summary': portfolio_summary,
            'client_evaluations': portfolio_results['client_evaluations']
        }
        json.dump(json_results, f, indent=2)
    
    print("6. Results saved to 'portfolio_analysis.json'")
    print("\n=== Demo completed successfully! ===")

if __name__ == "__main__":
    main()