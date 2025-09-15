# A-B-C Credit Evaluation System - One Command Setup for Windows
# Save this as setup.ps1 and run in PowerShell

param(
    [string]$ProjectPath = "C:\Projects\abc-credit-evaluation"
)

Write-Host "🚀 Setting up A-B-C Credit Evaluation System..." -ForegroundColor Green
Write-Host "📁 Project path: $ProjectPath" -ForegroundColor Yellow

# Create project directory
New-Item -ItemType Directory -Path $ProjectPath -Force | Out-Null
Set-Location $ProjectPath

# Check Python
try {
    $pythonVersion = python --version 2>$null
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python from python.org" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "📝 Creating project files..." -ForegroundColor Yellow

# Create requirements.txt
@"
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
imbalanced-learn==0.11.0
joblib==1.3.2
scipy==1.11.1
"@ | Out-File -FilePath "requirements.txt" -Encoding UTF8

Write-Host "✅ Created requirements.txt" -ForegroundColor Green

# Create simple_demo.py (working demo)
@'
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
    
    def evaluate_client(self, client_data: Dict, client_id: str = "Client") -> Dict:
        """Complete client evaluation"""
        risk_score = self.calculate_risk_score(client_data)
        risk_category = self.get_risk_category(risk_score)
        
        descriptions = {
            'A': {
                'risk_level': 'Low Risk',
                'recommendation': 'Approve loan with standard terms',
                'interest_rate_modifier': 0.0,
                'monitoring_frequency': 'Annual'
            },
            'B': {
                'risk_level': 'Medium Risk', 
                'recommendation': 'Approve with standard monitoring',
                'interest_rate_modifier': 1.5,
                'monitoring_frequency': 'Semi-annual'
            },
            'C': {
                'risk_level': 'High Risk',
                'recommendation': 'Requires additional documentation and higher interest rates',
                'interest_rate_modifier': 3.0,
                'monitoring_frequency': 'Quarterly'
            }
        }
        
        risk_info = descriptions.get(risk_category, descriptions['C'])
        
        return {
            'client_id': client_id,
            'risk_score': round(risk_score, 0),
            'risk_category': risk_category,
            'risk_level': risk_info['risk_level'],
            'recommendation': risk_info['recommendation'],
            'interest_rate_modifier': risk_info['interest_rate_modifier'],
            'monitoring_frequency': risk_info['monitoring_frequency']
        }

def main():
    print("=== A-B-C Credit Card Model Demo ===\n")
    
    # Initialize evaluator
    evaluator = SimpleCreditEvaluator()
    
    # Sample clients
    clients = [
        {
            'client_id': 'CLIENT_001',
            'age': 25, 'credit_score': 580, 'debt_to_income': 0.6,
            'payment_history_score': 0.7, 'employment_length': 1
        },
        {
            'client_id': 'CLIENT_002', 
            'age': 45, 'credit_score': 750, 'debt_to_income': 0.2,
            'payment_history_score': 0.95, 'employment_length': 10
        },
        {
            'client_id': 'CLIENT_003',
            'age': 35, 'credit_score': 620, 'debt_to_income': 0.5,
            'payment_history_score': 0.8, 'employment_length': 3
        }
    ]
    
    print("Client Risk Evaluations:\n")
    
    for client in clients:
        evaluation = evaluator.evaluate_client(client, client['client_id'])
        
        print(f"--- {evaluation['client_id']} ---")
        print(f"Risk Score: {evaluation['risk_score']}")
        print(f"Risk Category: {evaluation['risk_category']} ({evaluation['risk_level']})")
        print(f"Recommendation: {evaluation['recommendation']}")
        print(f"Interest Rate Modifier: +{evaluation['interest_rate_modifier']:.1f}%")
        print(f"Monitoring: {evaluation['monitoring_frequency']}")
        print()
    
    print("✅ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Install full ML dependencies: pip install -r requirements.txt")
    print("2. Get the complete ML implementation files")
    print("3. Set up GitHub repository")

if __name__ == "__main__":
    main()
'@ | Out-File -FilePath "simple_demo.py" -Encoding UTF8

Write-Host "✅ Created simple_demo.py" -ForegroundColor Green

# Create basic README
@"
# A-B-C Credit Evaluation System

## Quick Start
1. Test the system: ``python simple_demo.py``
2. Install dependencies: ``pip install -r requirements.txt``
3. Get full ML implementation from your AI assistant

## Risk Categories
- A (Low Risk): Score 0-300 - Excellent creditworthiness
- B (Medium Risk): Score 301-600 - Moderate risk profile  
- C (High Risk): Score 601-1000 - Elevated default risk

## Next Steps
1. Get the complete ML implementation files
2. Set up virtual environment
3. Train models with your actual data
4. Deploy to production
"@ | Out-File -FilePath "README.md" -Encoding UTF8

Write-Host "✅ Created README.md" -ForegroundColor Green

# Create virtual environment
Write-Host "🔧 Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
Write-Host "✅ Virtual environment created!" -ForegroundColor Green

# Test the demo
Write-Host "🧪 Testing simple demo..." -ForegroundColor Yellow
try {
    python simple_demo.py
    Write-Host "✅ Demo test successful!" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Demo test failed, but files are created" -ForegroundColor Yellow
}

Write-Host "`n🎯 Opening VS Code..." -ForegroundColor Cyan
code .

Write-Host "`n✅ Setup completed!" -ForegroundColor Green
Write-Host "📁 Project location: $ProjectPath" -ForegroundColor Yellow
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Activate virtual environment: venv\Scripts\activate" -ForegroundColor White
Write-Host "2. Get remaining ML files from your AI assistant" -ForegroundColor White  
Write-Host "3. Install dependencies: pip install -r requirements.txt" -ForegroundColor White
Write-Host "4. Test full system: python example_usage.py" -ForegroundColor White

Read-Host "`nPress Enter to exit"