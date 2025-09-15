#!/bin/bash

# GitHub Setup Script for A-B-C Credit Evaluation System
# Run this script in your local project directory after copying all files

echo "🚀 Setting up A-B-C Credit Evaluation System for GitHub..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    exit 1
fi

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
else
    echo "📁 Git repository already exists."
fi

# Create .gitignore file
echo "📝 Creating .gitignore file..."
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
env.bak/
venv.bak/

# Jupyter Notebook
.ipynb_checkpoints

# Model files (optional - uncomment if you don't want to track trained models)
# *.pkl
# *.joblib

# Data files (uncomment if you don't want to track data files)
# *.csv
# *.json
# data/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Results and outputs
results/
outputs/
plots/
*.png
*.jpg
*.pdf

EOF

# Add all files
echo "➕ Adding files to Git..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "Initial commit: A-B-C Credit Evaluation System

- Complete ML pipeline for credit risk assessment
- Support for multiple models (XGBoost, LightGBM, Random Forest, Logistic Regression)
- A-B-C risk classification system
- Feature engineering and data preprocessing
- Model evaluation and visualization
- Jupyter notebook for analysis
- Production-ready architecture"

# Check if remote origin exists
if git remote get-url origin &> /dev/null; then
    echo "🔗 Remote origin already configured."
    echo "📤 Pushing to existing repository..."
    git push -u origin main 2>/dev/null || git push -u origin master 2>/dev/null || echo "⚠️  Push failed. You may need to set up the remote repository first."
else
    echo "🔗 No remote repository configured."
    echo ""
    echo "📋 Next steps:"
    echo "1. Go to GitHub.com and create a new repository"
    echo "2. Copy the repository URL"
    echo "3. Run these commands:"
    echo ""
    echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
    echo "   git branch -M main"
    echo "   git push -u origin main"
    echo ""
fi

echo "✅ Git setup completed!"
echo ""
echo "📁 Project structure:"
find . -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.ipynb" | head -20

echo ""
echo "🎯 Quick test commands:"
echo "  python3 simple_demo.py           # Test without ML dependencies"
echo "  python3 example_usage.py         # Full ML demo (requires pip install -r requirements.txt)"
echo "  jupyter notebook                 # Start Jupyter for analysis"
echo ""
echo "📚 Documentation:"
echo "  README.md                        # Project overview"
echo "  IMPLEMENTATION_GUIDE.md          # Detailed implementation guide"
echo "  setup_instructions.md            # Local setup instructions"