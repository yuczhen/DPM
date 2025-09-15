# A-B-C Credit Evaluation System Setup Script for Windows
# Run this in PowerShell as Administrator

Write-Host "🚀 Setting up A-B-C Credit Evaluation System..." -ForegroundColor Green

# Create project directory
$projectPath = "C:\Projects\abc-credit-evaluation"
New-Item -ItemType Directory -Path $projectPath -Force
Set-Location $projectPath

Write-Host "📁 Project directory created: $projectPath" -ForegroundColor Yellow

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python from python.org" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "🔧 Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host "🔄 Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Install requirements (after you copy the files)
Write-Host "📦 To install dependencies later, run:" -ForegroundColor Cyan
Write-Host "   pip install -r requirements.txt" -ForegroundColor White

# Open VS Code
Write-Host "🎯 Opening VS Code..." -ForegroundColor Yellow
code .

Write-Host "✅ Setup completed!" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Copy all Python files to this directory" -ForegroundColor White
Write-Host "2. Run: pip install -r requirements.txt" -ForegroundColor White
Write-Host "3. Test: python simple_demo.py" -ForegroundColor White
