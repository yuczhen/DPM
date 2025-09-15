"""
Download Script Generator for Windows
Creates individual files with content for easy download
"""

import os

def create_download_files():
    """Create individual downloadable files"""
    
    # List of files to create
    files_to_copy = [
        'requirements.txt',
        'config.py',
        'data_processor.py', 
        'model_trainer.py',
        'risk_classifier.py',
        'evaluation_metrics.py',
        'credit_evaluator.py',
        'simple_demo.py',
        'example_usage.py',
        'README.md',
        'IMPLEMENTATION_GUIDE.md'
    ]
    
    # Create download directory
    download_dir = '/workspace/download_files'
    os.makedirs(download_dir, exist_ok=True)
    
    print("📁 Creating downloadable files...")
    
    for filename in files_to_copy:
        source_path = f'/workspace/{filename}'
        dest_path = f'{download_dir}/{filename}'
        
        if os.path.exists(source_path):
            # Copy file content
            with open(source_path, 'r', encoding='utf-8') as src:
                content = src.read()
            
            with open(dest_path, 'w', encoding='utf-8') as dst:
                dst.write(content)
            
            print(f"✅ Created: {filename}")
        else:
            print(f"❌ Not found: {filename}")
    
    # Create a PowerShell script for Windows setup
    powershell_script = '''# A-B-C Credit Evaluation System Setup Script for Windows
# Run this in PowerShell as Administrator

Write-Host "🚀 Setting up A-B-C Credit Evaluation System..." -ForegroundColor Green

# Create project directory
$projectPath = "C:\\Projects\\abc-credit-evaluation"
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
& ".\\venv\\Scripts\\Activate.ps1"

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
'''
    
    with open(f'{download_dir}/setup_windows.ps1', 'w', encoding='utf-8') as f:
        f.write(powershell_script)
    
    print(f"✅ Created: setup_windows.ps1")
    
    # Create a batch file alternative
    batch_script = '''@echo off
echo 🚀 Setting up A-B-C Credit Evaluation System...

REM Create project directory
set "PROJECT_PATH=C:\\Projects\\abc-credit-evaluation"
mkdir "%PROJECT_PATH%" 2>nul
cd /d "%PROJECT_PATH%"

echo 📁 Project directory: %PROJECT_PATH%

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Install from python.org
    pause
    exit /b 1
)

echo ✅ Python found!

REM Create virtual environment
echo 🔧 Creating virtual environment...
python -m venv venv

echo ✅ Virtual environment created!
echo.
echo 📋 Next steps:
echo 1. Copy all Python files to: %PROJECT_PATH%
echo 2. Run: venv\\Scripts\\activate
echo 3. Run: pip install -r requirements.txt
echo 4. Test: python simple_demo.py
echo.
echo 🎯 Opening VS Code...
code .

pause
'''
    
    with open(f'{download_dir}/setup_windows.bat', 'w', encoding='utf-8') as f:
        f.write(batch_script)
    
    print(f"✅ Created: setup_windows.bat")
    
    print(f"\n📁 All files created in: {download_dir}")
    print(f"📊 Total files: {len(os.listdir(download_dir))}")
    
    return download_dir

if __name__ == "__main__":
    download_dir = create_download_files()
    
    print("\n" + "="*60)
    print("🎯 WINDOWS SETUP INSTRUCTIONS")
    print("="*60)
    print("1. Download all files from the download_files directory")
    print("2. Copy them to: C:\\Projects\\abc-credit-evaluation\\")
    print("3. Run setup_windows.bat")
    print("4. Open VS Code and start coding!")
    print("="*60)