@echo off
echo 🚀 Setting up A-B-C Credit Evaluation System on Windows...
echo.

REM Create project directory
echo 📁 Creating project directory...
mkdir "C:\Projects\abc-credit-evaluation" 2>nul
cd /d "C:\Projects\abc-credit-evaluation"

echo ✅ Project directory created: %CD%
echo.
echo 📝 Next steps:
echo 1. Copy all the Python files (I'll show you the content)
echo 2. Create virtual environment
echo 3. Install dependencies
echo 4. Test the system
echo 5. Set up Git and push to GitHub
echo.

echo 💡 Ready to create files? Press any key to continue...
pause >nul

echo.
echo 🔧 Creating Python virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ❌ Python not found. Please install Python from python.org
    pause
    exit /b 1
)

echo ✅ Virtual environment created!
echo.
echo 📦 To activate virtual environment, run:
echo    venv\Scripts\activate
echo.
echo 📋 Files you need to create (I'll show you the content):
echo    - requirements.txt
echo    - config.py  
echo    - data_processor.py
echo    - model_trainer.py
echo    - risk_classifier.py
echo    - evaluation_metrics.py
echo    - credit_evaluator.py
echo    - simple_demo.py
echo    - example_usage.py
echo    - README.md
echo    - IMPLEMENTATION_GUIDE.md
echo.
echo ✅ Setup script completed!
pause