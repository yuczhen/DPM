@echo off
echo 🚀 Setting up A-B-C Credit Evaluation System...

REM Create project directory
set "PROJECT_PATH=C:\Projects\abc-credit-evaluation"
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
echo 2. Run: venv\Scripts\activate
echo 3. Run: pip install -r requirements.txt
echo 4. Test: python simple_demo.py
echo.
echo 🎯 Opening VS Code...
code .

pause
