"""
File Contents Generator for Windows Setup
Run this to get the content of each file for copy-paste into VS Code
"""

import os

def print_file_content(filename, description):
    """Print file content with clear formatting"""
    print("=" * 80)
    print(f"📄 {filename.upper()}")
    print(f"Description: {description}")
    print("=" * 80)
    
    if os.path.exists(f'/workspace/{filename}'):
        with open(f'/workspace/{filename}', 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    else:
        print(f"❌ File {filename} not found")
    
    print("\n" + "=" * 80)
    print(f"✅ Copy the above content and save as: {filename}")
    print("=" * 80)
    print("\n\n")

def main():
    """Generate all file contents"""
    
    files_info = [
        ("requirements.txt", "Python package dependencies"),
        ("config.py", "Configuration settings and thresholds"),
        ("data_processor.py", "Data preprocessing and feature engineering"),
        ("model_trainer.py", "Machine learning model training"),
        ("risk_classifier.py", "A-B-C risk classification logic"),
        ("evaluation_metrics.py", "Model evaluation and visualization"),
        ("credit_evaluator.py", "Main integration class"),
        ("simple_demo.py", "Simple demo without ML dependencies"),
        ("example_usage.py", "Full ML implementation demo"),
        ("README.md", "Project documentation"),
        ("IMPLEMENTATION_GUIDE.md", "Detailed implementation guide")
    ]
    
    print("🚀 A-B-C CREDIT EVALUATION SYSTEM - FILE CONTENTS")
    print("=" * 80)
    print("Copy each file content below into VS Code")
    print("Save each file with the exact filename shown")
    print("=" * 80)
    print("\n")
    
    for filename, description in files_info:
        print_file_content(filename, description)
        input("Press Enter to continue to next file...")

if __name__ == "__main__":
    main()