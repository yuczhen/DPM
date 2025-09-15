"""
GitHub File Creator - Generates ready-to-paste content for GitHub
Run this to get individual file contents you can copy-paste directly into GitHub
"""

import os

def create_github_files_guide():
    """Create a guide for manually creating files on GitHub"""
    
    files_to_include = [
        'requirements.txt',
        'README.md', 
        'config.py',
        'data_processor.py',
        'model_trainer.py',
        'risk_classifier.py',
        'evaluation_metrics.py',
        'credit_evaluator.py',
        'simple_demo.py',
        'example_usage.py',
        'IMPLEMENTATION_GUIDE.md'
    ]
    
    print("🚀 GitHub Repository Setup Guide")
    print("=" * 50)
    print()
    print("STEP 1: Create New Repository on GitHub")
    print("- Go to github.com")
    print("- Click 'New repository'")
    print("- Name: 'abc-credit-evaluation' or 'credit-risk-model'")
    print("- Description: 'A-B-C Credit Evaluation System for Loan Risk Assessment'")
    print("- Make it Public or Private (your choice)")
    print("- Don't initialize with README")
    print("- Click 'Create repository'")
    print()
    
    print("STEP 2: Create Files Manually")
    print("For each file below, click 'Add file' > 'Create new file' in your GitHub repo")
    print()
    
    for i, filename in enumerate(files_to_include, 1):
        if os.path.exists(f'/workspace/{filename}'):
            print(f"📄 FILE {i}: {filename}")
            print("-" * 40)
            print("Copy this content:")
            print()
            
            with open(f'/workspace/{filename}', 'r') as f:
                content = f.read()
                # Show first few lines as preview
                lines = content.split('\n')
                preview_lines = min(10, len(lines))
                
                print(f"```")
                for line in lines[:preview_lines]:
                    print(line)
                if len(lines) > preview_lines:
                    print(f"... ({len(lines) - preview_lines} more lines)")
                print(f"```")
                print()
                print(f"📋 Full content length: {len(content)} characters")
                print(f"📋 Total lines: {len(lines)}")
                print()
            
            print("=" * 50)
            print()
    
    print("STEP 3: Test Your Repository")
    print("- Clone your repository locally")
    print("- Run: python3 simple_demo.py")
    print("- Install dependencies and run: python3 example_usage.py")
    print()
    
    print("🎯 Alternative: Use GitHub CLI")
    print("If you have GitHub CLI installed:")
    print("gh repo create abc-credit-evaluation --public")
    print("git clone https://github.com/YOUR_USERNAME/abc-credit-evaluation.git")
    print("# Copy files to the cloned directory")
    print("git add .")
    print("git commit -m 'Initial commit'")
    print("git push")

def create_single_file_content(filename):
    """Get content for a specific file"""
    if os.path.exists(f'/workspace/{filename}'):
        with open(f'/workspace/{filename}', 'r') as f:
            return f.read()
    return None

if __name__ == "__main__":
    create_github_files_guide()
    
    print("\n" + "="*50)
    print("💡 QUICK TIP:")
    print("If you want the content of a specific file, run:")
    print("python3 -c \"from github_file_creator import create_single_file_content; print(create_single_file_content('filename.py'))\"")