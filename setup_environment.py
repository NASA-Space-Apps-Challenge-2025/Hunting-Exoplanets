#!/usr/bin/env python3
"""
Environment Setup Script for Exoplanet Detection Project
Installs required packages and sets up the Python environment
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def check_package(package):
    """Check if a package is already installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Exoplanet Detection Project Environment")
    print("=" * 60)
    
    # Core packages (should already be installed with requirements.txt)
    core_packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'jupyter'
    ]
    
    # Optional advanced packages
    advanced_packages = [
        'xgboost',
        'lightgbm',
        'plotly',
        'astropy',
        'astroquery'
    ]
    
    print("📦 Checking core packages...")
    missing_core = []
    for package in core_packages:
        if check_package(package):
            print(f"✅ {package} is installed")
        else:
            missing_core.append(package)
            print(f"❌ {package} is missing")
    
    print("\n📦 Checking advanced packages...")
    missing_advanced = []
    for package in advanced_packages:
        if check_package(package.replace('-', '_')):  # Handle package naming differences
            print(f"✅ {package} is installed")
        else:
            missing_advanced.append(package)
            print(f"❌ {package} is missing")
    
    # Install missing packages
    all_missing = missing_core + missing_advanced
    
    if all_missing:
        print(f"\n🔧 Installing {len(all_missing)} missing packages...")
        
        failed_installs = []
        for package in all_missing:
            if not install_package(package):
                failed_installs.append(package)
        
        if failed_installs:
            print(f"\n⚠️  Failed to install: {failed_installs}")
            print("You can try installing them manually:")
            for package in failed_installs:
                print(f"  pip install {package}")
        else:
            print("\n✅ All packages installed successfully!")
    else:
        print("\n✅ All packages are already installed!")
    
    # Verify Jupyter is working
    print("\n🔍 Verifying Jupyter installation...")
    try:
        result = subprocess.run([sys.executable, '-m', 'jupyter', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Jupyter is working correctly")
            print("💡 You can start Jupyter with: jupyter lab or jupyter notebook")
        else:
            print("❌ Jupyter verification failed")
    except Exception as e:
        print(f"❌ Error checking Jupyter: {e}")
    
    # Create .gitignore if it doesn't exist
    gitignore_path = ".gitignore"
    if not os.path.exists(gitignore_path):
        gitignore_content = '''# Python
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

# Jupyter Notebook
.ipynb_checkpoints

# Data files
*.csv
*.h5
*.hdf5
*.fits

# Models
*.pkl
*.joblib
*.model

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
'''
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print(f"✅ Created {gitignore_path}")
    
    print("\n🎉 Environment setup complete!")
    print("\n🚀 Next steps:")
    print("1. Run: python main.py (for complete pipeline)")
    print("2. Or: jupyter lab (to open the analysis notebook)")
    print("3. Or: cd notebooks && jupyter notebook exoplanet_detection_analysis.ipynb")

if __name__ == "__main__":
    main()
