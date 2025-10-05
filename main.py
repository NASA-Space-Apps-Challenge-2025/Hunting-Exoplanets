"""
Main execution script for the Exoplanet Detection Project
Orchestrates Phase 1 and Phase 2 of the project plan
"""

import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_loader import run_phase1_data_exploration
from src.data_preprocessor import run_phase1_preprocessing
from src.model_trainer import run_phase2_model_training
from config import *

def setup_project_structure():
    """Ensure all necessary directories exist"""
    directories = [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Directory ready: {directory}")

def main():
    """
    Main execution function for the complete exoplanet detection pipeline
    """
    print("🚀 EXOPLANET DETECTION PROJECT - COMPLETE PIPELINE 🚀")
    print("=" * 70)
    
    # Setup project structure
    print("\n📁 Setting up project structure...")
    setup_project_structure()
    
    try:
        # Phase 1: Data Exploration
        print("\n🔍 PHASE 1A: DATA EXPLORATION")
        print("-" * 50)
        loader, data_info = run_phase1_data_exploration()
        
        if loader is None:
            print("❌ Phase 1A failed. Exiting.")
            return False
        
        # Phase 1: Data Preprocessing
        print("\n🔧 PHASE 1B: DATA PREPROCESSING")
        print("-" * 50)
        preprocessor = run_phase1_preprocessing(target_method='discovery_method')
        
        if preprocessor is None:
            print("❌ Phase 1B failed. Exiting.")
            return False
        
        # Phase 2: Model Training
        print("\n🤖 PHASE 2: MODEL TRAINING")
        print("-" * 50)
        trainer = run_phase2_model_training()
        
        if trainer is None:
            print("❌ Phase 2 failed. Exiting.")
            return False
        
        # Project completion summary
        print("\n🎉 PROJECT COMPLETION SUMMARY")
        print("=" * 50)
        print("✅ Phase 1A: Data exploration completed")
        print("✅ Phase 1B: Data preprocessing completed") 
        print("✅ Phase 2: Model training completed")
        
        if hasattr(trainer, 'best_model_name'):
            print(f"🏆 Best performing model: {trainer.best_model_name}")
        
        print(f"\n📊 Results saved in: {RESULTS_DIR}")
        print(f"🤖 Models saved in: {MODELS_DIR}")
        print(f"📂 Processed data in: {PROCESSED_DATA_DIR}")
        
        print("\n🔬 Next steps:")
        print("1. Review model performance metrics")
        print("2. Analyze feature importance")
        print("3. Fine-tune hyperparameters")
        print("4. Deploy best model for prediction")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in main pipeline: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Exoplanet detection project completed successfully!")
    else:
        print("\n💥 Project execution failed. Please check the logs above.")
