#!/usr/bin/env python3
"""
Interactive model selection and testing on real SEVIR data
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import tensorflow as tf

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from test_actual_sevir import *

def select_model_interactively():
    """Let user choose which model to test"""
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Find all trained models
    model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith('trained_')]
    
    if not model_dirs:
        print("❌ No trained models found")
        return None
    
    # Sort by modification time (newest first)
    model_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print("🤖 Available trained models:")
    print("=" * 80)
    
    for i, model_dir in enumerate(model_dirs):
        model_path = model_dir / "model_mse.h5"
        if model_path.exists():
            # Get creation time
            mtime = datetime.fromtimestamp(model_dir.stat().st_mtime)
            
            # Get model file size
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            print(f"{i+1:2d}. {model_dir.name}")
            print(f"     📅 Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"     📁 Size: {model_size_mb:.1f} MB")
            print(f"     🗂️  Path: {model_path}")
            
            # Check if there's training history
            history_path = model_dir / "training_history.csv"
            if history_path.exists():
                print(f"     📊 Training history available")
            
            print()
        else:
            print(f"{i+1:2d}. {model_dir.name} (❌ model_mse.h5 not found)")
            print()
    
    # Get user choice
    try:
        print(f"📝 Choose model (1-{len(model_dirs)}) or 'q' to quit: ", end="")
        choice_input = input().strip().lower()
        
        if choice_input == 'q':
            print("👋 Goodbye!")
            return None
            
        choice = int(choice_input) - 1
        if 0 <= choice < len(model_dirs):
            selected_dir = model_dirs[choice]
            model_path = selected_dir / "model_mse.h5"
            
            if not model_path.exists():
                print("❌ Selected model file doesn't exist!")
                return None
                
            return model_path, selected_dir.name
        else:
            print("❌ Invalid choice")
            return None
    except (ValueError, KeyboardInterrupt):
        print("\n❌ Invalid input or cancelled")
        return None

def show_model_details(model_path, model_name):
    """Show detailed information about the selected model"""
    print(f"\n🔍 Model Details: {model_name}")
    print("=" * 50)
    
    model_dir = model_path.parent
    
    # Show training history if available
    history_path = model_dir / "training_history.csv"
    if history_path.exists():
        try:
            import pandas as pd
            history = pd.read_csv(history_path)
            print(f"📈 Training completed after {len(history)} epochs")
            
            if 'loss' in history.columns:
                final_loss = history['loss'].iloc[-1]
                print(f"📉 Final training loss: {final_loss:.6f}")
            
            if 'val_loss' in history.columns:
                final_val_loss = history['val_loss'].iloc[-1]
                best_val_loss = history['val_loss'].min()
                print(f"📉 Final validation loss: {final_val_loss:.6f}")
                print(f"🎯 Best validation loss: {best_val_loss:.6f}")
        except Exception as e:
            print(f"⚠️  Could not read training history: {e}")
    
    # Model file info
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
    print(f"📁 Model file size: {model_size_mb:.1f} MB")
    print(f"📅 Last modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

def main():
    print("🌩️  Interactive Model Testing on Real SEVIR Data")
    print("=" * 60)
    
    # Select model
    selection_result = select_model_interactively()
    if not selection_result:
        return
    
    model_path, model_name = selection_result
    
    print(f"\n🎯 Selected model: {model_name}")
    
    # Show model details
    show_model_details(model_path, model_name)
    
    # Confirm before proceeding
    proceed = input("🚀 Proceed with testing? (y/n): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("👋 Testing cancelled")
        return
    
    # Load the selected model
    try:
        print(f"\n🤖 Loading model from: {model_path}")
        model = tf.keras.models.load_model(str(model_path), compile=False)
        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {model.count_params():,}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load SEVIR data
    print("\n" + "="*60)
    X_test, y_test = load_real_sevir_data(num_samples=NUM_SAMPLES)
    if X_test is None:
        print("❌ Failed to load SEVIR data. Exiting.")
        return
    
    # Test the model
    print("\n" + "="*60)
    y_pred, mse, mae, y_pred_dbz, y_test_dbz = test_on_real_sevir(model, X_test, y_test)
    if y_pred is None:
        print("❌ Failed to make predictions. Exiting.")
        return
    
    # Compare with baseline
    y_baseline, mse_baseline, mae_baseline = compare_with_baseline(X_test, y_test)
    
    # Visualize results
    viz_file = visualize_real_sevir_results(X_test, y_test, y_pred, y_baseline)
    
    # Frame-by-frame analysis
    analyze_frame_by_frame(y_test, y_pred, y_baseline)
    
    # Final assessment
    improvement = ((mae_baseline - mae) / mae_baseline) * 100
    
    print(f"\n🎯 REAL SEVIR TEST SUMMARY for {model_name}:")
    print("=" * 80)
    print(f"   🤖 Our Model:")
    print(f"      MAE: {mae:.4f} (normalized), {mae * SEVIR_SCALE:.1f} dBZ")
    print(f"      MSE: {mse:.4f} (normalized), {mse * SEVIR_SCALE**2:.1f} dBZ²")
    
    print(f"   📊 Persistence Baseline:")
    print(f"      MAE: {mae_baseline:.4f} (normalized), {mae_baseline * SEVIR_SCALE:.1f} dBZ")
    print(f"      MSE: {mse_baseline:.4f} (normalized), {mse_baseline * SEVIR_SCALE**2:.1f} dBZ²")
    
    print(f"   📈 Improvement over baseline: {improvement:+.1f}%")
    
    # Performance interpretation
    if mae < 0.1:  # < ~4.8 dBZ
        print("🌟 Excellent performance on real SEVIR data!")
    elif mae < 0.2:  # < ~9.5 dBZ
        print("👍 Good performance on real SEVIR data!")
    elif mae < 0.3:  # < ~14.3 dBZ
        print("👌 Acceptable performance on real SEVIR data!")
    elif improvement > 0:
        print("📈 Model beats baseline but needs improvement!")
    else:
        print("⚠️  Model underperforms - needs more training on real data!")
    
    print(f"\n📊 Visualization saved: {viz_file}")
    
    # Save results summary
    results_file = RESULTS_DIR / f"test_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    RESULTS_DIR.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"Real SEVIR Test Results\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Samples Tested: {NUM_SAMPLES}\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"  MAE: {mae:.4f} (normalized), {mae * SEVIR_SCALE:.1f} dBZ\n")
        f.write(f"  MSE: {mse:.4f} (normalized), {mse * SEVIR_SCALE**2:.1f} dBZ²\n")
        f.write(f"  Baseline MAE: {mae_baseline:.4f} (normalized), {mae_baseline * SEVIR_SCALE:.1f} dBZ\n")
        f.write(f"  Improvement: {improvement:+.1f}%\n")
        f.write(f"  Visualization: {viz_file}\n")
    
    print(f"📝 Results summary saved: {results_file}")

if __name__ == "__main__":
    main()
