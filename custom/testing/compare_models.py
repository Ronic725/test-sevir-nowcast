#!/usr/bin/env python3
"""
Compare all trained models quickly
"""

import os
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

from test_actual_sevir import load_real_sevir_data, SEVIR_SCALE

# Project paths from config
MODELS_DIR = paths.models

def compare_all_models():
    """Test all available models and compare results"""
    print("🔍 Comparing All Trained Models")
    print("=" * 60)
    
    model_dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and d.name.startswith('trained_')]
    
    if not model_dirs:
        print("❌ No trained models found")
        return
    
    print(f"📁 Found {len(model_dirs)} trained models")
    
    # Load test data once (small sample for speed)
    print("\n📡 Loading SEVIR test data...")
    X_test, y_test = load_real_sevir_data(num_samples=3)  # Small sample for speed
    if X_test is None:
        print("❌ Failed to load test data")
        return
    
    results = []
    
    print(f"\n🧪 Testing models...")
    for i, model_dir in enumerate(model_dirs):
        model_path = model_dir / "model_mse.h5"
        if not model_path.exists():
            print(f"⚠️  {model_dir.name}: model_mse.h5 not found")
            continue
            
        print(f"\n{i+1:2d}. Testing {model_dir.name}...")
        
        try:
            # Load model
            model = tf.keras.models.load_model(str(model_path), compile=False)
            
            # Make predictions
            y_pred = model.predict(X_test, batch_size=1, verbose=0)
            
            # Calculate metrics
            mae = np.mean(np.abs(y_pred - y_test))
            mse = np.mean((y_pred - y_test)**2)
            
            # Get model info
            params = model.count_params()
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
            
            results.append({
                'name': model_dir.name,
                'mae': mae,
                'mse': mse,
                'mae_dbz': mae * SEVIR_SCALE,
                'mse_dbz': mse * SEVIR_SCALE**2,
                'params': params,
                'size_mb': model_size_mb,
                'date': mtime
            })
            
            print(f"     ✅ MAE: {mae * SEVIR_SCALE:.1f} dBZ, Size: {model_size_mb:.1f}MB")
            
        except Exception as e:
            print(f"     ❌ Failed: {e}")
    
    if not results:
        print("❌ No models could be tested")
        return
    
    # Show comparison
    print(f"\n📊 MODEL COMPARISON RESULTS:")
    print("=" * 80)
    
    # Sort by MAE (best first)
    results.sort(key=lambda x: x['mae'])
    
    print(f"{'Rank':<4} {'Model Name':<30} {'MAE (dBZ)':<10} {'Size (MB)':<10} {'Date':<12}")
    print("-" * 80)
    
    for i, result in enumerate(results):
        rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
        
        print(f"{rank_emoji:<4} {result['name']:<30} {result['mae_dbz']:<10.1f} "
              f"{result['size_mb']:<10.1f} {result['date'].strftime('%m-%d %H:%M'):<12}")
    
    # Show detailed comparison of top 3
    print(f"\n🏆 TOP 3 MODELS DETAILED:")
    print("-" * 50)
    
    for i, result in enumerate(results[:3]):
        rank = ["🥇 BEST", "🥈 2nd", "🥉 3rd"][i]
        print(f"\n{rank}: {result['name']}")
        print(f"   📈 MAE: {result['mae_dbz']:.2f} dBZ (normalized: {result['mae']:.4f})")
        print(f"   📈 MSE: {result['mse_dbz']:.1f} dBZ² (normalized: {result['mse']:.4f})")
        print(f"   🔧 Parameters: {result['params']:,}")
        print(f"   📁 Model size: {result['size_mb']:.1f} MB")
        print(f"   📅 Created: {result['date'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Performance analysis
    best_mae = results[0]['mae_dbz']
    worst_mae = results[-1]['mae_dbz']
    improvement = ((worst_mae - best_mae) / worst_mae) * 100
    
    print(f"\n📊 ANALYSIS:")
    print(f"   Best model MAE: {best_mae:.1f} dBZ")
    print(f"   Worst model MAE: {worst_mae:.1f} dBZ")
    print(f"   Best vs Worst improvement: {improvement:.1f}%")
    
    if best_mae < 10:
        print("   🌟 Best model shows excellent performance!")
    elif best_mae < 15:
        print("   👍 Best model shows good performance!")
    else:
        print("   ⚠️  Models may need more training!")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Use model: {results[0]['name']}")
    print(f"   Command: python test_actual_sevir.py --model {results[0]['name']}")

def main():
    compare_all_models()

if __name__ == "__main__":
    main()
