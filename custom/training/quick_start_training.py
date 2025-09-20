#!/usr/bin/env python3
"""
Quick Start Guide for Limited Data Training
Choose your training scenario based on your needs and resources
"""

import os
import subprocess
import sys

# Add parent directories to path to access original repo modules
sys.path.append('../../')
sys.path.append('../../src/')

def print_banner():
    print("🌩️  SEVIR Limited Data Training - Quick Start")
    print("=" * 60)
    print("Perfect for M1 MacBook Pro with 16GB RAM")
    print("=" * 60)

def show_training_options():
    print("\n🎯 Training Options:\n")
    
    options = [
        {
            "name": "Quick Test (Recommended for first run)",
            "desc": "Small dataset, fast training, MSE loss",
            "cmd": "--num_samples 256 --epochs 5 --loss_type mse",
            "time": "~5-10 minutes",
            "memory": "Low"
        },
        {
            "name": "Medium Training", 
            "desc": "Moderate dataset, good results, MSE loss",
            "cmd": "--num_samples 1024 --epochs 15 --loss_type mse",
            "time": "~20-30 minutes",
            "memory": "Medium"
        },
        {
            "name": "Advanced Training",
            "desc": "Larger dataset, VGG loss for better quality",
            "cmd": "--num_samples 2048 --epochs 20 --loss_type vgg",
            "time": "~1-2 hours",
            "memory": "High"
        },
        {
            "name": "Custom Training",
            "desc": "Choose your own parameters",
            "cmd": "Custom parameters",
            "time": "Varies",
            "memory": "Depends on settings"
        }
    ]
    
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt['name']}")
        print(f"   📝 {opt['desc']}")
        print(f"   ⏱️  Time: {opt['time']}")
        print(f"   💾 Memory: {opt['memory']}")
        print(f"   🖥️  Command: python train_limited_data.py {opt['cmd']}")
        print()

def run_training(option):
    """Run training based on selected option"""
    commands = {
        1: ["python", "train_limited_data.py", "--num_samples", "256", "--epochs", "5", "--loss_type", "mse"],
        2: ["python", "train_limited_data.py", "--num_samples", "1024", "--epochs", "15", "--loss_type", "mse"],
        3: ["python", "train_limited_data.py", "--num_samples", "2048", "--epochs", "20", "--loss_type", "vgg"]
    }
    
    if option in commands:
        print(f"\n🚀 Starting training option {option}...")
        print("Command:", " ".join(commands[option]))
        print("\nPress Ctrl+C to stop training if needed.\n")
        
        try:
            subprocess.run(commands[option], check=True)
        except KeyboardInterrupt:
            print("\n⏹️  Training stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with error: {e}")
    elif option == 4:
        show_custom_options()
    else:
        print("❌ Invalid option")

def show_custom_options():
    print("\n🛠️  Custom Training Parameters:")
    print("\nAvailable options:")
    print("  --num_samples: Number of training samples (256, 512, 1024, 2048, etc.)")
    print("  --epochs: Number of training epochs (5, 10, 15, 20, etc.)")
    print("  --loss_type: Loss function (mse, mae, vgg)")
    print("  --data_type: Data type (synthetic, real)")
    print("  --validation_split: Validation ratio (0.2, 0.3, etc.)")
    print("\nExample:")
    print("  python train_limited_data.py --num_samples 512 --epochs 10 --loss_type mae")
    print("\nMemory Guidelines for M1 MacBook Pro (16GB):")
    print("  • num_samples ≤ 1024 + batch_size ≤ 8: Safe")
    print("  • num_samples ≤ 2048 + batch_size ≤ 4: Should work")
    print("  • num_samples > 2048: Monitor memory usage")

def check_system():
    """Check if system is ready for training"""
    print("\n🔍 System Check:")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} detected")
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU acceleration available: {len(gpus)} device(s)")
        else:
            print("⚠️  No GPU detected - training will use CPU")
            
    except ImportError:
        print("❌ TensorFlow not found - please install dependencies")
        return False
    
    try:
        import torch
        if torch.backends.mps.is_available():
            print("✅ PyTorch MPS (M1 GPU) support available")
        else:
            print("⚠️  PyTorch MPS not available")
    except ImportError:
        print("⚠️  PyTorch not found")
    
    return True

def main():
    print_banner()
    
    if not check_system():
        print("\n❌ System check failed. Please install required dependencies first.")
        return
    
    show_training_options()
    
    while True:
        try:
            choice = input("Choose an option (1-4) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("👋 Goodbye!")
                break
            
            option = int(choice)
            if 1 <= option <= 4:
                run_training(option)
                break
            else:
                print("❌ Please choose a number between 1-4")
                
        except ValueError:
            print("❌ Please enter a valid number or 'q'")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break

if __name__ == "__main__":
    main()
