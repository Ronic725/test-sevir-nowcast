#!/usr/bin/env python3
"""
Demo script for real-time SEVIR prediction system
"""

import sys
from pathlib import Path
import time

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

# Now import from custom modules
sys.path.insert(0, str(PROJECT_ROOT / "custom/streaming"))
from sevir_data_streamer import SEVIRDataStreamer
from realtime_predictor import RealtimePredictor

def find_available_models():
    """Find all available trained models"""
    models_dir = paths.models
    available_models = []
    
    # Check for different model directories
    model_dirs = [
        models_dir / "trained_mse_20250918_120133",  # The good model first
        models_dir / "trained_mse_20250924_164649", 
        models_dir / "trained_mse_20250924_202640",
        models_dir / "nowcast"
    ]
    
    for model_dir in model_dirs:
        potential_models = [
            model_dir / "model_mse.h5",
            model_dir / "mse_model.h5"
        ]
        for potential_model in potential_models:
            if potential_model.exists():
                available_models.append({
                    'path': potential_model,
                    'dir': model_dir.name,
                    'name': potential_model.name
                })
                break
    
    return available_models

def select_model(model_name=None):
    """Select a model either by name or interactively"""
    available_models = find_available_models()
    
    if not available_models:
        print("❌ No trained models found in models directory")
        return None
    
    if model_name:
        # Find specific model by name
        for model in available_models:
            if model_name in model['dir']:
                print(f"✅ Selected model: {model['dir']}")
                return model['path']
        print(f"❌ Model '{model_name}' not found")
        return None
    
    # Interactive selection
    print("📁 Available trained models:")
    for i, model in enumerate(available_models):
        print(f"   {i+1}. {model['dir']} ({model['name']})")
    
    # Default to the first (recommended) model
    print(f"🎯 Using recommended model: {available_models[0]['dir']}")
    return available_models[0]['path']

def run_demo(model_name=None):
    """Run a simple demo of the real-time system"""
    print("🚀 SEVIR Real-time Prediction Demo")
    print("=" * 50)
    
    # Use configured SEVIR data path
    data_file = paths.sevir_vil_file
    
    # Select model
    model_file = select_model(model_name)
    
    if not data_file.exists():
        print(f"❌ Please ensure SEVIR data exists at: {data_file}")
        print("💡 You may need to run the data download script first")
        return

    if not model_file or not model_file.exists():
        print(f"❌ Selected model not found")
        return    print(f"✅ Using data file: {data_file}")
    print(f"✅ Using model: {model_file}")
    
    # Create streamer with fast interval for demo
    try:
        streamer = SEVIRDataStreamer(
            data_file=str(data_file),
            window_size=13,
            stream_interval=0.01,  # Very fast for demo (0.6 seconds between frames)
            normalize=True
        )
        print("✅ Data streamer initialized")
    except Exception as e:
        print(f"❌ Error creating streamer: {e}")
        return
    
    # Create predictor with overlapping sliding window
    try:
        predictor = RealtimePredictor(
            model_path=str(model_file),
            anomaly_threshold=70.0,  # Higher threshold due to model scaling issues
            save_predictions=True,
            advance_by=1  # Maximum overlap for realistic continuous prediction
        )
        print("✅ Predictor initialized")
    except Exception as e:
        print(f"❌ Error creating predictor: {e}")
        return
    
    # Start demo
    print("\n🌊 Starting streaming demo...")
    streamer.start_streaming(event_idx=5)  # Use event 5
    
    try:
        # Give some time for initial data to stream
        print("⏳ Waiting for initial data...")
        time.sleep(8)  # Wait longer for buffer to fill up
        
        # Process stream with more predictions for continuous feel
        predictor.process_stream(streamer, max_predictions=12)
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
    finally:
        streamer.stop_streaming()
        print("✅ Demo completed")

def test_streamer_only():
    """Test just the data streamer component"""
    print("🧪 Testing Data Streamer Only")
    print("=" * 40)
    
    data_file = PROJECT_ROOT / "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return
    
    try:
        streamer = SEVIRDataStreamer(
            data_file=str(data_file),
            window_size=13,
            stream_interval=0.05,  # 3 seconds for testing
            normalize=True
        )
        
        # Start streaming
        streamer.start_streaming(event_idx=0)
        
        # Monitor for a while
        for i in range(20):
            status = streamer.get_stream_status()
            print(f"📊 Status: Event {status['current_event']}, "
                  f"Frame {status['current_frame']}/{status['total_frames']}, "
                  f"Buffer: {status['buffer_size']}/{status['buffer_capacity']}")
            
            if status['buffer_size'] >= 13:
                window_data = streamer.get_sliding_window()
                if window_data is not None:
                    window, metadata = window_data
                    print(f"✅ Got sliding window: {window.shape}")
                    print(f"   Frame range: {metadata[0]['frame_idx']} to {metadata[-1]['frame_idx']}")
                    break
            
            time.sleep(1)
        
        streamer.stop_streaming()
        print("✅ Streamer test completed")
        
    except Exception as e:
        print(f"❌ Streamer test failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SEVIR Real-time Prediction Demo')
    parser.add_argument('--test-streamer', action='store_true', 
                       help='Test only the data streamer component')
    parser.add_argument('--model', type=str, 
                       help='Specific model directory name to use (partial name match)')
    
    args = parser.parse_args()
    
    if args.test_streamer:
        test_streamer_only()
    else:
        run_demo(args.model)
