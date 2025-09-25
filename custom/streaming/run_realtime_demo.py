#!/usr/bin/env python3
"""
Demo script for real-time SEVIR prediction system
"""

import sys
from pathlib import Path
import time

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "custom/streaming"))

from sevir_data_streamer import SEVIRDataStreamer
from realtime_predictor import RealtimePredictor

def run_demo():
    """Run a simple demo of the real-time system"""
    print("🚀 SEVIR Real-time Prediction Demo")
    print("=" * 50)
    
    # Paths - Update these based on your available files
    data_file = PROJECT_ROOT / "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
    
    # Try to find the latest trained model
    models_dir = PROJECT_ROOT / "models"
    model_file = None
    
    # Check for different model directories
    model_dirs = [
        models_dir / "trained_mse_20250924_202640",
        models_dir / "trained_mse_20250924_164649", 
        models_dir / "trained_mse_20250918_120133",
        models_dir / "nowcast"
    ]
    
    for model_dir in model_dirs:
        potential_models = [
            model_dir / "model_mse.h5",
            model_dir / "mse_model.h5"
        ]
        for potential_model in potential_models:
            if potential_model.exists():
                model_file = potential_model
                break
        if model_file:
            break
    
    if not data_file.exists():
        print(f"❌ Please ensure SEVIR data exists at: {data_file}")
        print("💡 You may need to run the data download script first")
        return
    
    if not model_file or not model_file.exists():
        print(f"❌ No trained model found in models directory")
        print("💡 Available model directories:")
        for d in models_dir.iterdir():
            if d.is_dir():
                print(f"   - {d.name}")
        return
    
    print(f"✅ Using data file: {data_file}")
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
    
    args = parser.parse_args()
    
    if args.test_streamer:
        test_streamer_only()
    else:
        run_demo()
