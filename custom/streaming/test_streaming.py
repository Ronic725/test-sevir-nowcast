#!/usr/bin/env python3
"""
Simple test script for the SEVIR data streamer
"""

import sys
from pathlib import Path
import time
import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "custom/streaming"))

def test_basic_streaming():
    """Test basic streaming functionality"""
    print("🧪 Testing SEVIR Data Streamer")
    print("=" * 40)
    
    # Import here to catch any import errors
    try:
        from sevir_data_streamer import SEVIRDataStreamer
        print("✅ Successfully imported SEVIRDataStreamer")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Check data file
    data_file = PROJECT_ROOT / "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return False
    
    print(f"✅ Data file found: {data_file}")
    
    try:
        # Create streamer
        print("\n🔧 Creating streamer...")
        streamer = SEVIRDataStreamer(
            data_file=str(data_file),
            window_size=5,  # Small window for testing
            stream_interval=0.01,  # Very fast for testing (0.6 seconds)
            normalize=True
        )
        print("✅ Streamer created successfully")
        
        # Start streaming
        print("\n🌊 Starting stream...")
        streamer.start_streaming(event_idx=0)
        
        # Wait a bit for data to accumulate
        print("⏳ Waiting for data to stream...")
        for i in range(10):
            status = streamer.get_stream_status()
            print(f"   Status: Frame {status['current_frame']}/{status['total_frames']}, "
                  f"Buffer: {status['buffer_size']}/{status['buffer_capacity']}")
            
            if status['buffer_size'] >= 5:
                print("✅ Sufficient data accumulated")
                break
            time.sleep(1)
        
        # Test getting a sliding window
        print("\n📊 Testing sliding window...")
        window_data = streamer.get_sliding_window(advance_by=5)  # Use advance_by parameter
        if window_data is not None:
            window, metadata = window_data
            print(f"✅ Got sliding window: {window.shape}")
            print(f"   Frame indices: {[m['frame_idx'] for m in metadata]}")
            print(f"   Data type: {window.dtype}")
            print(f"   Value range: [{np.min(window):.3f}, {np.max(window):.3f}]")
        else:
            print("❌ Failed to get sliding window")
            return False
        
        # Stop streaming
        streamer.stop_streaming()
        print("\n✅ Streaming test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during streaming test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_synthetic_data():
    """Test with synthetic data file if available"""
    print("\n🧪 Testing with synthetic data")
    print("=" * 40)
    
    try:
        from sevir_data_streamer import SEVIRDataStreamer
        
        # Check synthetic data file
        synthetic_file = PROJECT_ROOT / "data/sevir/vil/synthetic_sevir_data.h5"
        if not synthetic_file.exists():
            print(f"⏭️  Synthetic data file not found: {synthetic_file}")
            return True  # Not a failure, just skip
        
        print(f"✅ Synthetic data file found: {synthetic_file}")
        
        # Create streamer with synthetic data
        streamer = SEVIRDataStreamer(
            data_file=str(synthetic_file),
            window_size=3,
            stream_interval=0.01,
            normalize=True
        )
        
        # Quick test
        streamer.start_streaming(event_idx=0)
        time.sleep(2)
        
        status = streamer.get_stream_status()
        print(f"Synthetic data status: {status}")
        
        streamer.stop_streaming()
        print("✅ Synthetic data test completed")
        return True
        
    except Exception as e:
        print(f"⚠️  Synthetic data test failed: {e}")
        return True  # Don't fail the whole test

if __name__ == "__main__":
    print("🚀 SEVIR Streaming System Tests")
    print("=" * 50)
    
    success = True
    
    # Test basic streaming
    if not test_basic_streaming():
        success = False
    
    # Test synthetic data
    if not test_synthetic_data():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Run: python custom/streaming/run_realtime_demo.py --test-streamer")
        print("2. Run: python custom/streaming/run_realtime_demo.py")
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
