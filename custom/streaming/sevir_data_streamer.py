#!/usr/bin/env python3
"""
SEVIR Data Streaming Simulator
Simulates real-time weather radar data stream from SEVIR dataset
"""

import os
import sys
import numpy as np
import h5py
import time
import threading
import queue
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Generator

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

# SEVIR constants from config
SEVIR_MEAN = paths.sevir_mean
SEVIR_SCALE = paths.sevir_scale

class SEVIRDataStreamer:
    """
    Simulates real-time data streaming from SEVIR dataset
    """
    
    def __init__(self, 
                 data_file: str,
                 window_size: int = 13,
                 stream_interval: float = 5.0,  # 5 minutes between frames
                 normalize: bool = True,
                 buffer_size: int = 100):
        """
        Initialize the data streamer
        
        Args:
            data_file: Path to SEVIR H5 file
            window_size: Size of sliding window (should match model input)
            stream_interval: Time interval between frames in minutes
            normalize: Whether to normalize data using SEVIR constants
            buffer_size: Maximum size of data buffer
        """
        self.data_file = Path(data_file)
        self.window_size = window_size
        self.stream_interval = stream_interval
        self.normalize = normalize
        self.buffer_size = buffer_size
        
        # Internal state
        self.data_buffer = queue.Queue(maxsize=buffer_size)
        self.current_event_idx = 0
        self.current_frame_idx = 0
        self.is_streaming = False
        self.stream_thread = None
        
        # Load data
        self._load_sevir_data()
        
    def _load_sevir_data(self):
        """Load SEVIR data from H5 file"""
        print(f"📡 Loading SEVIR data from: {self.data_file}")
        
        if not self.data_file.exists():
            raise FileNotFoundError(f"SEVIR data file not found: {self.data_file}")
            
        try:
            with h5py.File(self.data_file, 'r') as f:
                # Check available datasets
                print(f"Available datasets: {list(f.keys())}")
                
                # Load VIL data - adjust key names based on actual file structure
                if 'vil' in f:
                    self.sevir_data = f['vil'][:]
                elif 'VIL' in f:
                    self.sevir_data = f['VIL'][:]
                elif 'data' in f:
                    self.sevir_data = f['data'][:]
                else:
                    # Try first available dataset
                    key = list(f.keys())[0]
                    print(f"Using dataset: {key}")
                    self.sevir_data = f[key][:]
                    
                print(f"✅ Loaded SEVIR data shape: {self.sevir_data.shape}")
                
                # Validate data shape
                if len(self.sevir_data.shape) != 4:
                    raise ValueError(f"Expected 4D data, got shape {self.sevir_data.shape}")
                    
                self.num_events, self.height, self.width, self.num_frames = self.sevir_data.shape
                print(f"   Events: {self.num_events}, Size: {self.height}x{self.width}, Frames: {self.num_frames}")
                
        except Exception as e:
            print(f"❌ Error loading SEVIR data: {e}")
            raise
    
    def start_streaming(self, event_idx: Optional[int] = None):
        """
        Start streaming data from specified event
        
        Args:
            event_idx: Event index to stream from (None for random)
        """
        if self.is_streaming:
            print("⚠️  Stream already running")
            return
            
        if event_idx is None:
            event_idx = np.random.randint(0, self.num_events)
            
        self.current_event_idx = event_idx
        self.current_frame_idx = 0
        self.is_streaming = True
        
        print(f"🌊 Starting data stream from event {event_idx}")
        
        # Start streaming thread
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        
    def stop_streaming(self):
        """Stop the data stream"""
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join()
        print("🛑 Data stream stopped")
        
    def _stream_worker(self):
        """Worker thread for streaming data"""
        while self.is_streaming and self.current_frame_idx < self.num_frames:
            try:
                # Get current frame
                frame = self.sevir_data[self.current_event_idx, :, :, self.current_frame_idx]
                
                # Normalize if requested
                if self.normalize:
                    frame = (frame - SEVIR_MEAN) / SEVIR_SCALE
                
                # Create data packet
                data_packet = {
                    'frame': frame,
                    'event_idx': self.current_event_idx,
                    'frame_idx': self.current_frame_idx,
                    'timestamp': datetime.now(),
                    'is_normalized': self.normalize
                }
                
                # Add to buffer (non-blocking)
                if not self.data_buffer.full():
                    self.data_buffer.put(data_packet)
                    print(f"📊 Streamed frame {self.current_frame_idx + 1}/{self.num_frames} "
                          f"from event {self.current_event_idx}")
                else:
                    print("⚠️  Buffer full, dropping frame")
                
                self.current_frame_idx += 1
                
                # Wait for next frame (simulate real-time)
                time.sleep(self.stream_interval * 60)  # Convert minutes to seconds
                
            except Exception as e:
                print(f"❌ Error in stream worker: {e}")
                break
                
        self.is_streaming = False
        print("✅ Stream completed")
    
    def get_next_frame(self, timeout: float = 10.0) -> Optional[dict]:
        """
        Get the next frame from stream
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Data packet or None if timeout
        """
        try:
            return self.data_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def peek_sliding_window(self) -> Optional[np.ndarray]:
        """
        Peek at current sliding window without consuming frames
        
        Returns:
            Sliding window array of shape (height, width, window_size) or None
        """
        if self.data_buffer.qsize() < self.window_size:
            return None
            
        # Get frames without removing them
        frames = []
        temp_items = []
        
        try:
            for _ in range(min(self.window_size, self.data_buffer.qsize())):
                item = self.data_buffer.get_nowait()
                temp_items.append(item)
                frames.append(item['frame'])
            
            # Put items back
            for item in temp_items:
                self.data_buffer.put_nowait(item)
                
            if len(frames) == self.window_size:
                return np.stack(frames, axis=-1)
                
        except queue.Empty:
            # Put back any items we got
            for item in temp_items:
                self.data_buffer.put_nowait(item)
                
        return None
    
    def get_sliding_window(self, advance_by: int = 1) -> Optional[Tuple[np.ndarray, list]]:
        """
        Get sliding window for prediction with configurable advance
        
        Args:
            advance_by: Number of frames to advance (1 = maximum overlap, window_size = no overlap)
        
        Returns:
            Tuple of (window_array, metadata_list) or None
        """
        if self.data_buffer.qsize() < self.window_size:
            return None
            
        # Get all items from buffer without consuming them initially
        all_items = []
        temp_items = []
        
        try:
            # Extract all items to examine them
            while not self.data_buffer.empty():
                item = self.data_buffer.get_nowait()
                temp_items.append(item)
                all_items.append(item)
            
            # Check if we have enough items for a window
            if len(all_items) < self.window_size:
                # Put everything back
                for item in temp_items:
                    self.data_buffer.put_nowait(item)
                return None
            
            # Create window from first window_size items
            frames = []
            metadata = []
            
            for i in range(self.window_size):
                packet = all_items[i]
                frames.append(packet['frame'])
                metadata.append({
                    'event_idx': packet['event_idx'],
                    'frame_idx': packet['frame_idx'],
                    'timestamp': packet['timestamp']
                })
            
            # Put back items we want to keep (advance by advance_by frames)
            items_to_keep = all_items[advance_by:]
            for item in items_to_keep:
                self.data_buffer.put_nowait(item)
            
            window = np.stack(frames, axis=-1)
            return window, metadata
            
        except queue.Empty:
            # Put back any items we managed to get
            for item in temp_items:
                self.data_buffer.put_nowait(item)
            return None
    
    def get_stream_status(self) -> dict:
        """Get current stream status"""
        return {
            'is_streaming': self.is_streaming,
            'current_event': self.current_event_idx,
            'current_frame': self.current_frame_idx,
            'total_frames': self.num_frames,
            'buffer_size': self.data_buffer.qsize(),
            'buffer_capacity': self.buffer_size,
            'window_size': self.window_size
        }


if __name__ == "__main__":
    # Test the streamer
    data_file = PROJECT_ROOT / "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
    
    if data_file.exists():
        streamer = SEVIRDataStreamer(
            data_file=str(data_file),
            window_size=13,
            stream_interval=0.1,  # Fast for testing
            normalize=True
        )
        
        # Start streaming
        streamer.start_streaming(event_idx=0)
        
        # Monitor for a while
        for i in range(20):
            status = streamer.get_stream_status()
            print(f"Status: {status}")
            
            if status['buffer_size'] >= 13:
                window, metadata = streamer.get_sliding_window()
                if window is not None:
                    print(f"Got sliding window: {window.shape}")
                    print(f"Frame range: {metadata[0]['frame_idx']} to {metadata[-1]['frame_idx']}")
            
            time.sleep(1)
        
        streamer.stop_streaming()
    else:
        print(f"❌ Test data file not found: {data_file}")
