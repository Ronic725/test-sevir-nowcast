#!/usr/bin/env python3
"""
Real-time Weather Prediction System
Continuously processes streaming SEVIR data and makes predictions
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import threading
import queue

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import centralized configuration
from config.project_paths import get_paths
paths = get_paths()
paths.setup_python_path()

from sevir_data_streamer import SEVIRDataStreamer

# SEVIR constants from config
SEVIR_MEAN = paths.sevir_mean
SEVIR_SCALE = paths.sevir_scale

class RealtimePredictor:
    """
    Real-time weather prediction system using sliding window approach
    """
    
    def __init__(self, 
                 model_path: str,
                 input_window_size: int = 13,
                 prediction_horizon: int = 12,
                 anomaly_threshold: float = 60.0,  # dBZ threshold for anomaly detection
                 save_predictions: bool = True,
                 advance_by: int = 1):  # How many frames to advance for sliding window
        """
        Initialize the real-time predictor
        
        Args:
            model_path: Path to trained model
            input_window_size: Size of input window for model
            prediction_horizon: Number of frames to predict
            anomaly_threshold: Threshold for anomaly detection (in dBZ)
            save_predictions: Whether to save prediction results
            advance_by: Number of frames to advance sliding window (1 = maximum overlap)
        """
        self.model_path = Path(model_path)
        self.input_window_size = input_window_size
        self.prediction_horizon = prediction_horizon
        self.anomaly_threshold = anomaly_threshold
        self.save_predictions = save_predictions
        self.advance_by = advance_by
        
        # Results storage
        self.predictions_history = []
        self.anomaly_events = []
        
        # Session info for consistent naming
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        # Load model
        self._load_model()
        
        # Create results directory
        if self.save_predictions:
            self.results_dir = PROJECT_ROOT / "custom/results/realtime_predictions"
            self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self):
        """Load the trained model"""
        print(f"🤖 Loading model from: {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            # Try different ways to load the model
            try:
                self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
            except AttributeError:
                # Alternative import method
                from keras.models import load_model
                self.model = load_model(str(self.model_path), compile=False)
            print(f"✅ Model loaded successfully")
            print(f"   Input shape: {self.model.input_shape}")
            print(f"   Output shape: {self.model.output_shape}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def predict_next_frames(self, input_window: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Make prediction for next frames
        
        Args:
            input_window: Input window of shape (height, width, window_size)
            
        Returns:
            Tuple of (predictions, prediction_info)
        """
        # Prepare input for model (add batch dimension)
        model_input = np.expand_dims(input_window, axis=0)
        
        # Make prediction
        start_time = time.time()
        predictions = self.model.predict(model_input, verbose=0)
        prediction_time = time.time() - start_time
        
        # Remove batch dimension
        predictions = predictions[0]
        
        # Prediction info
        pred_info = {
            'prediction_time': prediction_time,
            'input_shape': input_window.shape,
            'output_shape': predictions.shape,
            'timestamp': datetime.now(),
            'model_path': str(self.model_path)
        }
        
        return predictions, pred_info
    
    def detect_anomalies(self, predictions: np.ndarray, metadata: List[dict]) -> List[dict]:
        """
        Detect anomalies in predictions using threshold-based approach
        
        Args:
            predictions: Predicted frames (model output, range 0-1)
            metadata: Metadata for input frames
            
        Returns:
            List of anomaly events detected
        """
        anomalies = []
        
        # Convert to dBZ for threshold comparison
        predictions_dbz = predictions * SEVIR_SCALE + SEVIR_MEAN
        
        for frame_idx in range(predictions.shape[-1]):
            frame = predictions_dbz[:, :, frame_idx]
            max_intensity = np.max(frame)
            
            if max_intensity > self.anomaly_threshold:
                # Find locations of extreme values
                extreme_locations = np.where(frame > self.anomaly_threshold)
                
                anomaly = {
                    'frame_idx': frame_idx,
                    'max_intensity': max_intensity,
                    'threshold': self.anomaly_threshold,
                    'extreme_pixel_count': len(extreme_locations[0]),
                    'extreme_locations': (extreme_locations[0].tolist(), extreme_locations[1].tolist()),
                    'detection_time': datetime.now(),
                    'input_metadata': metadata
                }
                
                anomalies.append(anomaly)
                print(f"🚨 ANOMALY DETECTED: Frame {frame_idx}, Max intensity {max_intensity:.1f} dBZ")
        
        return anomalies
    
    def visualize_prediction(self, 
                           input_window: np.ndarray, 
                           predictions: np.ndarray,
                           metadata: List[dict],
                           prediction_id: int,
                           anomalies: Optional[List[dict]] = None,
                           save_plot: bool = True) -> Optional[str]:
        """
        Visualize real-time prediction results with proper indexing
        
        Args:
            input_window: Input window
            predictions: Predicted frames
            metadata: Input metadata
            prediction_id: Current prediction number
            anomalies: Detected anomalies
            save_plot: Whether to save the plot
            
        Returns:
            Path to saved plot or None
        """
        # Convert to dBZ for visualization
        input_dbz = input_window * SEVIR_SCALE + SEVIR_MEAN
        pred_dbz = predictions * SEVIR_SCALE + SEVIR_MEAN
        
        # Clip extreme values for better visualization
        input_dbz = np.clip(input_dbz, 0, 75)
        pred_dbz = np.clip(pred_dbz, 0, 75)
        
        # Get actual frame indices from metadata
        start_frame = metadata[0]['frame_idx']
        end_frame = metadata[-1]['frame_idx']
        
        # Create figure
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        fig.suptitle(f"Real-time Prediction #{prediction_id:03d} - Session {self.session_id}", 
                     fontsize=16, y=0.98)
        
        # Show last 4 input frames with correct indices
        input_frames_to_show = [-4, -3, -2, -1]
        for i, frame_idx in enumerate(input_frames_to_show):
            ax = axes[0, i]
            if abs(frame_idx) <= input_window.shape[-1]:
                im = ax.imshow(input_dbz[:, :, frame_idx], cmap='viridis', vmin=0, vmax=70)
                actual_frame_num = start_frame + (self.input_window_size + frame_idx)
                ax.set_title(f'Input T-{abs(frame_idx)}\n(Frame {actual_frame_num})', fontsize=10)
            else:
                ax.axis('off')
                ax.set_title('N/A', fontsize=10)
            ax.axis('off')
        
        # Show first 4 predicted frames with correct indices
        pred_frames_to_show = [0, 1, 2, 3]
        for i, frame_idx in enumerate(pred_frames_to_show):
            ax = axes[1, i]
            if frame_idx < pred_dbz.shape[-1]:
                im = ax.imshow(pred_dbz[:, :, frame_idx], cmap='viridis', vmin=0, vmax=70) 
                predicted_frame_num = end_frame + 1 + frame_idx
                ax.set_title(f'Predict T+{frame_idx + 1}\n(Frame {predicted_frame_num})', fontsize=10)
                
                # Highlight anomalies if present
                if anomalies:
                    for anomaly in anomalies:
                        if anomaly['frame_idx'] == frame_idx:
                            locations = anomaly['extreme_locations']
                            if len(locations[0]) > 0:  # Only if there are actual locations
                                ax.scatter(locations[1], locations[0], c='red', s=0.3, alpha=0.6, marker='x')
                                # Add text box with background for better visibility
                                ax.text(5, 15, f"⚠️{anomaly['max_intensity']:.0f}dBZ", 
                                       color='white', fontsize=8, weight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
            else:
                ax.axis('off')
                ax.set_title('N/A', fontsize=10)
            ax.axis('off')
        
        # Show last 2 predicted frames with correct indices
        for i, frame_idx in enumerate([10, 11], start=4):
            ax = axes[1, i]
            if frame_idx < pred_dbz.shape[-1]:
                im = ax.imshow(pred_dbz[:, :, frame_idx], cmap='viridis', vmin=0, vmax=70)
                predicted_frame_num = end_frame + 1 + frame_idx
                ax.set_title(f'Predict T+{frame_idx + 1}\n(Frame {predicted_frame_num})', fontsize=10)
                
                # Highlight anomalies
                if anomalies:
                    for anomaly in anomalies:
                        if anomaly['frame_idx'] == frame_idx:
                            locations = anomaly['extreme_locations']
                            if len(locations[0]) > 0:  # Only if there are actual locations
                                ax.scatter(locations[1], locations[0], c='red', s=0.3, alpha=0.6, marker='x')
                                # Add text box with background for better visibility
                                ax.text(5, 15, f"⚠️{anomaly['max_intensity']:.0f}dBZ", 
                                       color='white', fontsize=8, weight='bold',
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
            else:
                ax.axis('off')
                ax.set_title('N/A', fontsize=10)
            ax.axis('off')
        
        # Add colorbar - positioned to not overlap with plots
        # Ensure we have an image to create colorbar from
        if pred_dbz.shape[-1] > 0:
            # Use the last valid image for colorbar
            im = axes[1, 0].imshow(pred_dbz[:, :, 0], cmap='viridis', vmin=0, vmax=70)
            axes[1, 0].clear()  # Clear it since we're just using it for colorbar
            axes[1, 0].imshow(pred_dbz[:, :, 0], cmap='viridis', vmin=0, vmax=70)
            axes[1, 0].set_title(f'Predict T+1\n(Frame {end_frame + 1})', fontsize=10)
            axes[1, 0].axis('off')
            
            # Create a separate axis for colorbar to avoid overlap
            cbar_ax = fig.add_axes((0.15, 0.12, 0.7, 0.03))  # [left, bottom, width, height]
            cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal', 
                               label='Radar Reflectivity (dBZ)')
            cbar.ax.tick_params(labelsize=10)
        
        # Add info text with proper spacing
        info_text = f"Input: Frames {start_frame}-{end_frame} | "
        info_text += f"Predict: Frames {end_frame + 1}-{end_frame + pred_dbz.shape[-1]} | "
        if anomalies:
            info_text += f"🚨 {len(anomalies)} ANOMALIES DETECTED"
        else:
            info_text += "✅ No anomalies detected"
        
        plt.figtext(0.5, 0.05, info_text, ha='center', fontsize=11, weight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.18)  # More space at bottom for colorbar
        
        # Save plot with consistent naming
        if save_plot and self.save_predictions:
            filename = f"realtime_pred_{self.session_id}_{prediction_id:03d}.png"
            save_path = self.results_dir / filename
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualization saved to: {save_path}")
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return None
    
    def process_stream(self, streamer: SEVIRDataStreamer, max_predictions: int = 15):
        """
        Process data stream and make continuous predictions with overlapping sliding window
        
        Args:
            streamer: Data streamer instance
            max_predictions: Maximum number of predictions to make
        """
        print(f"🔄 Starting real-time prediction process...")
        print(f"   Session ID: {self.session_id}")
        print(f"   Max predictions: {max_predictions}")
        print(f"   Window size: {self.input_window_size}")
        print(f"   Window advance: {self.advance_by} frames")
        print(f"   Anomaly threshold: {self.anomaly_threshold} dBZ")
        
        prediction_count = 0
        consecutive_waits = 0
        max_consecutive_waits = 10
        
        while prediction_count < max_predictions and streamer.is_streaming:
            # Get overlapping sliding window
            window_data = streamer.get_sliding_window(advance_by=self.advance_by)
            
            if window_data is None:
                consecutive_waits += 1
                if consecutive_waits <= max_consecutive_waits:
                    print(f"⏳ Waiting for sufficient data... ({consecutive_waits}/{max_consecutive_waits})")
                    time.sleep(1)
                    continue
                else:
                    print("⚠️  Max wait time exceeded, stopping predictions")
                    break
            
            # Reset wait counter
            consecutive_waits = 0
            
            input_window, metadata = window_data
            prediction_count += 1
            
            print(f"\n🔮 Making prediction {prediction_count}/{max_predictions}")
            print(f"   Input frames: {metadata[0]['frame_idx']} to {metadata[-1]['frame_idx']}")
            
            # Make prediction
            predictions, pred_info = self.predict_next_frames(input_window)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(predictions, metadata)
            
            # Store results
            result = {
                'prediction_id': prediction_count,
                'input_metadata': metadata,
                'predictions': predictions,
                'pred_info': pred_info,
                'anomalies': anomalies
            }
            self.predictions_history.append(result)
            
            if anomalies:
                self.anomaly_events.extend(anomalies)
            
            # Visualize with prediction ID
            if self.save_predictions:
                self.visualize_prediction(input_window, predictions, metadata, prediction_count, anomalies)
            
            print(f"✅ Prediction completed in {pred_info['prediction_time']:.3f}s")
            if anomalies:
                print(f"🚨 {len(anomalies)} anomalies detected!")
            
            # Brief pause before next prediction (shorter for more realistic streaming)
            time.sleep(0.5)
        
        print(f"\n📊 Real-time prediction completed:")
        print(f"   Session ID: {self.session_id}")
        print(f"   Total predictions: {len(self.predictions_history)}")
        print(f"   Total anomalies: {len(self.anomaly_events)}")
        print(f"   Results saved to: {self.results_dir}")


def main():
    """Main function to run real-time prediction system"""
    print("🌩️  SEVIR Real-time Prediction System")
    print("=" * 60)
    
    # Configuration
    data_file = PROJECT_ROOT / "data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5"
    models_dir = PROJECT_ROOT / "models"
    
    # Find latest model
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('trained_')]
    if not model_dirs:
        print("❌ No trained model found")
        return
    
    latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
    model_path = latest_model_dir / "model_mse.h5"
    
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not data_file.exists():
        print(f"❌ SEVIR data file not found: {data_file}")
        return
    
    # Initialize components
    print("🔧 Initializing components...")
    
    # Data streamer - faster streaming for more realistic continuous prediction
    streamer = SEVIRDataStreamer(
        data_file=str(data_file),
        window_size=13,
        stream_interval=0.05,  # 3 seconds for demo (much faster for continuous feel)
        normalize=True
    )
    
    # Predictor with overlapping sliding window
    predictor = RealtimePredictor(
        model_path=str(model_path),
        input_window_size=13,
        prediction_horizon=12,
        anomaly_threshold=70.0,  # Higher threshold due to model scaling issues
        save_predictions=True,
        advance_by=1  # Advance by 1 frame for maximum overlap
    )
    
    # Start streaming
    streamer.start_streaming(event_idx=0)
    
    try:
        # Process stream with more predictions
        predictor.process_stream(streamer, max_predictions=20)
        
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    
    finally:
        # Clean up
        streamer.stop_streaming()
        print("🏁 Real-time prediction system stopped")


if __name__ == "__main__":
    main()
