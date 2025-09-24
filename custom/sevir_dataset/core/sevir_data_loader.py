#!/usr/bin/env python3
'''
Efficient SEVIR Data Loader for M1 MacBook Pro
Loads data in batches to avoid memory issues
'''

import h5py
import numpy as np
import os

class SEVIRDataLoader:
    def __init__(self, filepath, batch_size=10):
        self.filepath = filepath
        self.batch_size = batch_size
        self._file = None
        self._dataset = None
        self._num_events = 0
        
    def __enter__(self):
        self._file = h5py.File(self.filepath, 'r')
        if 'vil' in self._file:
            self._dataset = self._file['vil']
            self._num_events = self._dataset.shape[0]
            print(f"Opened SEVIR file with {self._num_events:,} events")
        else:
            print(f"Available datasets: {list(self._file.keys())}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
    
    def get_batch(self, start_idx=0, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        end_idx = min(start_idx + batch_size, self._num_events)
        
        if self._dataset is not None:
            batch = self._dataset[start_idx:end_idx]
            print(f"Loaded batch {start_idx}:{end_idx} - Shape: {batch.shape}")
            return batch
        return None
    
    def get_sample(self, num_events=5):
        return self.get_batch(0, num_events)
        
    @property
    def num_events(self):
        return self._num_events

# Usage example:
# with SEVIRDataLoader('data/sevir/vil/SEVIR_VIL_STORMEVENTS_2019_0101_0630.h5') as loader:
#     sample = loader.get_sample(5)  # Load 5 events
#     print(f"Sample shape: {sample.shape}")
