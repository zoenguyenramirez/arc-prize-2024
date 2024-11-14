import multiprocessing as mp
import queue
import random
import math
import logging

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataset, Subset

from src.utils.helper import set_deterministic

# Replace the original line with this implementation
def get_gaussian_index(dataset_length, offset_percentage, sigma_percentage=10.0/400):
    """
    Generate an index using Gaussian distribution
    
    Args:
        dataset_length: Length of the dataset
        offset_percentage: Where to center the distribution (0.0 to 1.0)
        sigma_percentage: Standard deviation as percentage of dataset length
    
    Returns:
        Integer index within valid range
    """
    # Calculate mean position based on offset percentage
    mu = int(dataset_length * offset_percentage)
    
    # Calculate standard deviation based on dataset length
    sigma = dataset_length * sigma_percentage
    
    # Generate Gaussian random number
    idx = int(random.gauss(mu=mu, sigma=sigma))
    mod_idx = idx % dataset_length
    return mod_idx


def split_dataset_and_keeping_the_oder(dataset, split_sizes):
    """
    Randomly split a dataset while maintaining order within each split.
    
    Args:
        dataset: The dataset to split
        split_sizes: List of proportions for splits. Should sum to 1.0
                    Example: [0.8, 0.2] for 80% train, 20% validation
    
    Returns:
        List of Subset datasets
    """
    assert sum(split_sizes) == len(dataset)
    assert all(x > 0 for x in split_sizes)

    # Calculate actual sizes
    total_size = len(dataset)
    
    # Create random permutation of indices
    indices = np.random.permutation(total_size)
    
    # Split indices according to sizes
    start = 0
    subsets = []
    for size in split_sizes:
        subset_indices = sorted(indices[start:start + size])  # Sort to maintain order
        subsets.append(Subset(dataset, subset_indices))
        start += size
    
    return subsets

class CustomDataLoader:
    def __init__(self, dataset, batch_size, max_batches_in_memory, *, num_workers=1, collate_fn=None, sampling_strategy='none', sigma_percentage = 5/400.):
        assert len(dataset) > 0
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_samples_in_memory = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn or (lambda x: x)
        
        self.sample_queue = mp.Queue(maxsize=self.max_samples_in_memory)
        self.batch_queue = mp.Queue(maxsize=max_batches_in_memory)
        
        self.stop_event = mp.Event()
        self.worker_processes = []
        self.batch_process = None
        self.is_stopped = False
        self.sampling_strategy = sampling_strategy
        self.sigma_percentage = sigma_percentage

    def start(self, seed_offset):
        assert not self.is_stopped, "Cannot restart a stopped loader"

        # Clear the stop event
        self.stop_event.clear()

        if self.sampling_strategy == 'gaussian':
            offset_percentage = random.random()
            logging.info(f'Sampling strategy: gaussian, mu:{offset_percentage *100:.1f}%, sigma:{self.sigma_percentage}')
        else:
            offset_percentage = float('inf')
        
        # Start worker processes to load samples
        for worker_id in range(self.num_workers):
            p = mp.Process(target=self._sample_worker, args=(worker_id + seed_offset, self.sampling_strategy, self.sigma_percentage, offset_percentage))
            p.start()
            self.worker_processes.append(p)
        
        # Start batch preparation process
        self.batch_process = mp.Process(target=self._batch_worker)
        self.batch_process.start()

    def stop(self):
        self.stop_event.set()
        
        # Clear the queues
        def clear_queue(q):
            while not q.empty():
                try:
                    q.get_nowait()
                except (ValueError, queue.Empty, FileNotFoundError, ConnectionResetError, EOFError, BrokenPipeError):
                    break

        clear_queue(self.sample_queue)
        clear_queue(self.batch_queue)
        
        # Terminate processes
        for p in self.worker_processes + [self.batch_process]:
            if p:
                p.terminate()
                p.join(timeout=1)
        
        # Close the queues
        try:
            self.sample_queue.close()
            self.batch_queue.close()
        except Exception as e:
            print(f"Error closing queues: {e}")        

        # Clear the processes
        self.worker_processes = []
        self.batch_process = None
        self.is_stopped = True

    def _sample_worker(self, random_seed, sampling_strategy, sigma_percentage, offset_percentage):
        set_deterministic(random_seed)
        while not self.stop_event.is_set():
            assert len(self.dataset) > 0
            
            if sampling_strategy == 'none':
                idx = random.randint(0, len(self.dataset) - 1)
            elif sampling_strategy == 'gaussian':
                idx = get_gaussian_index(len(self.dataset), offset_percentage=offset_percentage, sigma_percentage=sigma_percentage)
            else:
                assert False
            
            sample = self.dataset[idx]
            while not self.stop_event.is_set():
                try:
                    self.sample_queue.put(sample, timeout=1)
                    break
                except queue.Full:
                    continue

    def _batch_worker(self):
        set_deterministic(0)
        batch = []
        while not self.stop_event.is_set():
            try:
                sample = self.sample_queue.get(timeout=1)
                batch.append(sample)
                
                if len(batch) == self.batch_size:
                    collated_batch = self.collate_fn(batch)
                    while not self.stop_event.is_set():
                        try:
                            self.batch_queue.put(collated_batch, timeout=3)
                            batch = []
                            break
                        except queue.Full:
                            continue
                    batch = []
            except queue.Empty:
                continue

    def __iter__(self):
        return self
    
    def __len__(self):
        raise NotImplementedError("CustomDataLoader does not support len() as it's an infinite loader.")

    def __next__(self):
        if self.stop_event.is_set():
            raise StopIteration
        
        try:
            batch = self.batch_queue.get(timeout=1)
            return batch
        except queue.Empty:
            raise StopIteration  

# Usage example:
# custom_loader = CustomDataLoader(dataset, batch_size=32, max_batches_in_memory=10, num_workers=4, collate_fn=dataset.pad_collate)
# custom_loader.start()
# 
# try:
#     for batch in custom_loader:
#         # Process batch
#         ...
# finally:
#     custom_loader.stop()

class CacheAllDataLoader:
    """
    A data loader that pre-loads and pre-collates all data from a dataset.
    Useful for validation sets where data doesn't change and we want fast access.
    """
    def __init__(self, dataset, batch_size, *, collate_fn=None):
        self.batch_size = batch_size
        self.collate_fn = collate_fn or (lambda x: x)
        
        # Pre-load and pre-collate all data
        all_samples = [dataset[i] for i in range(len(dataset))]
        
        # Create batches
        self.cached_batches = []
        for i in range(0, len(all_samples), batch_size):
            batch = all_samples[i:i + batch_size]
            collated_batch = self.collate_fn(batch)
            self.cached_batches.append(collated_batch)
        
        logging.info(f'CacheAllDataLoader: cached {len(self.cached_batches)} batches, of {len(dataset)} samples')

    def __iter__(self):
        return iter(self.cached_batches)
    
    def __len__(self):
        return len(self.cached_batches)