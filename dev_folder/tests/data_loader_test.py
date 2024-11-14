import unittest
import time
import threading
import random
from time import sleep
import numpy as np

import torch
from torch.utils.data import Dataset

from typing import Dict, List, Tuple, Any, Set, List

from src.utils.data_loader import CustomDataLoader, get_gaussian_index, split_dataset_and_keeping_the_oder
from src.utils.grid_data_process import end_of_examples_mark, SpecialToken

class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [torch.tensor([i]) for i in range(size)]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

def dummy_collate_fn(batch):
    return torch.stack(batch)

class TestCustomDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset = DummyDataset(100)
        self.batch_size = 10
        self.max_batches_in_memory = 20
        self.num_workers = 2

    def test_initialization(self):
        loader = CustomDataLoader(self.dataset, self.batch_size, self.max_batches_in_memory, num_workers=self.num_workers)
        self.assertEqual(loader.batch_size, self.batch_size)
        self.assertEqual(loader.max_samples_in_memory, self.batch_size)
        self.assertEqual(loader.num_workers, self.num_workers)

    def test_iteration(self):
        loader = CustomDataLoader(self.dataset, self.batch_size, self.max_batches_in_memory, num_workers=self.num_workers, collate_fn=dummy_collate_fn)
        loader.start(1)

        try:
            for batch_count in range (5):
                while True:
                    try:            
                        batch = next(loader)
                        break
                    except StopIteration:
                        print('Data producer is too slow!')
                        continue

                self.assertEqual(batch.shape[0], self.batch_size)
        finally:
            loader.stop()

    def test_prevent_list_conversion(self):
        loader = CustomDataLoader(self.dataset, self.batch_size, self.max_batches_in_memory, num_workers=self.num_workers, collate_fn=dummy_collate_fn)
        loader.start(1)

        try:
            with self.assertRaises(NotImplementedError):
                list(loader)
        finally:
            loader.stop()

    def test_shuffle(self):
        loader1 = CustomDataLoader(self.dataset, self.batch_size, self.max_batches_in_memory, num_workers=self.num_workers, collate_fn=dummy_collate_fn)
        loader2 = CustomDataLoader(self.dataset, self.batch_size, self.max_batches_in_memory, num_workers=self.num_workers, collate_fn=dummy_collate_fn)
        
        loader1.start(1)
        loader2.start(1)

        sleep(1)

        try:
            batches1 = [next(loader1) for _ in range(5)]
            batches2 = [next(loader2) for _ in range(5)]

            # Check if at least one batch is different
            are_different = any(not torch.equal(b1, b2) for b1, b2 in zip(batches1, batches2))
            self.assertTrue(are_different, "The batches from the two loaders should be different due to shuffling")
        finally:
            loader1.stop()
            loader2.stop()

    def test_stop_and_restart(self):
        loader = CustomDataLoader(self.dataset, self.batch_size, self.max_batches_in_memory, num_workers=self.num_workers, collate_fn=dummy_collate_fn)
        loader.start(1)

        sleep(1)

        try:
            next(loader)  # Get one batch
            loader.stop()

            # Ensure the loader has stopped
            with self.assertRaises(StopIteration):
                next(loader)

            # Try to restart the loader and expect an AssertionError
            with self.assertRaises(AssertionError):
                loader.start(1)

        finally:
            loader.stop()

    def test_empty_dataset(self):
        empty_dataset = DummyDataset(0)
        loader = None

        with self.assertRaises(AssertionError):
            try:
                loader = CustomDataLoader(empty_dataset, self.batch_size, self.max_batches_in_memory, num_workers=self.num_workers, collate_fn=dummy_collate_fn)
                loader.start(1)
                next(loader)
            finally:
                if loader is not None:
                    loader.stop()

    def test_worker_stopping(self):
        loader = CustomDataLoader(self.dataset, self.batch_size, self.max_batches_in_memory, num_workers=self.num_workers, collate_fn=dummy_collate_fn)
        loader.start(1)

        # Get a few batches
        for _ in range(3):
            next(loader)

        # Store the batch_process reference before stopping
        batch_process = loader.batch_process

        # Stop the loader
        loader.stop()

        # Check if all processes have stopped
        time.sleep(0.1)  # Give processes a moment to stop
        self.assertTrue(all(not p.is_alive() for p in loader.worker_processes))
        self.assertIsNone(loader.batch_process)  # Check if batch_process is set to None
        self.assertFalse(batch_process.is_alive())  # Check if the original batch_process has stopped

    def test_concurrent_production_consumption(self):
        loader = CustomDataLoader(self.dataset, self.batch_size, self.max_batches_in_memory, num_workers=self.num_workers, collate_fn=dummy_collate_fn)
        loader.start(1)

        try:
            batches = []
            def consumer():
                for _ in range(500):  # Consume 50 batches
                    batches.append(next(loader))

            consumer_thread = threading.Thread(target=consumer)
            consumer_thread.start()
            consumer_thread.join(timeout=10)  # Wait for up to 10 seconds

            self.assertEqual(len(batches), 500, "Expected 50 batches to be consumed")
            self.assertTrue(all(batch.shape[0] == self.batch_size for batch in batches))
        finally:
            loader.stop()

    def test_slow_consumer(self):
        loader = CustomDataLoader(self.dataset, self.batch_size, self.max_batches_in_memory, num_workers=self.num_workers, collate_fn=dummy_collate_fn)
        loader.start(1)

        try:
            for _ in range(self.max_batches_in_memory * 2):  # Try to get more batches than max_batches_in_memory
                time.sleep(0.1)  # Simulate slow consumption
                batch = next(loader)
                self.assertEqual(batch.shape[0], self.batch_size)
        finally:
            loader.stop()

    def test_slow_producer(self):
        class SlowDataset(DummyDataset):
            def __getitem__(self, idx):
                time.sleep(0.01)  # Simulate slow data loading
                return super().__getitem__(idx)

        slow_dataset = SlowDataset(100)
        loader = CustomDataLoader(slow_dataset, self.batch_size, self.max_batches_in_memory, num_workers=20, collate_fn=dummy_collate_fn)
        loader.start(1)

        try:
            start_time = time.time()
            batches = [next(loader) for _ in range(50)]
            end_time = time.time()

            self.assertEqual(len(batches), 50)
            self.assertTrue(all(batch.shape[0] == self.batch_size for batch in batches))
            
            thearatical_speed = 50 * self.batch_size / 20 * 0.01
            
            # Check if the time taken is less than what it would take to load sequentially
            # (which would be at least 5 * 10 * 0.1 = 5 seconds)
            self.assertLess(end_time - start_time, thearatical_speed + 2, "Parallel loading should be faster than sequential loading")
        finally:
            loader.stop()

    def test_very_slow_producer(self):
        class VerySlowDataset(DummyDataset):
            def __getitem__(self, idx):
                # print('getting', idx)
                time.sleep(random.uniform(0, 3))
                # print('got', idx)
                return super().__getitem__(idx)

        very_slow_dataset = VerySlowDataset(100)
        loader = CustomDataLoader(very_slow_dataset, self.batch_size, self.max_batches_in_memory, num_workers=5, collate_fn=dummy_collate_fn)
        loader.start(1)

        try:
            batches = []
            for _ in range(9):  # Try to get batches 10 times
                try:
                    batch = next(loader)
                    batches.append(batch)
                except StopIteration:
                    continue

            # Check if we got any batches
            self.assertTrue(len(batches) > 0, "Should have received at least one batch")
            
            self.assertTrue(len(batches) < 5, "Should not have received all batches")

            # Check if all received batches have the correct size
            self.assertTrue(all(batch.shape[0] == self.batch_size for batch in batches))

            # Check if the loader eventually raises StopIteration
            with self.assertRaises(StopIteration):
                while True:
                    next(loader)

        finally:
            loader.stop()

    @staticmethod
    def expand_to_compact_grid_format(lst: List[int]):
        return [[elem, 0, 0] for elem in lst]
    
    def test_end_of_encoder_section_mark(self):
        # Test case 1: Normal case with multiple START_OUTPUT tokens
        task1 = [1, 2, SpecialToken.START_OUTPUT.value, 3, 4, SpecialToken.START_OUTPUT.value, 5, 6]
        self.assertEqual(end_of_examples_mark(self.expand_to_compact_grid_format(task1)), 5)

        # Test case 2: Only one START_OUTPUT token at the end
        task2 = [1, 2, 3, 4, SpecialToken.START_OUTPUT.value]
        self.assertEqual(end_of_examples_mark(self.expand_to_compact_grid_format(task2)), 4)

        # Test case 3: No START_OUTPUT token
        task3 = [1, 2, 3, 4, 5]
        self.assertEqual(end_of_examples_mark(self.expand_to_compact_grid_format(task3)), -1)

        # Test case 4: Empty list
        task4 = []
        self.assertEqual(end_of_examples_mark(self.expand_to_compact_grid_format(task4)), -1)

        # Test case 5: START_OUTPUT token at the beginning
        task5 = [SpecialToken.START_OUTPUT.value, 1, 2, 3, 4]
        self.assertEqual(end_of_examples_mark(self.expand_to_compact_grid_format(task5)), 0)            

class TestGaussianIndex(unittest.TestCase):
    def test_get_gaussian_index_basic(self):
        dataset_length = 100
        # Test multiple times to account for randomness
        for _ in range(1000):
            idx = get_gaussian_index(dataset_length, offset_percentage=0.5)
            # Check if index is within valid range
            self.assertTrue(0 <= idx < dataset_length)

    def test_get_gaussian_index_offset(self):
        dataset_length = 100
        samples = []
        # Collect samples to verify distribution
        for _ in range(1000):
            idx = get_gaussian_index(dataset_length, offset_percentage=0.75)
            samples.append(idx)
        
        # Mean should be close to 75 (dataset_length * 0.75)
        mean = sum(samples) / len(samples)
        self.assertTrue(60 < mean < 90)  # Allow some variance due to randomness

    def test_get_gaussian_index_sigma(self):
        dataset_length = 100
        # Test with very small sigma
        samples_tight = []
        for _ in range(1000):
            idx = get_gaussian_index(dataset_length, offset_percentage=0.5, sigma_percentage=0.1)
            samples_tight.append(idx)
        
        # Test with larger sigma
        samples_wide = []
        for _ in range(1000):
            idx = get_gaussian_index(dataset_length, offset_percentage=0.5, sigma_percentage=0.3)
            samples_wide.append(idx)
        
        # Calculate standard deviations
        std_tight = np.std(samples_tight)
        std_wide = np.std(samples_wide)
        
        # The wider distribution should have larger standard deviation
        self.assertTrue(std_wide > std_tight)

    def test_get_gaussian_index_edge_cases(self):
        for _ in range(10):
            # Test with small dataset
            small_length = 2
            idx = get_gaussian_index(small_length, offset_percentage=0.5)
            self.assertTrue(0 <= idx < small_length)
            
            # Test with extreme offset
            idx = get_gaussian_index(100, offset_percentage=0.0)
            self.assertTrue(0 <= idx < 100)
            
            idx = get_gaussian_index(100, offset_percentage=1.0)
            self.assertTrue(0 <= idx < 100)
            
            # Test with extreme sigma
            idx = get_gaussian_index(100, offset_percentage=0.5, sigma_percentage=0.01)
            self.assertTrue(0 <= idx < 100)
            
            idx = get_gaussian_index(100, offset_percentage=0.5, sigma_percentage=1.0)
            self.assertTrue(0 <= idx < 100)

    def test_get_gaussian_index_edge_case(self):
        for _ in range(10):            
            idx = get_gaussian_index(10000, offset_percentage=0.0, sigma_percentage=0.01)
            self.assertTrue(0 <= idx < 10000)

            self.assertTrue(9000 <= idx or idx <= 1000)
            
class TestSplitDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = DummyDataset(100)  # Create a dataset with 100 elements

    def test_basic_split(self):
        # Test basic 80-20 split
        splits = split_dataset_and_keeping_the_oder(self.dataset, [80, 20])
        
        # Check if we got the correct number of splits
        self.assertEqual(len(splits), 2)
        
        # Check if the splits have the correct sizes
        self.assertEqual(len(splits[0]), 80)
        self.assertEqual(len(splits[1]), 20)

    def test_order_preservation(self):
        splits = split_dataset_and_keeping_the_oder(self.dataset, [60, 40])
        
        # Check if elements within each split are in order
        for split in splits:
            values = [item.item() for item in split]
            self.assertEqual(values, sorted(values))

    def test_multiple_splits(self):
        # Test splitting into three parts
        splits = split_dataset_and_keeping_the_oder(self.dataset, [50, 30, 20])
        
        self.assertEqual(len(splits), 3)
        self.assertEqual(len(splits[0]), 50)
        self.assertEqual(len(splits[1]), 30)
        self.assertEqual(len(splits[2]), 20)

    def test_invalid_splits(self):
        # Test when split sizes don't sum to dataset length
        with self.assertRaises(AssertionError):
            split_dataset_and_keeping_the_oder(self.dataset, [60, 20])  # Sums to 80, not 100

        # Test negative split sizes
        with self.assertRaises(AssertionError):
            split_dataset_and_keeping_the_oder(self.dataset, [110, -10])

    def test_empty_splits(self):
        # Test with zero-sized splits
        with self.assertRaises(AssertionError):
            split_dataset_and_keeping_the_oder(self.dataset, [100, 0])

    def test_disjoint_sets(self):
        splits = split_dataset_and_keeping_the_oder(self.dataset, [50, 50])
        
        # Convert splits to sets of indices
        set1 = set([item.item() for item in splits[0]])
        set2 = set([item.item() for item in splits[1]])
        
        # Check that there's no overlap between splits
        self.assertEqual(len(set1.intersection(set2)), 0)
        
        # Check that union of splits covers all indices
        self.assertEqual(len(set1.union(set2)), len(self.dataset))

    def test_reproducibility(self):
        # Test if splits are reproducible with same random seed
        np.random.seed(42)
        splits1 = split_dataset_and_keeping_the_oder(self.dataset, [60, 40])
        
        np.random.seed(42)
        splits2 = split_dataset_and_keeping_the_oder(self.dataset, [60, 40])
        
        # Check if splits are identical
        for split1, split2 in zip(splits1, splits2):
            values1 = [item.item() for item in split1]
            values2 = [item.item() for item in split2]
            self.assertEqual(values1, values2)
    
if __name__ == '__main__':
    unittest.main(); exit(0)

    suite = unittest.TestSuite()
    suite.addTest(TestGaussianIndex('test_get_gaussian_index_edge_case'))

    # loader = unittest.TestLoader()
    # suite = loader.loadTestsFromTestCase(TestSplitDataset)

    unittest.TextTestRunner().run(suite)
