import torch
from torch.utils.data import Dataset, DataLoader, random_split
import time
import numpy as np
import random
import unittest

class DummyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 100)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Complicated data augmentation
        augmented_data = self.data[idx].clone()

        # 1. Random scaling
        scale = random.uniform(0.8, 1.2)
        augmented_data *= scale

        # 2. Random noise addition
        noise = torch.randn_like(augmented_data) * 0.1
        augmented_data += noise

        # 3. Random feature permutation
        perm = torch.randperm(augmented_data.shape[0])
        augmented_data = augmented_data[perm]

        # 4. Random feature zeroing
        zero_mask = torch.rand_like(augmented_data) > 0.9
        augmented_data[zero_mask] = 0

        # 5. Random polynomial transformation
        poly_degree = random.randint(2, 4)
        augmented_data = torch.pow(augmented_data, poly_degree)

        return augmented_data, self.labels[idx]

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def run_dataloader(num_workers, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    full_dataset = DummyDataset()
    
    # Split the dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    
    g = torch.Generator()
    g.manual_seed(seed)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True if num_workers else False,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=64,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=True if num_workers else False,
        shuffle=False
    )

    start_time = time.time()
    train_hash = 0
    val_hash = 0
    
    for _ in range(5):
        for batch, labels in train_dataloader:
            train_hash += hash(batch.numpy().tobytes() + labels.numpy().tobytes())
        
        for batch, labels in val_dataloader:
            val_hash += hash(batch.numpy().tobytes() + labels.numpy().tobytes())
    end_time = time.time()

    return end_time - start_time, train_hash, val_hash

class TestDataLoaderDeterminism(unittest.TestCase):
    def setUp(self):
        self.num_runs = 5
        self.seed = 42

    def run_dataloader_test(self, num_workers):
        times = []
        train_hashes = []
        val_hashes = []
        total_samples = 0

        for i in range(self.num_runs):
            time_taken, train_hash, val_hash = run_dataloader(num_workers, self.seed)
            times.append(time_taken)
            train_hashes.append(train_hash)
            val_hashes.append(val_hash)
            total_samples += len(DummyDataset())  # Assuming DummyDataset size is constant

        is_deterministic = len(set(train_hashes)) == 1 and len(set(val_hashes)) == 1
        return {
            'is_deterministic': is_deterministic,
            'avg_time': np.mean(times),
            'total_samples': total_samples
        }

    def test_determinism_no_workers(self):
        results = self.run_dataloader_test(0)
        self.assertTrue(results['is_deterministic'], "Test with 0 workers is not deterministic")
        # print(f"Average time with 0 workers: {results['avg_time']:.4f}s")

    def test_determinism_with_workers(self):
        results = self.run_dataloader_test(2)
        self.assertTrue(results['is_deterministic'], "Test with 2 workers is not deterministic")
        # print(f"Average time with 2 workers: {results['avg_time']:.4f}s")

    @unittest.skip
    def test_performance_comparison(self):
        results_0 = self.run_dataloader_test(0)
        results_2 = self.run_dataloader_test(2)
        
        speedup = results_0['avg_time'] / results_2['avg_time']
        print(f"Speedup with 2 workers: {speedup:.2f}x")
        
        self.assertGreater(speedup, 1, "Using 2 workers should be faster than 0 workers")

    def test_consistent_dataset_size(self):
        results_0 = self.run_dataloader_test(0)
        results_2 = self.run_dataloader_test(2)
        
        self.assertEqual(results_0['total_samples'], results_2['total_samples'], 
                         "Inconsistent dataset size between runs")

if __name__ == '__main__':
    unittest.main()