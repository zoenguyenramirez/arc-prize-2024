import unittest
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from itertools import cycle
from src.utils.iterable_helper import custom_cycle  # Replace the import of cycle

class FakeDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor([[i for i in range(3)]])
        self.access_count = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx not in self.access_count:
            self.access_count[idx] = 0
        self.access_count[idx] += 1
        
        # Return the actual data item
        return self.data[idx]

class TestFakeDataset(unittest.TestCase):
    def setUp(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        self.dataset = FakeDataset()

    def test_getitem_multiple_access(self):
        # Test accessing the same index multiple times
        for _ in range(5):
            item = self.dataset[0]
            self.assertIsNotNone(item)
        
        self.assertEqual(self.dataset.access_count[0], 5)

    def test_dataloader_epoch(self):
        # Test accessing items through a DataLoader for multiple epochs
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        
        for epoch in range(3):
            for batch in dataloader:
                self.assertIsNotNone(batch)
            
        # Check that each item was accessed once per epoch
        for idx in range(len(self.dataset)):
            self.assertEqual(self.dataset.access_count[idx], 3)

    def test_dataloader_with_workers(self):
        # Test accessing items through a DataLoader with multiple workers
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=2)
        
        for _ in range(3):
            for batch in dataloader:
                self.assertIsNotNone(batch)
        
        # print('self.dataset.access_count', self.dataset.access_count)
        # Check that each item was accessed three times (once per epoch)
        self.assertEqual(self.dataset.access_count, {})

    def test_dataloader_with_0_workers(self):
        # Test accessing items through a DataLoader with multiple workers
        dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        
        for _ in range(3):
            for batch in dataloader:
                self.assertIsNotNone(batch)
        
        # Check that each item was accessed three times (once per epoch)
        for idx in range(len(self.dataset)):
            self.assertEqual(self.dataset.access_count[idx], 3)

    def test_infinite_dataloader(self):
        # Create a small dataset
        dataset = FakeDataset()
        
        # Create a DataLoader with batch_size=1
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Create an infinite dataloader
        infinite_dataloader = cycle(dataloader)
        
        # Number of iterations to test (more than the dataset size)
        num_iterations = 9
        
        for _ in range(num_iterations):
            batch = next(infinite_dataloader)
            self.assertIsNotNone(batch)
            self.assertEqual(batch.shape, (1, 3))  # Assuming FakeDataset returns tensors of shape (1,)
        
        # Check that each item was accessed multiple times
        total_accesses = sum(dataset.access_count.values())
        self.assertEqual(total_accesses, 1) # cycle sucks, "returning elements from the iterable and saving a copy of each"
        
        # Check that the access count for each item is roughly equal
        expected_accesses_per_item = num_iterations // len(dataset)
        for count in dataset.access_count.values():
            self.assertNotEqual(count, expected_accesses_per_item) # cycle sucks, "returning elements from the iterable and saving a copy of each"

    def test_custom_infinite_dataloader(self):
        # Create a small dataset
        dataset = FakeDataset()
        
        # Create a DataLoader with batch_size=1
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Create an infinite dataloader using custom_cycle
        infinite_dataloader = custom_cycle(dataloader)
        
        # Number of iterations to test (more than the dataset size)
        num_iterations = 9
        
        for _ in range(num_iterations):
            batch = next(infinite_dataloader)
            self.assertIsNotNone(batch)
            self.assertEqual(batch.shape, (1, 3))  # Assuming FakeDataset returns tensors of shape (1, 3)
        
        # Check that each item was accessed multiple times
        total_accesses = sum(dataset.access_count.values())
        self.assertEqual(total_accesses, num_iterations)
        
        # Check that the access count for each item is roughly equal
        expected_accesses_per_item = num_iterations // len(dataset)
        for count in dataset.access_count.values():
            self.assertAlmostEqual(count, expected_accesses_per_item, delta=1)
            
if __name__ == '__main__':
    unittest.main(); exit(0) 

    suite = unittest.TestSuite()
    suite.addTest(TestFakeDataset('test_dataloader_with_workers'))
    unittest.TextTestRunner().run(suite)
