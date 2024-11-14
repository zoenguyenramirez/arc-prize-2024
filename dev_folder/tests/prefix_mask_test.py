import unittest
import torch
from enum import Enum

from src.token import SpecialToken

class TestPrefixLength(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.seq_length = 10

    def calculate_prefix_length(self, input_ids):
        # Create a mask for sequences that don't contain START_OUTPUT
        contains_start_output = (input_ids == SpecialToken.START_OUTPUT.value).any(dim=1)
        
        # Calculate prefix_length using the current method
        prefix_length = input_ids.size(1) - 1 - (input_ids == SpecialToken.START_OUTPUT.value).long().flip(dims=[1]).argmax(dim=1)
        
        # Adjust prefix_length for sequences without START_OUTPUT
        sequence_length = input_ids.size(1)
        prefix_length = torch.where(
            contains_start_output,
            prefix_length,
            sequence_length + 1
        )
        
        return prefix_length

    def test_prefix_length_with_start_output(self):
        # Create input_ids with START_OUTPUT token
        input_ids = torch.randint(0, 10, (self.batch_size, self.seq_length))
        input_ids[0, 5] = SpecialToken.START_OUTPUT.value
        input_ids[1, 7] = SpecialToken.START_OUTPUT.value

        prefix_length = self.calculate_prefix_length(input_ids)

        self.assertEqual(prefix_length[0].item(), 5)
        self.assertEqual(prefix_length[1].item(), 7)

    def test_prefix_length_without_start_output(self):
        # Create input_ids without START_OUTPUT token
        input_ids = torch.randint(0, 10, (self.batch_size, self.seq_length))

        prefix_length = self.calculate_prefix_length(input_ids)

        self.assertEqual(prefix_length[0].item(), self.seq_length + 1)
        self.assertEqual(prefix_length[1].item(), self.seq_length + 1)

    def test_prefix_length_with_mixed_start_output(self):
        # Create input_ids without START_OUTPUT token
        input_ids = torch.randint(0, 10, (self.batch_size, self.seq_length))
        input_ids[1, 7] = SpecialToken.START_OUTPUT.value

        prefix_length = self.calculate_prefix_length(input_ids)

        self.assertEqual(prefix_length[0].item(), self.seq_length + 1)
        self.assertEqual(prefix_length[1].item(), 7)
        self.assertEqual(prefix_length[2].item(), self.seq_length + 1)

if __name__ == '__main__':
    unittest.main()
