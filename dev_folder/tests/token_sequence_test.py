# test_long_sequences.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import random

from src.load_data import GridDataset
from src.model import Transformer
from src.train import prefix_mask
from src.token import SpecialToken, VOCAB_SIZE


class TestLongSequences(unittest.TestCase):
    def setUp(self):
        self.dataset = GridDataset.load_from_paired_file({}, {})

    # Add this method to the GridDataset class in load_data.py

    def generate_long_sequence(self, length):
        sequence = [SpecialToken.START.value, SpecialToken.START_INPUT.value]
        for _ in range(length//2):
            sequence.append(random.randint(0, 9))
        sequence.append(SpecialToken.START_OUTPUT.value)
        while len(sequence) < length:
            sequence.append(random.randint(0, 9))
        sequence.append(SpecialToken.END.value)
        return sequence

    def test_very_long_sequence(self):
        long_sequence = self.generate_long_sequence(4)
        self.assertGreater(len(long_sequence), 3)
        self.assertEqual(long_sequence[0], SpecialToken.START.value)
        self.assertEqual(long_sequence[-1], SpecialToken.END.value)

    def test_prefix_mask_with_long_sequence(self):
        device = torch.device("cpu")
        long_sequence = self.generate_long_sequence(4)
        batch = torch.tensor([long_sequence], dtype=torch.long).to(device)
        
        input_ids = batch[:, :-1]
        target = batch[:, 1:]
        # print('batch', batch)
        # print('input_ids', input_ids)
        # print('target', target)
        
        outputs = torch.rand(1, input_ids.size(1), VOCAB_SIZE).to(device)
        # print('outputs', outputs)
        
        start_output_index = (input_ids == SpecialToken.START_OUTPUT.value).nonzero(as_tuple=True)[1][0]
        masked_target = prefix_mask(target, len(long_sequence) - 1, [int(start_output_index.item())])

        # print('masked_outputs', masked_outputs)
        # print('masked_target', masked_target)
        
        # Check if the prefix is masked correctly
        # print('start_output_index', start_output_index)
        # print('masked_target[:, :start_output_index]', masked_target[:, :start_output_index])
        self.assertTrue(torch.all(masked_target[:, :start_output_index - 1] == SpecialToken.PAD.value))
        
        # Check if the output after START_OUTPUT is not masked
        self.assertFalse(torch.all(masked_target[:, start_output_index:] == SpecialToken.PAD.value))

    def test_prefix_mask_with_long_sequence_no_OUTPUT_TOKEN(self):
        device = torch.device("cpu")
        long_sequence = self.generate_long_sequence(18)[:7]
        batch = torch.tensor([long_sequence], dtype=torch.long).to(device)
        
        input_ids = batch[:, :-1]
        target = batch[:, 1:]
        # print('batch', batch)
        # print('input_ids', input_ids)
        # print('target', target)
        
        outputs = torch.rand(1, input_ids.size(1), VOCAB_SIZE).to(device)
        # print('outputs', outputs)
        
        masked_target = prefix_mask(target, len(long_sequence) - 1, [99999])

        # print('masked_outputs', masked_outputs)
        # print('masked_target', masked_target)
        
        # print('masked_target[:, :start_output_index]', masked_target[:, :start_output_index])
        self.assertTrue(torch.all(masked_target[:, :] == SpecialToken.PAD.value))
                
    def test_very_long_sequence_with_transformer(self):
        edge_length = 100
        # Create a mock challenges and solutions dictionary
        mock_challenges = {
            'test_challenge': {
                'train': [
                    {'input': [[1] * edge_length] * edge_length, 'output': [[2] * edge_length] * edge_length}
                ] * 10,  # 10 training examples
                'test': [{'input': [[3] * edge_length] * edge_length}]
            }
        }
        mock_solutions = {'test_challenge': [[[4] * edge_length] * edge_length]}

        # Create GridDataset instance
        dataset = GridDataset.load_from_paired_file(mock_challenges, mock_solutions)
        dataset.set_max_length(800)

        # Check the length of the first (and only) item in the dataset
        long_sequence = dataset[0]
        # print(f"Length of the generated sequence: {len(long_sequence)}")

        # Create a small batch
        batch = [long_sequence]
        padded_batch = dataset.pad_collate(batch)

        # Print the shape of the padded batch
        # print(f"Shape of padded batch: {padded_batch.shape}")

        # Create a Transformer model with a smaller max_length
        max_length = 800
        model = Transformer(vocab_size=SpecialToken.COUNT_OF_TOKENS.value, 
                            embed_size=40, 
                            num_layers=2, 
                            heads=4, 
                            max_length=max_length,
                            use_grid_encoder=False)

        # Try to feed the padded batch to the Transformer
        try:
            mask = torch.triu(torch.ones((1, max_length, max_length)), diagonal=1).bool()
            if random.randint(0, 1) == 0:
                output = model(padded_batch['data'][:, :max_length], mask)
            else:
                output = model(padded_batch['data'], mask)
            # print(f"Shape of Transformer output: {output.shape}")
        except RuntimeError as e:
            self.fail(f"RuntimeError occurred: {str(e)}")

        # Assert that the output shape matches the expected shape
        self.assertEqual(output.shape, (1, max_length, SpecialToken.COUNT_OF_TOKENS.value))

    def test_pad_collate_last_column_setting(self):
        # Create a mock dataset
        dataset = GridDataset.load_from_paired_file({}, {})
        dataset.set_max_length(20)  # Set a small max length for testing

        # Create a sample sequence
        sample_sequence = [
            [SpecialToken.START.value, 0, 0, 0, 0],
            [SpecialToken.START_INPUT.value, 0, 0, 0, 1],
            [1, 0, 0, 1, 1],
            [2, 0, 1, 2, 3],
            [1, 0, 2, 1, 1],
            [2, 1, 0, 2, 1],
            [1, 1, 1, 1, 1],
            [2, 1, 2, 2, 1],
            [SpecialToken.START_OUTPUT.value, 0, 0, 0, 2],
            [3, 0, 0, 3, 2],
            [4, 0, 1, 4, 2],
            [SpecialToken.END.value, 0, 0, 0, -1]
        ]

        # Create a batch with two sequences
        batch = [{'task': sample_sequence, 'idx': 0, 'end_of_examples_index': 8}, 
                 {'task': sample_sequence[:8], 'idx': 1, 'end_of_examples_index': -1}]  # Second sequence is shorter

        # Apply pad_collate
        padded_batch = dataset.pad_collate(batch)
        padded_batch = padded_batch['data']

        # Check the shape of the padded batch
        self.assertEqual(padded_batch.shape, (2, 12, 5))

        # Check if the last column after the last START_OUTPUT is set to -1
        seq = padded_batch[0]
        last_start_output = (seq[:, 0] == SpecialToken.START_OUTPUT.value).nonzero(as_tuple=True)[0][-1]
        self.assertTrue(torch.all(seq[last_start_output:, -1] == -1))

        # Check if the values before the last START_OUTPUT are preserved
        self.assertEqual(padded_batch[0, 2, -1].item(), 1)
        self.assertEqual(padded_batch[0, 3, -1].item(), 3)

        # Check if padding is applied correctly
        self.assertTrue(torch.all(padded_batch[1, -4:, 0] == SpecialToken.PAD.value))
        self.assertTrue(torch.all(padded_batch[1, -4:, 1:] == -1))        

if __name__ == '__main__':
    unittest.main();  exit(0) 

    suite = unittest.TestSuite()
    suite.addTest(TestLongSequences('test_prefix_mask_with_long_sequence_no_OUTPUT_TOKEN'))
    unittest.TextTestRunner().run(suite)
