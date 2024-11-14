import unittest
import torch
import torch.nn as nn
from src.model import Transformer
from src.token import VOCAB_SIZE
from src.utils.transformer_helper import mask_expansion
from src.utils.transformer_helper import create_mask

class TestTransformer(unittest.TestCase):
    def setUp(self):
        # torch.manual_seed(42)
        self.vocab_size = VOCAB_SIZE
        self.embed_size = 48
        self.num_layers = 2
        self.heads = 2
        self.max_length = 100
        self.batch_size = 4
        self.seq_length = 50
        self.cell_size = 5

        self.transformer = Transformer(
            vocab_size=self.vocab_size,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            heads=self.heads,
            max_length=self.max_length
        )

    def test_transformer_initialization(self):
        self.assertIsInstance(self.transformer, nn.Module)
        self.assertEqual(self.transformer.embed_size, self.embed_size)
        self.assertEqual(len(self.transformer.layers), self.num_layers)
        self.assertEqual(self.transformer.fc_out.out_features, self.vocab_size)

    def test_transformer_forward(self):
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length, self.cell_size))
        mask = torch.ones(4, self.seq_length, self.seq_length).triu(1).bool()
        
        output = self.transformer(x, mask)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.vocab_size))

    def test_transformer_output_range(self):
        x = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length, self.cell_size))
        mask = torch.ones(4, self.seq_length, self.seq_length).triu(1).bool()
        
        output = self.transformer(x, mask)
        
        self.assertTrue(torch.isfinite(output).all())


    def test_grid_encoding(self):
        # Create a simple input where we can predict the effect of grid encoding
        x = torch.zeros((1, 4, 5), dtype=torch.long)
        x[0, 0] = torch.tensor([1, 0, 0, 0, 0])  # token 1, position (0, 0)
        x[0, 1] = torch.tensor([2, 1, 1, 0, 0])  # token 2, position (1, 1)
        x[0, 2] = torch.tensor([3, 2, 2, 0, 0])  # token 3, position (2, 2)
        x[0, 3] = torch.tensor([4, -1, -1, 0, 0])  # token 4, invalid position (-1, -1)

        mask = torch.ones(1, 4, 4).triu(1).bool()

        # Get the initial tensor after embedding and positional encoding
        with torch.no_grad():
            # Get the output from the transformer
            output = self.transformer(x, mask)

        # Verify that the output has the correct shape
        self.assertEqual(output.shape, (1, 4, self.vocab_size))


    def test_mask_expansion(self):
        batch_size = 2
        sequence_length = 10
        num_heads = 4

        # Create a sample mask
        mask = torch.ones((batch_size, sequence_length, sequence_length)).bool()
        mask[:, :, 0] = False  # Set the first column to False for testing

        # Apply mask_expansion
        expanded_mask = mask_expansion(mask, num_heads)

        # Check the shape of the expanded mask
        self.assertEqual(expanded_mask.shape, (batch_size * num_heads, sequence_length, sequence_length))

        # Check that the expansion preserved the mask values
        for i in range(batch_size * num_heads):
            self.assertTrue(torch.all(expanded_mask[i, :, 1:]))
            self.assertFalse(torch.any(expanded_mask[i, :, 0]))

    def test_mask_expansion2(self):
        batch_size = 2
        sequence_length = 10
        num_heads = 4

        # Create a sample mask
        mask = torch.ones((batch_size, sequence_length, sequence_length)).bool()
        mask[0, :, :] = False  # Set the first column to False for testing

        # Apply mask_expansion
        expanded_mask = mask_expansion(mask, num_heads)

        # Check the shape of the expanded mask
        self.assertEqual(expanded_mask.shape, (batch_size * num_heads, sequence_length, sequence_length))

        # Check that the mask is correctly expanded for each head
        for i in range(batch_size):
            for j in range(num_heads):
                idx = i * num_heads + j
                self.assertTrue(torch.all(expanded_mask[idx] == mask[i]))

    def test_mask_slicing_consistency(self):
        self.transformer.eval()  # Make sure there's no dropout during evaluation by setting the model to eval mode
        # Create a simple, deterministic input sequence
        seq_length = 10
        cell_size = 5
        single_sample = torch.zeros((1, seq_length, cell_size), dtype=torch.long)
        # Fill with some pattern
        for i in range(seq_length):
            single_sample[0, i] = torch.tensor([i % self.vocab_size, i % 3, i % 2, 0, 0])

        # Create inputs with different batch sizes by repeating the same data
        input_1 = single_sample
        input_2 = single_sample.repeat(2, 1, 1)
        input_4 = single_sample.repeat(4, 1, 1)

        # Get outputs for different batch sizes
        with torch.no_grad():
            mask_1 = create_mask(input_1, single_sample.device, [5])
            output_1 = self.transformer(input_1, mask_1)
            output_11 = self.transformer(input_1, mask_1)
            mask_2 = create_mask(input_2, single_sample.device, [5, 5])
            output_2 = self.transformer(input_2, mask_2)
            mask_4 = create_mask(input_4, single_sample.device, [5, 5, 5, 5])
            output_4 = self.transformer(input_4, mask_4)

        self.assertTrue(torch.equal(output_11, output_1))
        self.assertTrue(torch.allclose(output_1[0], output_2[0], atol=1e-5))
        self.assertTrue(torch.allclose(output_1[0], output_4[0], atol=1e-5))

        # Check that all samples within the same batch are identical
        for i in range(1, 2):
            self.assertTrue(torch.allclose(output_2[0], output_2[i], atol=1e-5))
        for i in range(1, 4):
            self.assertTrue(torch.allclose(output_4[0], output_4[i], atol=1e-5))

        # Test with progressive heads
        # Create a transformer with progressive heads
        transformer_prog = Transformer(
            vocab_size=self.vocab_size,
            embed_size=self.embed_size,
            num_layers=7,
            heads=4,  # More heads to test progressive reduction
            progressive_head=2,  # Every 2 layers, divide number of heads by 2
            max_length=self.max_length
        )

        transformer_prog.eval()

        with torch.no_grad():
            prog_output_1 = transformer_prog(input_1, mask_1)
            prog_output_2 = transformer_prog(input_2, mask_2)
            prog_output_4 = transformer_prog(input_4, mask_4)

        # Check that outputs are consistent across different batch sizes
        self.assertTrue(torch.allclose(prog_output_1[0], prog_output_2[0], atol=1e-5))
        self.assertTrue(torch.allclose(prog_output_1[0], prog_output_4[0], atol=1e-5))

        # Check that all samples within the same batch are identical
        for i in range(1, 2):
            self.assertTrue(torch.allclose(prog_output_2[0], prog_output_2[i], atol=1e-5))
        for i in range(1, 4):
            self.assertTrue(torch.allclose(prog_output_4[0], prog_output_4[i], atol=1e-5))

    def test_mask_slicing_consistency_of_different_samples(self):
        self.transformer.eval()  # Make sure there's no dropout during evaluation

        # Create different sequence lengths
        seq_lengths = [5, 8, 10]
        cell_size = 5
        
        # Create different samples with different patterns
        samples = []
        for seq_len in seq_lengths:
            sample = torch.zeros((1, seq_len, cell_size), dtype=torch.long)
            # Fill with different patterns for each sequence
            for i in range(seq_len):
                sample[0, i] = torch.tensor([(i + 1) % self.vocab_size, i % 3, i % 2, 0, 0])
            samples.append(sample)

        # Test individual samples
        outputs = []
        for sample in samples:
            with torch.no_grad():
                mask = create_mask(sample, sample.device, [sample.shape[1] // 2])
                output = self.transformer(sample, mask)
                outputs.append(output)

        # Combine samples into batches
        # Create batch of 2 with padding
        max_len = max(seq_lengths[:2])
        batch_2 = torch.zeros((2, max_len, cell_size), dtype=torch.long)
        batch_2[0, :seq_lengths[0]] = samples[0][0, :seq_lengths[0]]
        batch_2[1, :seq_lengths[1]] = samples[1][0, :seq_lengths[1]]

        # Create batch of 3 with padding
        max_len_3 = max(seq_lengths)
        batch_3 = torch.zeros((3, max_len_3, cell_size), dtype=torch.long)
        for i in range(3):
            batch_3[i, :seq_lengths[i]] = samples[i][0, :seq_lengths[i]]

        batch_31 = torch.zeros((3, max_len_3, cell_size), dtype=torch.long)
        for i in range(3):
            batch_31[i] = batch_3[2 - i]

        # Test batched samples
        with torch.no_grad():
            # Test batch of 2
            mask_2 = create_mask(batch_2, batch_2.device, [l // 2 for l in seq_lengths[:2]])
            output_batch_2 = self.transformer(batch_2, mask_2)
            
            batch3_end_of_example_indices = [l // 2 for l in seq_lengths]
            # Test batch of 3
            mask_3 = create_mask(batch_3, batch_3.device, batch3_end_of_example_indices)
            output_batch_3 = self.transformer(batch_3, mask_3)

            batch3_end_of_example_indices.reverse()
            mask_31 = create_mask(batch_31, batch_31.device, batch3_end_of_example_indices)
            output_batch_31 = self.transformer(batch_31, mask_31)            

            # Verify that individual samples match their batched counterparts
            # For batch_2
            self.assertTrue(torch.allclose(
                output_batch_2[0, :seq_lengths[0]], 
                outputs[0][0, :seq_lengths[0]], 
                atol=1e-5
            ))
            self.assertTrue(torch.allclose(
                output_batch_2[1, :seq_lengths[1]], 
                outputs[1][0, :seq_lengths[1]], 
                atol=1e-5
            ))

            # For batch_3
            for i in range(3):
                self.assertTrue(torch.allclose(
                    output_batch_3[i, :seq_lengths[i]], 
                    outputs[i][0, :seq_lengths[i]], 
                    atol=1e-5
                ))
                self.assertTrue(torch.allclose(
                    output_batch_31[i, :seq_lengths[2 - i]], 
                    outputs[2 - i][0, :seq_lengths[2 - i]], 
                    atol=1e-5
                ))

            # Test consistency of multiple forward passes with batches
            output_batch_2_repeat = self.transformer(batch_2, mask_2)
            output_batch_3_repeat = self.transformer(batch_3, mask_3)

            self.assertTrue(torch.allclose(output_batch_2, output_batch_2_repeat, atol=1e-5))
            self.assertTrue(torch.allclose(output_batch_3, output_batch_3_repeat, atol=1e-5))

if __name__ == '__main__':
    unittest.main(); exit(0) 

    suite = unittest.TestSuite()
    suite.addTest(TestTransformer('test_mask_slicing_consistency_of_different_samples'))
    unittest.TextTestRunner().run(suite)
