import unittest
import random
import numpy as np

from src.load_data import GridDataset, SpecialToken
from src.utils.data_augment import augment_compact_grids, rotate_grid, flip_grid_vertical, permute, permute_mapping, permute, SpecialToken, reverse_augment_compact_grids
from src.utils.grid_data_process import shuffle_all_but_last_pair, tokenize_compact_task, compact_grid as compact_grid_func, extend_and_tokenize_compact_grid, detokenize_to_compact_grids, compact_grid, count_grids_of_compact_grids, remove_last_grid, estimate_output_grid_token_length

class TestGridDataset(unittest.TestCase):
    def setUp(self):
        random.seed(42)  # Set a fixed seed for reproducibility

    @staticmethod
    def to_legacy_token_sequence(seq):
        return [w[0] for w in seq]

    def test_preprocess_for_fast_access(self):
        challenges = {
            "challenge1": {
                "train": [
                    {
                        "input": [[0, 1], [2, 3]],
                        "output": [[1, 2], [3, 4]]
                    }
                ],
                "test": [
                    {
                        "input": [[4, 5], [6, 7]]
                    }
                ]
            }
        }
        
        solutions = {
            "challenge1": [[[5, 6], [7, 8]]]
        }

        dataset = GridDataset.load_from_paired_file(challenges, solutions)

        self.assertEqual(len(dataset.data), 1)  # One challenge
        
        expected_sequence = [
            SpecialToken.START_INPUT.value,
            2, 2, 0, 1, 2, 3,  # Input grid
            SpecialToken.START_OUTPUT.value,
            2, 2, 1, 2, 3, 4,  # Output grid
            SpecialToken.START_INPUT.value,
            2, 2, 4, 5, 6, 7,  # Test input grid
            SpecialToken.START_OUTPUT.value,
            2, 2, 5, 6, 7, 8   # Solution grid
        ]

        self.assertEqual(list(dataset.data[0]), expected_sequence)

    def test_empty_challenges(self):
        challenges = {}
        solutions = {}
        dataset = GridDataset.load_from_paired_file(challenges, solutions)
        self.assertEqual(len(dataset.data), 0)

    def test_multiple_challenges(self):
        challenges = {
            "challenge1": {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]]}]
            },
            "challenge2": {
                "train": [{"input": [[4, 5]], "output": [[6, 7]]}],
                "test": [{"input": [[8, 9]]}]
            }
        }
        solutions = {
            "challenge1": [[[4]]],
            "challenge2": [[[10, 11]]]
        }
        dataset = GridDataset.load_from_paired_file(challenges, solutions)
        self.assertEqual(len(dataset.data), 2)

    def test_multiple_train_examples(self):
        challenges = {
            "challenge1": {
                "train": [
                    {"input": [[1]], "output": [[2]]},
                    {"input": [[3]], "output": [[4]]}
                ],
                "test": [{"input": [[5]]}]
            }
        }
        solutions = {
            "challenge1": [[[6]]]
        }
        dataset = GridDataset.load_from_paired_file(challenges, solutions)
        self.assertEqual(len(dataset.data), 1)
        expected_sequence = [
            SpecialToken.START_INPUT.value, 1, 1, 1,
            SpecialToken.START_OUTPUT.value, 1, 1, 2,
            SpecialToken.START_INPUT.value, 1, 1, 3,
            SpecialToken.START_OUTPUT.value, 1, 1, 4,
            SpecialToken.START_INPUT.value, 1, 1, 5,
            SpecialToken.START_OUTPUT.value, 1, 1, 6
        ]
        self.assertEqual(list(dataset.data[0]), expected_sequence)

    def test_different_grid_sizes(self):
        challenges = {
            "challenge1": {
                "train": [{"input": [[1, 2, 3], [4, 5, 6]], "output": [[7, 8], [9, 0]]}],
                "test": [{"input": [[1, 2], [3, 4], [5, 6]]}]
            }
        }
        solutions = {
            "challenge1": [[[7, 8], [9, 0], [1, 2]]]
        }
        dataset = GridDataset.load_from_paired_file(challenges, solutions)
        self.assertEqual(len(dataset.data), 1)
        expected_sequence = [
            SpecialToken.START_INPUT.value, 2, 3, 1, 2, 3, 4, 5, 6,
            SpecialToken.START_OUTPUT.value, 2, 2, 7, 8, 9, 0,
            SpecialToken.START_INPUT.value, 3, 2, 1, 2, 3, 4, 5, 6,
            SpecialToken.START_OUTPUT.value, 3, 2, 7, 8, 9, 0, 1, 2
        ]
        self.assertEqual(list(dataset.data[0]), expected_sequence)

    def test_missing_solution(self):
        challenges = {
            "challenge1": {
                "train": [{"input": [[1]], "output": [[2]]}],
                "test": [{"input": [[3]]}]
            },
            "challenge2": {
                "train": [{"input": [[4]], "output": [[5]]}],
                "test": [{"input": [[6]]}]
            }
        }
        solutions = {
            "challenge1": [[[7]]]
            # challenge2 solution is missing
        }
        with self.assertRaises(KeyError):
            GridDataset.load_from_paired_file(challenges, solutions)

    def test_extend_and_tokenize_compact_grid(self):
        dataset = GridDataset.load_from_paired_file({}, {})  # Empty dataset for testing
        compact_task = [9, 2, 3, 1, 2, 3, 4, 5, 6]  # 3x2 grid: [[1, 2, 3], [4, 5, 6]]
        offset = 1
        working_sequence = []

        new_offset = extend_and_tokenize_compact_grid(compact_task, offset, working_sequence, grid_index = 0)

        working_sequence = self.to_legacy_token_sequence(working_sequence)

        expected_sequence = [1, 2, 3, SpecialToken.ROW_SEPARATOR.value, 4, 5, 6, SpecialToken.ROW_SEPARATOR.value]
        expected_offset = 9

        self.assertEqual(working_sequence, expected_sequence)
        self.assertEqual(new_offset, expected_offset)

    def test_tokenize_compact_task(self):
        dataset = GridDataset.load_from_paired_file({}, {})  # Empty dataset for testing
        compact_task = [
            SpecialToken.START_INPUT.value,
            3, 2, 1, 2, 3, 4, 5, 6,
            SpecialToken.START_OUTPUT.value,
            2, 2, 7, 8, 9, 7,
            SpecialToken.START_INPUT.value,
            1, 1, 1,
            SpecialToken.START_OUTPUT.value,
            2, 2, 7, 8, 9, 7
        ]

        tokenized_sequence = tokenize_compact_task(compact_task)
        tokenized_sequence = self.to_legacy_token_sequence(tokenized_sequence)

        expected_sequence = [
            SpecialToken.START.value,
            SpecialToken.START_INPUT.value,
            1, 2, SpecialToken.ROW_SEPARATOR.value,
            3, 4, SpecialToken.ROW_SEPARATOR.value,
            5, 6, SpecialToken.ROW_SEPARATOR.value,
            SpecialToken.END_INPUT.value,
            SpecialToken.START_OUTPUT.value,
            7, 8, SpecialToken.ROW_SEPARATOR.value,
            9, 7, SpecialToken.ROW_SEPARATOR.value,
            SpecialToken.END_OUTPUT.value,
            SpecialToken.START_INPUT.value,
            1,SpecialToken.ROW_SEPARATOR.value,
            SpecialToken.END_INPUT.value,
            SpecialToken.START_OUTPUT.value,
            7, 8, SpecialToken.ROW_SEPARATOR.value,
            9, 7, SpecialToken.ROW_SEPARATOR.value,
            SpecialToken.END_OUTPUT.value,
            SpecialToken.END.value
        ]

        self.assertEqual(tokenized_sequence, expected_sequence)

    def test_tokenize_compact_task_no_solution(self):
        dataset = GridDataset.load_from_paired_file({}, {})  # Empty dataset for testing
        compact_task = [
            SpecialToken.START_INPUT.value,
            2, 3, 1, 2, 3, 4, 5, 6,
            SpecialToken.START_OUTPUT.value,
            2, 2, 7, 8, 9, 7,
            SpecialToken.START_INPUT.value,
            1, 1, 1
        ]

        tokenized_sequence = tokenize_compact_task(compact_task)
        tokenized_sequence = self.to_legacy_token_sequence(tokenized_sequence)

        expected_sequence = [
            SpecialToken.START.value,
            SpecialToken.START_INPUT.value,
            1, 2, 3, SpecialToken.ROW_SEPARATOR.value,
            4, 5, 6, SpecialToken.ROW_SEPARATOR.value,
            SpecialToken.END_INPUT.value,
            SpecialToken.START_OUTPUT.value,
            7, 8, SpecialToken.ROW_SEPARATOR.value,
            9, 7, SpecialToken.ROW_SEPARATOR.value,
            SpecialToken.END_OUTPUT.value,
            SpecialToken.START_INPUT.value,
            1,SpecialToken.ROW_SEPARATOR.value,
            SpecialToken.END_INPUT.value,
        ]

        self.assertEqual(tokenized_sequence, expected_sequence)

    def test_compact_grid(self):
        dataset = GridDataset.load_from_paired_file({}, {})  # Empty dataset for testing
        raw_grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12]
        ]
        
        compact_grid = compact_grid_func(raw_grid)
        
        expected_result = [4, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        
        self.assertEqual(compact_grid, expected_result)
        self.assertEqual(len(compact_grid), 2 + 3 * 4)  # 2 for dimensions + 3 columns * 4 rows
        self.assertEqual(compact_grid[0], 4)  # Height
        self.assertEqual(compact_grid[1], 3)  # Width
        
        # Check the assertion in compact_grid
        height = compact_grid[0]
        width = compact_grid[1]
        self.assertEqual(len(compact_grid[2:]), height * width, 
                        "The assertion in compact_grid should pass: len(result) == height * width")

        # Test with an irregular grid (should raise an AssertionError)
        irregular_grid = [
            [1, 2, 3],
            [4, 5],
            [6, 7, 8, 9]
        ]
        
        with self.assertRaises(AssertionError):
            compact_grid_func(irregular_grid)

    def test_augment_data(self):
        compact_grid = [3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        augmented_grid = augment_compact_grids(compact_grid)
        
        self.assertIsInstance(augmented_grid, list)
        self.assertEqual(len(augmented_grid), len(compact_grid))
        self.assertEqual(augmented_grid[:2], compact_grid[:2])  # Dimensions should remain the same

    def test_rotate_grid(self):
        compact_grid = [3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        rotated_grid = rotate_grid(compact_grid, 1)
        
        expected_rotated = [3, 3, 7, 4, 1, 8, 5, 2, 9, 6, 3]
        self.assertEqual(rotated_grid, expected_rotated)

    def test_flip_grid_vertical(self):
        compact_grid = [3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        flipped_grid = flip_grid_vertical(compact_grid)
        
        expected_flipped = [3, 3, 7, 8, 9, 4, 5, 6, 1, 2, 3]
        self.assertEqual(flipped_grid, expected_flipped)

        # Test with a different grid size
        compact_grid_2 = [2, 4, 1, 2, 3, 4, 5, 6, 7, 8]
        flipped_grid_2 = flip_grid_vertical(compact_grid_2)
        
        expected_flipped_2 = [2, 4, 5, 6, 7, 8, 1, 2, 3, 4]
        self.assertEqual(flipped_grid_2, expected_flipped_2)

        compact_grid_3 = [3, 4, 7, 4, 9, 5, 8, 9, 5, 9, 8, 0, 4, 0]
        flipped_grid_3 = flip_grid_vertical(compact_grid_3)
        
        expected_flipped_3 = [3, 4, 8, 0, 4, 0, 8, 9, 5, 9, 7, 4, 9, 5]
        self.assertEqual(flipped_grid_3, expected_flipped_3)

        # Test with a 1x1 grid
        compact_grid_3 = [1, 1, 5]
        flipped_grid_3 = flip_grid_vertical(compact_grid_3)
        
        expected_flipped_3 = [1, 1, 5]
        self.assertEqual(flipped_grid_3, expected_flipped_3)

        

    def test_augment_data_integration(self):
        challenges = {
            "challenge1": {
                "train": [{"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]}],
                "test": [{"input": [[9, 0], [1, 2]]}]
            }
        }
        solutions = {
            "challenge1": [[[3, 4], [5, 6]]]
        }
        
        dataset = GridDataset.load_from_paired_file(challenges, solutions)
        dataset.set_augment_seed(42)
        
        augmented_item_1 = dataset[0]
        dataset.set_augment_seed(43)
        augmented_item_2 = dataset[0]
        
        self.assertNotEqual(augmented_item_1, augmented_item_2)
        self.assertEqual(len(augmented_item_1), len(augmented_item_2))

    def test_augment_data_consistency(self):
        challenges = {
            "challenge1": {
                "train": [{"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]}],
                "test": [{"input": [[9, 0], [1, 2]]}]
            }
        }
        solutions = {
            "challenge1": [[[3, 4], [5, 6]]]
        }
        
        dataset = GridDataset.load_from_paired_file(challenges, solutions)
        dataset.set_augment_seed(42)
        
        augmented_item1 = dataset[0]
        augmented_item2 = dataset[0]
        
        self.assertNotEqual(augmented_item1, augmented_item2)

        dataset.set_augment_seed(43)
        augmented_item3 = dataset[0]
        
        self.assertNotEqual(augmented_item1, augmented_item3)

    @unittest.skip
    def test_augment_data_hacked(self):
        compact_grid = [3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6]
        augmented_grid = augment_compact_grids(compact_grid)
        
        self.assertIsInstance(augmented_grid, list)
        self.assertEqual(len(augmented_grid), len(compact_grid))
        self.assertEqual(augmented_grid[:2], compact_grid[:2])  # Dimensions should remain the same

        self.assertEqual(augmented_grid, compact_grid)

    def test_permute_mapping(self):
        compact_grid = [2, 2, 1, 2, 3, 4]
        mapping = permute_mapping()
        self.assertEqual(len(mapping), 10)  # 0, 1, 2, 3, 4
        self.assertEqual(mapping[0], 0)
         # based on a fixed seed:
        self.assertNotEqual(mapping[1], 1)
        self.assertNotEqual(mapping[2], 2)
        self.assertNotEqual(mapping[3], 3)
        self.assertNotEqual(mapping[4], 4)

    def test_permute(self):
        compact_grid = [2, 2, 1, 2, 3, 4]
        mapping = {0: 0, 1: 3, 2: 1, 3: 4, 4: 2}
        permuted = permute(compact_grid, mapping)
        self.assertEqual(permuted, [2, 2, 3, 1, 4, 2])

    def test_augment_compact_grid_consistency(self):
        compact_grids = [2, 2, 1, 2, 3, 4, SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8]
        random.seed(43)
        augmented1 = augment_compact_grids(compact_grids)
        random.seed(43)
        augmented2 = augment_compact_grids(compact_grids)
        self.assertEqual(augmented1, augmented2)

    def test_augment_compact_grid_different_seed(self):
        compact_grids = [2, 2, 1, 2, 3, 4, SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8]
        augmented1 = augment_compact_grids(compact_grids)
        random.seed(43)
        augmented2 = augment_compact_grids(compact_grids)
        self.assertNotEqual(augmented1, augmented2)

    def test_augment_compact_grid_structure(self):
        compact_grids = [2, 2, 1, 2, 3, 4, SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8]
        augmented = augment_compact_grids(compact_grids)
        self.assertEqual(len(augmented), len(compact_grids))
        self.assertEqual(augmented[6], SpecialToken.START_OUTPUT.value)
        self.assertEqual(augmented[0], 2)
        self.assertEqual(augmented[1], 2)
        self.assertEqual(augmented[7], 2)
        self.assertEqual(augmented[8], 2)

    def test_augment_compact_grid_color_preservation(self):
        compact_grids = [2, 2, 1, 2, 3, 4, SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8]
        augmented = augment_compact_grids(compact_grids)
        original_colors = set(compact_grids[2:6] + compact_grids[9:])
        augmented_colors = set(augmented[2:6] + augmented[9:])
        self.assertNotEqual(original_colors, augmented_colors)

    def test_augment_longer_compact_grids(self):
        compact_grids = [
            2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value,
            2, 2, 5, 6, 7, 8,
            SpecialToken.START_INPUT.value,
            3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            SpecialToken.START_OUTPUT.value,
            3, 3, 9, 8, 7, 6, 5, 4, 3, 2, 1
        ]
        augmented = augment_compact_grids(compact_grids)

        # Check if the length of the augmented grid is the same as the original
        self.assertEqual(len(augmented), len(compact_grids))

        # Check if special tokens are preserved
        self.assertEqual(augmented[6], SpecialToken.START_OUTPUT.value)
        self.assertEqual(augmented[13], SpecialToken.START_INPUT.value)
        self.assertEqual(augmented[25], SpecialToken.START_OUTPUT.value)

        # Check if grid dimensions are preserved
        self.assertEqual(augmented[0:2], [2, 2])
        self.assertEqual(augmented[7:9], [2, 2])
        self.assertEqual(augmented[14:16], [3, 3])
        self.assertEqual(augmented[26:28], [3, 3])

        # Check if the colors in each grid are consistent (permuted but preserved)
        original_colors = set(compact_grids[2:6] + compact_grids[9:13] + 
                              compact_grids[16:25] + compact_grids[28:])
        augmented_colors = set(augmented[2:6] + augmented[9:13] + 
                               augmented[16:25] + augmented[28:])
        self.assertEqual(len(original_colors), len(augmented_colors))

        # Check if the augmented grids are different from the original
        self.assertNotEqual(compact_grids[2:6], augmented[2:6])
        self.assertNotEqual(compact_grids[9:13], augmented[9:13])
        self.assertNotEqual(compact_grids[16:25], augmented[16:25])
        self.assertNotEqual(compact_grids[28:], augmented[28:])

    def test_shuffle_first_n_1_compact_grid_pairs(self):
        dataset = GridDataset.load_from_paired_file({}, {})  # Empty dataset for testing
        random.seed(43)  # Set a fixed seed for reproducibility

        # Test case 1: Multiple input-output pairs
        compact_task = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8,
            SpecialToken.START_INPUT.value, 2, 2, 9, 0, 1, 2,
            SpecialToken.START_OUTPUT.value, 2, 2, 3, 4, 5, 6,
            SpecialToken.START_INPUT.value, 2, 2, 5, 0, 1, 2,
            SpecialToken.START_OUTPUT.value, 2, 2, 6, 4, 5, 6,
            SpecialToken.START_INPUT.value, 2, 2, 7, 8, 9, 0,
            SpecialToken.START_OUTPUT.value, 2, 2, 1, 2, 3, 4
        ]

        shuffled_task = shuffle_all_but_last_pair(compact_task)

        # Check that the last pair is not shuffled
        self.assertEqual(shuffled_task[-12:], compact_task[-12:])
        # Check that the first two pairs are shuffled
        self.assertNotEqual(shuffled_task[:24], compact_task[:24])
        # Check that all elements are present
        self.assertEqual(sorted(shuffled_task), sorted(compact_task))

        # Test case 2: Single input-output pair
        compact_task_single = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8
        ]

        shuffled_task_single = shuffle_all_but_last_pair(compact_task_single)

        # Check that the single pair is not shuffled
        self.assertEqual(shuffled_task_single, compact_task_single)

        # Test case 3: Multiple inputs with last input missing output
        compact_task_missing_output = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8,
            SpecialToken.START_INPUT.value, 2, 2, 9, 0, 1, 2,
            SpecialToken.START_OUTPUT.value, 2, 2, 3, 4, 5, 6,
            SpecialToken.START_INPUT.value, 2, 2, 7, 8, 9, 0
        ]

        shuffled_task_missing_output = shuffle_all_but_last_pair(compact_task_missing_output)

        # Check that the last input without output is not shuffled
        self.assertEqual(shuffled_task_missing_output[-6:], compact_task_missing_output[-6:])
        # Check that the first pair is shuffled
        self.assertNotEqual(shuffled_task_missing_output[:24], compact_task_missing_output[:24])
        # Check that all elements are present
        self.assertEqual(sorted(shuffled_task_missing_output), sorted(compact_task_missing_output))


        compact_task_large_w_and_h = [13, 10, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 7, 7, 7, 9, 7, 9, 7, 7, 7, 7, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 15, 10, 10, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 7, 7, 7, 8, 7, 8, 7, 7, 7, 7, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 7, 7, 7, 9, 7, 9, 7, 7, 7, 7, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 13, 15, 12, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 7, 0, 0, 0, 0, 9, 7, 9, 0, 0, 0, 0, 7, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 7, 0, 0, 0, 0, 9, 7, 9, 0, 0, 0, 0, 7, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 15, 15, 12, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 8, 8, 8, 0, 0, 8, 9, 7, 9, 8, 8, 8, 8, 7, 8, 8, 8, 0, 9, 9, 9, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 9, 9, 9, 0, 0, 8, 8, 7, 8, 8, 8, 8, 9, 7, 9, 8, 8, 0, 8, 8, 8, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 7, 0, 0, 0, 13, 12, 12, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 15, 12, 12, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 8, 8, 8, 8, 8, 8, 9, 7, 9, 8, 8, 8, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 13, 16, 14, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 9, 7, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 9, 7, 9, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 16, 14, 0, 0, 9, 9, 9, 0, 0, 0, 8, 8, 8, 0, 0, 0, 7, 7, 9, 7, 9, 7, 7, 7, 8, 7, 8, 7, 7, 7, 0, 0, 9, 9, 9, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 7, 7, 8, 7, 8, 7, 7, 7, 8, 7, 8, 7, 7, 7, 0, 0, 8, 8, 8, 0, 0, 0, 8, 8, 8, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 8, 8, 8, 0, 0, 0, 9, 9, 9, 0, 0, 0, 7, 7, 8, 7, 8, 7, 7, 7, 9, 7, 9, 7, 7, 7, 0, 0, 8, 8, 8, 0, 0, 0, 9, 9, 9, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 13, 16, 18, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 9, 9, 9, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 9, 7, 9, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 9, 9, 9, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 9, 9, 9, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 9, 7, 9, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 9, 9, 9, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 15, 16, 18, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 8, 8, 8, 0, 0, 9, 9, 9, 0, 8, 8, 8, 0, 0, 8, 8, 8, 0, 8, 7, 8, 8, 8, 9, 7, 9, 8, 8, 7, 8, 8, 8, 8, 7, 8, 8, 8, 8, 8, 0, 0, 9, 9, 9, 0, 8, 8, 8, 0, 0, 8, 8, 8, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 8, 8, 8, 0, 0, 8, 8, 8, 0, 8, 8, 8, 0, 0, 9, 9, 9, 0, 8, 7, 8, 8, 8, 8, 7, 8, 8, 8, 7, 8, 8, 8, 9, 7, 9, 8, 8, 8, 8, 0, 0, 8, 8, 8, 0, 8, 8, 8, 0, 0, 9, 9, 9, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 0, 0, 0, 7, 0, 0]
        shuffled_compact_task_large_w_and_h = shuffle_all_but_last_pair(compact_task_large_w_and_h)
        self.assertNotEqual(shuffled_task_missing_output[1:3], shuffled_compact_task_large_w_and_h[1:3])

    def test_tokenize_detokenize_conversion(self):
        # Test case 1: Simple grid
        compact_task = [
            SpecialToken.START_INPUT.value,
            2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value,
            2, 2, 5, 6, 7, 8,
            SpecialToken.START_INPUT.value,
            2, 3, 1, 2, 3, 4, 0, 9,
            SpecialToken.START_OUTPUT.value,
            3, 2, 5, 6, 7, 8, 9, 0,
        ]
        
        tokenized = tokenize_compact_task(compact_task)
        detokenized = detokenize_to_compact_grids(tokenized)
        
        self.assertEqual(compact_task, detokenized)

        # Test case 2: Multiple input-output pairs
        compact_task_multiple = [
            SpecialToken.START_INPUT.value,
            2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value,
            2, 2, 5, 6, 7, 8,
            SpecialToken.START_INPUT.value,
            2, 3, 9, 0, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value,
            3, 2, 5, 6, 7, 8, 9, 0
        ]
        
        tokenized_multiple = tokenize_compact_task(compact_task_multiple)
        detokenized_multiple = detokenize_to_compact_grids(tokenized_multiple)
        
        self.assertEqual(compact_task_multiple, detokenized_multiple)

        # Test case 3: Different grid sizes
        compact_task_different_sizes = [
            SpecialToken.START_INPUT.value,
            3, 2, 1, 2, 3, 4, 5, 6,
            SpecialToken.START_OUTPUT.value,
            2, 3, 7, 8, 9, 1, 2, 3
        ]
        
        tokenized_different = tokenize_compact_task(compact_task_different_sizes)
        detokenized_different = detokenize_to_compact_grids(tokenized_different)
        
        self.assertEqual(compact_task_different_sizes, detokenized_different)

        # Test case 4: Single cell grids
        compact_task_single = [
            SpecialToken.START_INPUT.value,
            1, 1, 5,
            SpecialToken.START_OUTPUT.value,
            1, 1, 7
        ]
        
        tokenized_single = tokenize_compact_task(compact_task_single)
        detokenized_single = detokenize_to_compact_grids(tokenized_single)
        
        self.assertEqual(compact_task_single, detokenized_single)

        # Test case 5: Missing output (test input only)
        compact_task_no_output = [
            SpecialToken.START_INPUT.value,
            2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value,
            2, 2, 5, 6, 7, 8,
            SpecialToken.START_INPUT.value,
            2, 2, 9, 0, 1, 2
        ]
        
        tokenized_no_output = tokenize_compact_task(compact_task_no_output)
        detokenized_no_output = detokenize_to_compact_grids(tokenized_no_output)
        
        self.assertEqual(compact_task_no_output, detokenized_no_output)

        # Test case 6: Real example from ARC dataset
        compact_task_arc = [
            SpecialToken.START_INPUT.value, 3, 1, 0, 7, 8,
            SpecialToken.START_OUTPUT.value, 1, 1, 0,
            SpecialToken.START_INPUT.value, 1, 3, 1, 5, 1,
            SpecialToken.START_OUTPUT.value, 1, 1, 0
        ]
        
        tokenized_arc = tokenize_compact_task(compact_task_arc)
        detokenized_arc = detokenize_to_compact_grids(tokenized_arc)
        
        self.assertEqual(compact_task_arc, detokenized_arc)

    def test_tokenize_detokenize_structure(self):
        """Test that the tokenized sequence maintains proper structure"""
        compact_task = [
            SpecialToken.START_INPUT.value,
            2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value,
            2, 2, 5, 6, 7, 8
        ]
        
        tokenized = tokenize_compact_task(compact_task)
        
        # Check structure of tokenized sequence
        self.assertEqual(tokenized[0][0], SpecialToken.START.value)
        self.assertEqual(tokenized[1][0], SpecialToken.START_INPUT.value)
        
        # Check that each grid cell has proper format [token, y, x, chart, grid_index]
        for token_info in tokenized:
            self.assertEqual(len(token_info), 5)
            self.assertIsInstance(token_info[0], int)  # token
            self.assertIsInstance(token_info[1], int)  # y
            self.assertIsInstance(token_info[2], int)  # x
            self.assertIsInstance(token_info[3], int)  # chart
            self.assertIsInstance(token_info[4], int)  # grid_index

    def test_tokenize_detokenize_edge_cases(self):
        """Test edge cases in tokenization/detokenization"""
        # Empty grid
        compact_task_empty = [
            SpecialToken.START_INPUT.value,
            0, 0,
            SpecialToken.START_OUTPUT.value,
            0, 0
        ]
        
        tokenized_empty = tokenize_compact_task(compact_task_empty)
        detokenized_empty = detokenize_to_compact_grids(tokenized_empty)
        
        self.assertEqual(compact_task_empty, detokenized_empty)

        # Large grid
        large_grid = [[i for i in range(10)] for _ in range(10)]
        compact_large = [
            SpecialToken.START_INPUT.value,
            *compact_grid(large_grid),
            SpecialToken.START_OUTPUT.value,
            *compact_grid(large_grid)
        ]
        
        tokenized_large = tokenize_compact_task(compact_large)
        detokenized_large = detokenize_to_compact_grids(tokenized_large)
        
        self.assertEqual(compact_large, detokenized_large)

    def test_tokenize_detokenize_conversion_with_actual_data(self):
        compact_task_arc = [13, 9, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 9, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 2, 2, 2, 5, 5, 0, 0, 0, 0, 0, 0, 0, 5, 5, 2, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 2, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 7, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 5, 2, 5, 5, 0, 0, 0, 0, 2, 5, 5, 2, 5, 5, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5, 5, 2, 0, 0, 0, 0, 0, 0, 0, 5, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 2, 2, 2, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 5, 5, 2, 0, 0, 0, 0, 0, 0, 5, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 8, 11, 0, 0, 0, 7, 7, 7, 0, 7, 7, 0, 0, 0, 0, 0, 0, 7, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 8, 11, 0, 0, 0, 2, 2, 2, 0, 5, 5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 2, 5, 5, 2, 2, 2, 0, 0, 0, 0, 0, 2, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        tokenized_arc = tokenize_compact_task(compact_task_arc)
        detokenized_arc = detokenize_to_compact_grids(tokenized_arc)
        
        self.assertEqual(compact_task_arc, detokenized_arc)

    def test_count_grids_of_compact_grids(self):
        # Test case 1: Multiple input-output pairs
        compact_grid = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8,
            SpecialToken.START_INPUT.value, 2, 2, 9, 0, 1, 2,
            SpecialToken.START_OUTPUT.value, 2, 2, 3, 4, 5, 6
        ]
        self.assertEqual(count_grids_of_compact_grids(compact_grid), 2)

        # Test case 2: Single input-output pair
        compact_grid_single = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8
        ]
        self.assertEqual(count_grids_of_compact_grids(compact_grid_single), 1)

        # Test case 3: Input only (no output)
        compact_grid_input_only = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4
        ]
        self.assertEqual(count_grids_of_compact_grids(compact_grid_input_only), 1)

        # Test case 4: Empty sequence
        compact_grid_empty = []
        self.assertEqual(count_grids_of_compact_grids(compact_grid_empty), 0)

    def test_remove_last_grid(self):
        # Test case 1: Multiple input-output pairs
        compact_grid = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8,
            SpecialToken.START_INPUT.value, 2, 2, 9, 0, 1, 2,
            SpecialToken.START_OUTPUT.value, 2, 2, 3, 4, 5, 6
        ]
        expected_result = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8
        ]
        self.assertEqual(remove_last_grid(compact_grid), expected_result)

        # Test case 2: Single input-output pair
        compact_grid_single = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8
        ]
        self.assertEqual(remove_last_grid(compact_grid_single), [])

        # Test case 3: Input only (no output)
        compact_grid_input_only = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4
            ]
        self.assertEqual(remove_last_grid(compact_grid_input_only), [])

        # Test case 4: Empty sequence
        compact_grid_empty = []
        self.assertEqual(remove_last_grid(compact_grid_empty), [])

        # Test case 5: Complex sequence with multiple grids
        complex_grid = [
            SpecialToken.START_INPUT.value, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            SpecialToken.START_OUTPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_INPUT.value, 1, 1, 5,
            SpecialToken.START_OUTPUT.value, 1, 1, 6,
            SpecialToken.START_INPUT.value, 2, 3, 7, 8, 9, 1, 2, 3,
        ]
        expected_complex = [
            SpecialToken.START_INPUT.value, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            SpecialToken.START_OUTPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_INPUT.value, 1, 1, 5,
            SpecialToken.START_OUTPUT.value, 1, 1, 6,
        ]
        self.assertEqual(remove_last_grid(complex_grid), expected_complex)

        # Test case 6: Real example from ARC dataset
        arc_grid = [
            SpecialToken.START_INPUT.value, 3, 1, 0, 7, 8,
            SpecialToken.START_OUTPUT.value, 1, 1, 0,
            SpecialToken.START_INPUT.value, 1, 3, 1, 5, 1,
            SpecialToken.START_OUTPUT.value, 1, 1, 0,
            SpecialToken.START_INPUT.value, 2, 2, 5, 2, 3, 8,
            SpecialToken.START_OUTPUT.value, 1, 1, 0,
        ]
        expected_arc = [
            SpecialToken.START_INPUT.value, 3, 1, 0, 7, 8,
            SpecialToken.START_OUTPUT.value, 1, 1, 0,
            SpecialToken.START_INPUT.value, 1, 3, 1, 5, 1,
            SpecialToken.START_OUTPUT.value, 1, 1, 0
        ]
        self.assertEqual(remove_last_grid(arc_grid), expected_arc)


        real_data =                     [13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 3, 0, 4, 0, 0, 9, 9, 9, 0, 5, 0, 0, 7, 3, 0, 4, 0, 0, 9, 0, 9, 0, 5, 0, 0, 7, 3, 0, 4, 0, 0, 9, 0, 9, 0, 5, 0, 0, 7, 3, 3, 4, 9, 9, 9, 0, 9, 5, 5, 0, 0, 7, 3, 0, 4, 0, 0, 9, 0, 9, 0, 5, 0, 0, 7, 3, 0, 4, 0, 0, 9, 0, 9, 0, 5, 0, 0, 7, 3, 0, 4, 0, 0, 9, 9, 9, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 9, 9, 9, 4, 3, 0, 7, 0, 0, 0, 5, 0, 0, 9, 0, 9, 4, 3, 0, 7, 0, 0, 0, 5, 0, 0, 9, 0, 9, 4, 3, 0, 7, 0, 0, 5, 5, 9, 9, 9, 0, 9, 4, 3, 3, 7, 0, 0, 0, 5, 0, 0, 9, 0, 9, 4, 3, 0, 7, 0, 0, 0, 5, 0, 0, 9, 0, 9, 4, 3, 0, 7, 0, 0, 0, 5, 0, 0, 9, 9, 9, 4, 3, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 9, 0, 3, 3, 3, 0, 0, 0, 5, 8, 8, 0, 0, 9, 0, 3, 0, 3, 0, 0, 0, 5, 8, 8, 0, 0, 9, 9, 3, 0, 3, 4, 4, 0, 5, 8, 0, 0, 0, 0, 9, 3, 0, 3, 4, 0, 4, 5, 8, 0, 0, 0, 9, 9, 3, 0, 3, 4, 4, 0, 5, 8, 0, 0, 0, 9, 0, 3, 0, 3, 0, 0, 0, 5, 8, 8, 0, 0, 9, 0, 3, 3, 3, 0, 0, 0, 5, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 5, 0, 0, 0, 3, 3, 3, 9, 0, 0, 0, 8, 8, 5, 0, 0, 0, 3, 0, 3, 9, 0, 0, 0, 8, 0, 5, 4, 4, 0, 3, 0, 3, 9, 9, 0, 0, 8, 0, 5, 4, 0, 4, 3, 0, 3, 0, 9, 0, 0, 8, 0, 5, 4, 4, 0, 3, 0, 3, 9, 9, 0, 0, 8, 8, 5, 0, 0, 0, 3, 0, 3, 9, 0, 0, 0, 8, 8, 5, 0, 0, 0, 3, 3, 3, 9, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 4, 0, 0, 0, 9, 3, 3, 4, 0, 0, 0, 9, 3, 3, 4, 4, 0, 0, 9, 3, 3, 4, 0, 0, 0, 9, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 4, 0, 3, 3, 9, 0, 0, 4, 4, 3, 3, 9, 0, 0, 4, 0, 3, 3, 9, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 13, 10, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 4, 0, 4, 3, 3, 3, 0, 5, 7, 0, 1, 0, 0, 4, 4, 4, 3, 0, 3, 3, 5, 0, 7, 1, 0, 0, 4, 4, 4, 3, 0, 3, 3, 5, 0, 7, 1, 0, 0, 4, 0, 4, 3, 3, 3, 0, 5, 7, 0, 1, 0, 0, 4, 4, 4, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 10, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 4, 4, 0, 0, 1, 7, 0, 5, 3, 3, 3, 0, 4, 0, 4, 0, 0, 1, 0, 7, 5, 3, 0, 3, 3, 4, 4, 4, 0, 0, 1, 0, 7, 5, 3, 0, 3, 3, 4, 4, 4, 0, 0, 1, 7, 0, 5, 3, 3, 3, 0, 4, 0, 4, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        expected_answer_for_real_data = [13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 3, 0, 4, 0, 0, 9, 9, 9, 0, 5, 0, 0, 7, 3, 0, 4, 0, 0, 9, 0, 9, 0, 5, 0, 0, 7, 3, 0, 4, 0, 0, 9, 0, 9, 0, 5, 0, 0, 7, 3, 3, 4, 9, 9, 9, 0, 9, 5, 5, 0, 0, 7, 3, 0, 4, 0, 0, 9, 0, 9, 0, 5, 0, 0, 7, 3, 0, 4, 0, 0, 9, 0, 9, 0, 5, 0, 0, 7, 3, 0, 4, 0, 0, 9, 9, 9, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 9, 9, 9, 4, 3, 0, 7, 0, 0, 0, 5, 0, 0, 9, 0, 9, 4, 3, 0, 7, 0, 0, 0, 5, 0, 0, 9, 0, 9, 4, 3, 0, 7, 0, 0, 5, 5, 9, 9, 9, 0, 9, 4, 3, 3, 7, 0, 0, 0, 5, 0, 0, 9, 0, 9, 4, 3, 0, 7, 0, 0, 0, 5, 0, 0, 9, 0, 9, 4, 3, 0, 7, 0, 0, 0, 5, 0, 0, 9, 9, 9, 4, 3, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 9, 0, 3, 3, 3, 0, 0, 0, 5, 8, 8, 0, 0, 9, 0, 3, 0, 3, 0, 0, 0, 5, 8, 8, 0, 0, 9, 9, 3, 0, 3, 4, 4, 0, 5, 8, 0, 0, 0, 0, 9, 3, 0, 3, 4, 0, 4, 5, 8, 0, 0, 0, 9, 9, 3, 0, 3, 4, 4, 0, 5, 8, 0, 0, 0, 9, 0, 3, 0, 3, 0, 0, 0, 5, 8, 8, 0, 0, 9, 0, 3, 3, 3, 0, 0, 0, 5, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 5, 0, 0, 0, 3, 3, 3, 9, 0, 0, 0, 8, 8, 5, 0, 0, 0, 3, 0, 3, 9, 0, 0, 0, 8, 0, 5, 4, 4, 0, 3, 0, 3, 9, 9, 0, 0, 8, 0, 5, 4, 0, 4, 3, 0, 3, 0, 9, 0, 0, 8, 0, 5, 4, 4, 0, 3, 0, 3, 9, 9, 0, 0, 8, 8, 5, 0, 0, 0, 3, 0, 3, 9, 0, 0, 0, 8, 8, 5, 0, 0, 0, 3, 3, 3, 9, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 4, 0, 0, 0, 9, 3, 3, 4, 0, 0, 0, 9, 3, 3, 4, 4, 0, 0, 9, 3, 3, 4, 0, 0, 0, 9, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 9, 0, 0, 4, 0, 3, 3, 9, 0, 0, 4, 4, 3, 3, 9, 0, 0, 4, 0, 3, 3, 9, 0, 0, 4, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertEqual(remove_last_grid(real_data), expected_answer_for_real_data)

    def test_reverse_augment_compact_grids(self):
        # Test case 1: Basic grid with all transformations

        for test_index in range(5):
            original_grid = [
                SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
                SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8
            ]
            
            # First augment the grid
            augmented, rotation, flip, mapping = augment_compact_grids(original_grid, return_config=True)
            
            # Then reverse the augmentation
            reversed_grid = reverse_augment_compact_grids(augmented, rotation, flip, mapping)
            
            # The reversed grid should match the original
            self.assertEqual(original_grid, reversed_grid)

            # Test case 2: Multiple input-output pairs
            complex_grid = [
                SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
                SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8,
                SpecialToken.START_INPUT.value, 2, 3, 1, 2, 3, 4, 5, 6,
                SpecialToken.START_OUTPUT.value, 3, 2, 7, 8, 9, 1, 2, 3
            ]
            
            augmented, rotation, flip, mapping = augment_compact_grids(complex_grid, return_config=True)
            reversed_grid = reverse_augment_compact_grids(augmented, rotation, flip, mapping)
            
            self.assertEqual(complex_grid, reversed_grid)

            # Test case 3: No transformations
            simple_grid = [
                SpecialToken.START_INPUT.value, 1, 1, 5,
                SpecialToken.START_OUTPUT.value, 1, 1, 7
            ]
            
            augmented, rotation, flip, mapping = augment_compact_grids(simple_grid, return_config=True)
            reversed_grid = reverse_augment_compact_grids(augmented, rotation, flip, mapping)
            
            self.assertEqual(simple_grid, reversed_grid)

            # Test case 4: Test with actual augmentation config
            test_grid = [
                SpecialToken.START_INPUT.value, 2, 3, 1, 2, 3, 4, 5, 6,
                SpecialToken.START_OUTPUT.value, 2, 2, 7, 8, 9, 1
            ]
            
            # Get the actual augmentation configuration
            augmented, rotation, flip, mapping = augment_compact_grids(test_grid, return_config=True)
            
            # Reverse the augmentation using the returned configuration
            reversed_grid = reverse_augment_compact_grids(augmented, rotation, flip, mapping)
            
            self.assertEqual(test_grid, reversed_grid)

            # Test case 5: Edge case with empty grid sections
            edge_grid = [
                SpecialToken.START_INPUT.value, 2, 2, 0, 0, 0, 0,
                SpecialToken.START_OUTPUT.value, 1, 1, 0
            ]
            
            augmented, rotation, flip, mapping = augment_compact_grids(edge_grid, return_config=True)
            reversed_grid = reverse_augment_compact_grids(augmented, rotation, flip, mapping)
            
            self.assertEqual(edge_grid, reversed_grid)

    def test_estimate_output_grid_token_length(self):
        # Test case 1: Simple grid with consistent output size
        compact_task = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8,
            SpecialToken.START_INPUT.value, 3, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8
        ]
        self.assertEqual(estimate_output_grid_token_length(compact_task), 4)  # 2x2 is the largest output

        # Test case 2: Varying output sizes
        compact_task_varying = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 3, 3, 5, 6, 7, 8, 9, 1, 2, 3, 4,
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 2, 2, 5, 6, 7, 8
        ]
        self.assertEqual(estimate_output_grid_token_length(compact_task_varying), 9)  # 3x3 is the largest output

        # Test case 3: Single input-output pair
        compact_task_single = [
            SpecialToken.START_INPUT.value, 1, 1, 5,
            SpecialToken.START_OUTPUT.value, 2, 3, 7, 8, 9, 1, 2, 3
        ]
        self.assertEqual(estimate_output_grid_token_length(compact_task_single), 6)  # 2x3 output

        # Test case 4: Input only (no output)
        compact_task_no_output = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4
        ]
        self.assertEqual(estimate_output_grid_token_length(compact_task_no_output), 0)  # No output

        # Test case 5: Multiple pairs with empty grids
        compact_task_empty = [
            SpecialToken.START_INPUT.value, 2, 2, 1, 2, 3, 4,
            SpecialToken.START_OUTPUT.value, 0, 0,
            SpecialToken.START_INPUT.value, 1, 1, 5,
            SpecialToken.START_OUTPUT.value, 1, 1, 7
        ]
        self.assertEqual(estimate_output_grid_token_length(compact_task_empty), 1)  # 1x1 is the largest output

        # Test case 6: Real example from ARC dataset
        compact_task_arc = [
            SpecialToken.START_INPUT.value, 3, 1, 0, 7, 8,
            SpecialToken.START_OUTPUT.value, 1, 1, 0,
            SpecialToken.START_INPUT.value, 1, 3, 1, 5, 1,
            SpecialToken.START_OUTPUT.value, 2, 2, 0, 1, 2, 3
        ]
        self.assertEqual(estimate_output_grid_token_length(compact_task_arc), 4)  # 2x2 is the largest output            

if __name__ == '__main__':
    unittest.main(); exit(0) 

    suite = unittest.TestSuite()
    suite.addTest(TestGridDataset('test_estimate_output_grid_token_length'))
    unittest.TextTestRunner().run(suite)
