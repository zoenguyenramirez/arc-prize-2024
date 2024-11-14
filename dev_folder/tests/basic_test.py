import unittest
from src.augment_json import augment_data
from src.augment_json import flip_grid_vertical, rotate_grid
import copy

class TestFlipGridVertical(unittest.TestCase):

    def test_square_grid(self):
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        expected = [
            [7, 8, 9],
            [4, 5, 6],
            [1, 2, 3]
        ]
        self.assertEqual(flip_grid_vertical(grid), expected)

    def test_rectangular_grid(self):
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
        expected = [
            [5, 6, 7, 8],
            [1, 2, 3, 4]
        ]
        self.assertEqual(flip_grid_vertical(grid), expected)

    def test_single_row_grid(self):
        grid = [[1, 2, 3, 4]]
        expected = [[1, 2, 3, 4]]
        self.assertEqual(flip_grid_vertical(grid), expected)

    def test_single_column_grid(self):
        grid = [[1], [2], [3], [4]]
        expected = [[4], [3], [2], [1]]
        self.assertEqual(flip_grid_vertical(grid), expected)

    def test_empty_grid(self):
        grid = []
        expected = []
        self.assertEqual(flip_grid_vertical(grid), expected)

class TestRotateGrid(unittest.TestCase):

    def test_rotate_0_degrees(self):
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        expected = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        self.assertEqual(rotate_grid(grid, 0), expected)

    def test_rotate_90_degrees(self):
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        expected = [
            [7, 4, 1],
            [8, 5, 2],
            [9, 6, 3]
        ]
        self.assertEqual(rotate_grid(grid, 1), expected)

    def test_rotate_180_degrees(self):
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        expected = [
            [9, 8, 7],
            [6, 5, 4],
            [3, 2, 1]
        ]
        self.assertEqual(rotate_grid(grid, 2), expected)

    def test_rotate_270_degrees(self):
        grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        expected = [
            [3, 6, 9],
            [2, 5, 8],
            [1, 4, 7]
        ]
        self.assertEqual(rotate_grid(grid, 3), expected)

    def test_rectangular_grid(self):
        grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
        expected = [
            [5, 1],
            [6, 2],
            [7, 3],
            [8, 4]
        ]
        self.assertEqual(rotate_grid(grid, 1), expected)

    def test_single_row_grid(self):
        grid = [[1, 2, 3, 4]]
        expected = [[1], [2], [3], [4]]
        self.assertEqual(rotate_grid(grid, 1), expected)

    def test_single_column_grid(self):
        grid = [[1], [2], [3], [4]]
        expected = [[4, 3, 2, 1]]
        self.assertEqual(rotate_grid(grid, 1), expected)

    def test_invalid_rotation(self):
        grid = [[1, 2], [3, 4]]
        with self.assertRaises(ValueError):
            rotate_grid(grid, 4)


class TestAugmentProcess(unittest.TestCase):

    def setUp(self):
        self.original_grid = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]

    def test_deepcopy_rotate_flip(self):
        # Test for all rotations (0, 90, 180, 270 degrees)
        for rotation in range(4):
            with self.subTest(rotation=rotation*90):
                # Step 1: Deep copy
                copied_grid = copy.deepcopy(self.original_grid)
                self.assertEqual(copied_grid, self.original_grid)
                self.assertIsNot(copied_grid, self.original_grid)

                # Step 2: Rotate
                rotated_grid = rotate_grid(copied_grid, rotation)
                expected_rotated = self.get_expected_rotation(rotation)
                self.assertEqual(rotated_grid, expected_rotated)

                # Step 3: Flip
                flipped_grid = flip_grid_vertical(rotated_grid)
                expected_flipped = self.get_expected_flip(expected_rotated)
                self.assertEqual(flipped_grid, expected_flipped)

    def get_expected_rotation(self, rotation):
        if rotation == 0:
            return [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        elif rotation == 1:
            return [
                [7, 4, 1],
                [8, 5, 2],
                [9, 6, 3]
            ]
        elif rotation == 2:
            return [
                [9, 8, 7],
                [6, 5, 4],
                [3, 2, 1]
            ]
        elif rotation == 3:
            return [
                [3, 6, 9],
                [2, 5, 8],
                [1, 4, 7]
            ]

    def get_expected_flip(self, grid):
        return grid[::-1]

    def test_deepcopy_rotate_flip_rectangular(self):
        rectangular_grid = [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
        
        # Test for 90 degree rotation and flip
        copied_grid = copy.deepcopy(rectangular_grid)
        self.assertEqual(copied_grid, rectangular_grid)
        self.assertIsNot(copied_grid, rectangular_grid)

        rotated_grid = rotate_grid(copied_grid, 1)
        expected_rotated = [
            [5, 1],
            [6, 2],
            [7, 3],
            [8, 4]
        ]
        self.assertEqual(rotated_grid, expected_rotated)

        flipped_grid = flip_grid_vertical(rotated_grid)
        expected_flipped = [
            [8, 4],
            [7, 3],
            [6, 2],
            [5, 1]
        ]
        self.assertEqual(flipped_grid, expected_flipped)


class TestAugmentData(unittest.TestCase):

    def setUp(self):
        # Sample input data
        self.challenges = {
            "task1": {
                "train": [
                    {"input": [[1, 2], [3, 4]], "output": [[5, 6], [7, 8]]},
                ],
                "test": [
                    {"input": [[1, 2], [3, 4]]},
                ]
            }
        }
        self.solutions = {
            "task1": [[[5, 6], [7, 8]]]
        }

    def test_augment_data_output_size(self):
        augmented_challenges, augmented_solutions = augment_data(self.challenges, self.solutions, perm_count = 40)
        
        # Check if the number of augmented tasks is correct
        # (1 original + 3 rotations + 4 flipped versions) * 41 permutations = 48
        self.assertEqual(len(augmented_challenges), (1 + 3) * 2 * 41)
        self.assertEqual(len(augmented_solutions), (1 + 3) * 2 * 41)

    def test_augment_data_content(self):
        augmented_challenges, augmented_solutions = augment_data(self.challenges, self.solutions, perm_count = 2)
        
        # Check if original task is preserved
        self.assertIn("task1", augmented_challenges)
        self.assertIn("task1", augmented_solutions)
        
        # Check for rotated versions
        self.assertIn("task1_rot90", augmented_challenges.keys())
        self.assertIn("task1_rot180", augmented_challenges)
        self.assertIn("task1_rot270", augmented_challenges)
        
        # Check for flipped versions
        self.assertIn("task1_rot0_flipped", augmented_challenges)
        self.assertIn("task1_rot90_flipped", augmented_challenges)
        
        # Check for permuted versions (just check if at least one exists)
        self.assertTrue(any("task1_p" in key for key in augmented_challenges.keys()))

    def test_augment_data_structure(self):
        augmented_challenges, augmented_solutions = augment_data(self.challenges, self.solutions)
        
        for challenge in augmented_challenges.values():
            self.assertIn("train", challenge)
            self.assertIn("test", challenge)
            self.assertTrue(isinstance(challenge["train"], list))
            self.assertTrue(isinstance(challenge["test"], list))
        
        for solution in augmented_solutions.values():
            self.assertTrue(isinstance(solution, list))
            self.assertTrue(isinstance(solution[0], list))

if __name__ == '__main__':
    unittest.main()    
