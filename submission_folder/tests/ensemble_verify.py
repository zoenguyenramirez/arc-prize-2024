import unittest

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../working'))
sys.path.insert(0, project_root)

from ensemble import decision_tree_drop

class TestDecisionTreeDrop(unittest.TestCase):
    def test_decision_tree_4_items(self):
        # Create sample attempts list
        attempts = [
            [[1, 1]],  # attempt 1
            [[2, 2]],  # attempt 2
            [[3, 3]],  # attempt 3
            [[4, 4]]   # attempt 4
        ]
        parameters = [1, 1]  # parameters used in the original code

        self.assertEqual(len(decision_tree_drop(1, attempts.copy(), parameters)), 4)
        self.assertEqual(len(decision_tree_drop(51, attempts.copy(), parameters)), 4)
        self.assertEqual(len(decision_tree_drop(3, attempts.copy(), parameters)), 3)
        self.assertEqual(len(decision_tree_drop(54, attempts.copy(), parameters)), 3)
        self.assertEqual(len(decision_tree_drop(6, attempts.copy(), parameters)), 3)
            

    def test_decision_tree_2_items(self):
        # Create sample attempts list
        attempts = [
            [[1, 1]],  # attempt 1
            [[2, 2]],  # attempt 2
        ]
        parameters = [1, 1]  # parameters used in the original code

        self.assertEqual(len(decision_tree_drop(1, attempts.copy(), parameters)), 2)
        self.assertEqual(len(decision_tree_drop(51, attempts.copy(), parameters)), 2)
        self.assertEqual(len(decision_tree_drop(3, attempts.copy(), parameters)), 1)
        self.assertEqual(len(decision_tree_drop(54, attempts.copy(), parameters)), 1)
        self.assertEqual(len(decision_tree_drop(6, attempts.copy(), parameters)), 1)

    def test_decision_tree_0_items(self):
        # Create sample attempts list
        attempts = [
            [[1, 1]],  # attempt 1
        ]
        parameters = [1, 1]  # parameters used in the original code

        # Test for numbers 0 to 99
        for n in range(100):
            result = decision_tree_drop(n, attempts.copy(), parameters)
            self.assertEqual(len(result), 1)

    def test_decision_tree_1_items(self):
        # Create sample attempts list
        attempts = [
        ]
        parameters = [1, 1]  # parameters used in the original code

        # Test for numbers 0 to 99
        for n in range(100):
            result = decision_tree_drop(n, attempts.copy(), parameters)
            self.assertEqual(len(result), 0)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)