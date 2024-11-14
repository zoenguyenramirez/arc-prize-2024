import unittest

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../working'))
sys.path.insert(0, project_root)

from ensemble import build_top_2_attempts

class TestBuildTop2Attempts(unittest.TestCase):
    def setUp(self):
        # Setup some test data
        self.soma_attempts = [[[1, 1], [1, 1]], [[2, 2], [2, 2]]]
        self.ice_attempts = [[[3, 3], [3, 3]], [[4, 4], [4, 4]]]
        self.transformer_attempts = [[[5, 5], [5, 5]], [[6, 6], [6, 6]]]

    def test_normal_case(self):
        """Test when all three models provide valid attempts"""
        result = build_top_2_attempts(
            soma=self.soma_attempts,
            ice=self.ice_attempts,
            transformer=self.transformer_attempts
        )
        
        # Check that result has correct structure
        self.assertIsInstance(result, dict)
        self.assertIn('attempt_1', result)
        self.assertIn('attempt_2', result)
        
        # Check that attempts are lists
        self.assertIsInstance(result['attempt_1'], list)
        self.assertIsInstance(result['attempt_2'], list)

    def test_empty_inputs(self):
        """Test when empty lists are provided"""
        result = build_top_2_attempts(
            soma=[],
            ice=[],
            transformer=[]
        )
        
        # Should return default values
        self.assertEqual(result['attempt_1'], [[0]])
        self.assertEqual(result['attempt_2'], [[0]])

    def test_partial_inputs(self):
        """Test when some inputs are None or raise exceptions"""
        result = build_top_2_attempts(
            soma=None,
            ice=self.ice_attempts,
            transformer=self.transformer_attempts
        )
        
        # Should still return valid attempts
        self.assertIsInstance(result, dict)
        self.assertIn('attempt_1', result)
        self.assertIn('attempt_2', result)

    def test_scoring_order(self):
        """Test that attempts are properly scored and ordered"""
        # ICE has highest weight (0.3), should appear first
        simple_ice = [[[1, 1]]]
        simple_soma = [[[2, 2]]]
        simple_transformer = [[[3, 3]]]
        
        result = build_top_2_attempts(
            soma=simple_soma,
            ice=simple_ice,
            transformer=simple_transformer
        )
        
        # ICE should be attempt_1 due to highest weight (0.3)
        self.assertEqual(result['attempt_1'], [[1, 1]])
        self.assertEqual(result['attempt_2'], [[3, 3]])

    def test_duplicate_attempts(self):
        """Test handling of duplicate attempts"""
        same_attempt = [[[1, 1]]]
        result = build_top_2_attempts(
            soma=same_attempt,
            ice=same_attempt,
            transformer=same_attempt
        )
        
        # Should combine scores for duplicate attempts
        self.assertEqual(result['attempt_1'], [[1, 1]])
        self.assertEqual(result['attempt_2'], [[0]])

    def test_single_attempt(self):
        """Test when only one model provides attempts"""
        result = build_top_2_attempts(
            soma=self.soma_attempts,
            ice=[],
            transformer=[]
        )
        
        # Should have one valid attempt and one default
        self.assertNotEqual(result['attempt_1'], [[0]])
        self.assertEqual(result['attempt_2'], [[2, 2], [2, 2]])

    def test_complex_scoring_order(self):
        """Test that attempts are properly scored and ordered"""
        # ICE has highest weight (0.3), should appear first
        complex_ice = [[[1, 1]], [[4, 4]], [[3, 3]]]
        complex_soma = [[[2, 2]], [[1, 1]], [[3, 3]]]
        complex_transformer = [[[3, 3]], [[2, 2]], [[1, 1]]]
        
        result = build_top_2_attempts(
            soma=complex_soma,
            ice=complex_ice,
            transformer=complex_transformer
        )
        
        # ICE should be attempt_1 due to highest weight (0.3)
        self.assertEqual(result['attempt_1'], [[1, 1]])
        self.assertEqual(result['attempt_2'], [[3, 3]])

if __name__ == '__main__':
    unittest.main(); exit(0)

    suite = unittest.TestSuite()
    suite.addTest(TestBuildTop2Attempts('test_complex_scoring_order'))

    unittest.TextTestRunner().run(suite)
