import unittest
import json

import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../working'))
sys.path.insert(0, project_root)

from ensemble import merge_all_with_sample

class TestMergeAllWithSample(unittest.TestCase):
    def setUp(self):
        # Create sample test data
        self.test_data = {
            "task1": {
                "test": [{}, {}]  # Two test cases
            },
            "task2": {
                "test": [{}]  # One test case
            },
            "task3": {
                "test": [{}, {}, {}]
            },
            "task4": {
                "test": [{}, {}, {}]
            }
        }
        
        # Create temporary test data file
        with open('test_data.json', 'w') as f:
            json.dump(self.test_data, f)
            
        # Sample submissions
        self.transformer_submission = {
            "task1": [
                {"attempt_1": [[1]], "attempt_2": [[2]]},
                {"attempt_1": [[3]], "attempt_2": [[4]]}
            ],
            "task2": [
                {"attempt_1": [[5]], "attempt_2": [[6]]}
            ],
            "task3": [
                {},
                {"attempt_1": [[305]], "attempt_2": [[306]]},
                {},
            ],
            "task4": [
                {"attempt_1": [[401]], "attempt_2": [[402]]},
                {"attempt_1": [[405]]},
                {"attempt_1": [[403]]},
            ]
        }
        
        self.soma_submission = {
            "task1": [
                {"attempt_1": [[7]], "attempt_2": [[8]]},
                {"attempt_1": [[9]], "attempt_2": [[10]]}
            ],
            "task2": [
                {"attempt_1": [[11]], "attempt_2": [[12]]}
            ],
            "task3": [
                {"attempt_1": [[311]], "attempt_2": [[312]]},
                {},
                {},
            ],
            "task4": [
                {"attempt_1": [[407]], "attempt_2": [[408]]},
                {"attempt_1": [[411]]},
                {},
            ]
        }
        
        self.ice_submission = {
            "task1": [
                {"attempt_1": [[13]], "attempt_2": [[14]]},
                {"attempt_1": [[15]], "attempt_2": [[16]]}
            ],
            "task2": [
                {"attempt_1": [[17]], "attempt_2": [[18]]}
            ],
            "task3": [
                {},
                {},
                {"attempt_1": [[317]], "attempt_2": [[318]]},
            ],
            "task4": [
                {"attempt_1": [[413]], "attempt_2": [[414]]},
                {"attempt_1": [[417]]},
                {"attempt_1": [[415]], "attempt_2": [[416]]},
            ]
        }

    def test_basic_merge(self):
        """Test basic merging functionality"""
        result = merge_all_with_sample(
            'test_data.json',
            self.transformer_submission,
            self.soma_submission,
            self.ice_submission
        )
        
        # Check if all tasks are present
        self.assertEqual(set(result.keys()), {"task1", "task2", "task3", "task4"})
        
        # Check if correct number of test cases are present
        self.assertEqual(len(result["task1"]), 2)
        self.assertEqual(len(result["task2"]), 1)
        
        # Check if each test case has attempt_1 and attempt_2
        for task_results in result.values():
            for test_case in task_results:
                self.assertIn("attempt_1", test_case)
                self.assertIn("attempt_2", test_case)

        self.assertEqual(result["task4"][0]['attempt_1'][0][0], 407) # soma highest priority if full
        self.assertEqual(result["task4"][0]['attempt_2'][0][0], 413) # ice seconary priority if full
        self.assertEqual(result["task4"][1]['attempt_1'][0][0], 411) # soma highest priority even single
        self.assertEqual(result["task4"][1]['attempt_2'][0][0], 417) # ice seconary priority even single
        self.assertEqual(result["task4"][2]['attempt_1'][0][0], 415) # the most common case, soma has no answer, use ice first
        self.assertEqual(result["task4"][2]['attempt_2'][0][0], 403) # the most common case, soma has no answer, use transformer second

        print('result', result)

    def test_missing_submissions(self):
        """Test handling of missing submissions"""
        # Create submissions with missing data
        incomplete_transformer = {"task1": [{"attempt_1": [[1]]}]}
        incomplete_soma = {"task2": [{"attempt_1": [[2]]}]}
        incomplete_ice = {"task1": [{"attempt_1": [[3]]}]}
        
        result = merge_all_with_sample(
            'test_data.json',
            incomplete_transformer,
            incomplete_soma,
            incomplete_ice
        )
        
        # Check if default [[0]] is used when no valid attempts are available
        self.assertIn("task2", result)
        self.assertEqual(len(result["task2"]), 1)

    def test_duplicate_attempts(self):
        """Test that duplicate attempts are not added"""
        # Create submissions with duplicate attempts
        transformer_dup = {
            "task1": [
                {"attempt_1": [[1]], "attempt_2": [[1]]},
            ]
        }
        soma_dup = {
            "task1": [
                {"attempt_1": [[1]], "attempt_2": [[2]]},
            ]
        }
        ice_dup = {
            "task1": [
                {"attempt_1": [[1]], "attempt_2": [[3]]},
            ]
        }
        
        result = merge_all_with_sample(
            'test_data.json',
            transformer_dup,
            soma_dup,
            ice_dup
        )
        
        # Check that the first test case doesn't have duplicate attempts
        attempts = result["task1"][0]
        self.assertNotEqual(attempts["attempt_1"], attempts["attempt_2"])

    def tearDown(self):
        # Clean up test data file
        import os
        try:
            os.remove('test_data.json')
        except:
            pass

if __name__ == '__main__':
    unittest.main(); exit(0)

    suite = unittest.TestSuite()
    suite.addTest(TestMergeAllWithSample('test_basic_merge'))

    unittest.TextTestRunner().run(suite)
