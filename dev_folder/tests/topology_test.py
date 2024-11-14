import unittest
from src.utils.topology_helper import *
import time

class TestColorCharts(unittest.TestCase):
    def test_simple_case(self):
        input_data = [3, 3,
                      1, 1, 2,
                      1, 2, 2,
                      1, 3, 3]
        expected_output = [0, 0, 1,
                           0, 1, 1,
                           0, 2, 2]
        self.assertEqual(color_charts(input_data), expected_output)

    def test_simple_case_with_corner_connection(self):
        input_data = [4, 3,
                      0, 0, 0,
                      0, 4, 0,
                      4, 0, 4,
                      0, 0, 0]
        expected_output = [ 
                            0, 0, 0,
                            0, 1, 0,
                            1, 0, 1,
                            0, 0, 0]
        self.assertEqual(color_charts(input_data), expected_output)

    def test_single_color(self):
        input_data = [2, 2, 1, 1, 1, 1]
        expected_output = [0, 0, 0, 0]
        self.assertEqual(color_charts(input_data), expected_output)

    def test_all_different_colors(self):
        input_data = [2, 2, 1, 2, 3, 4]
        expected_output = [0, 1, 2, 3]
        self.assertEqual(color_charts(input_data), expected_output)

    def test_larger_grid(self):
        input_data = [4, 4, 
                      1, 1, 2, 2,
                      1, 1, 2, 2,
                      3, 3, 4, 4,
                      3, 3, 4, 4]
        expected_output = [0, 0, 1, 1,
                           0, 0, 1, 1,
                           2, 2, 3, 3,
                           2, 2, 3, 3]
        self.assertEqual(color_charts(input_data), expected_output)

    def test_complex_case(self):
        input_data = [5, 5,
                      1, 1, 2, 3, 3,
                      1, 2, 2, 2, 3,
                      4, 4, 2, 5, 5,
                      4, 4, 5, 5, 5,
                      6, 6, 6, 6, 6]
        expected_output = [4, 4, 0, 5, 5,
                           4, 0, 0, 0, 5,
                           3, 3, 0, 1, 1,
                           3, 3, 1, 1, 1,
                           2, 2, 2, 2, 2]
        self.assertEqual(color_charts(input_data), expected_output)

    def test_synthetic_case_0(self):
        input_data = [4, 4,
                      4, 4, 7, 6,
                      4, 4, 7, 6,
                      4, 4, 7, 6,
                      4, 4, 7, 6]
        output = color_charts(input_data)
        self.assertEqual(output[0], output[1])

    def test_get_neighbor_charts(self):
        chart_grid = [
            [0, 0, 1, 2, 2],
            [0, 1, 1, 1, 2],
            [3, 3, 1, 4, 4],
            [3, 3, 4, 4, 4],
            [5, 5, 5, 5, 5]
        ]
        
        # Test case 1: Center of the grid
        neighbors = get_neighbor_charts(chart_grid, 2, 2)
        self.assertEqual(neighbors, {0, 1, 2, 3, 4, 5})
        
        # Test case 2: Corner of the grid
        neighbors = get_neighbor_charts(chart_grid, 0, 0)
        self.assertEqual(neighbors, {0, 1, 2, 3, 4, 5})
        
        # Test case 3: Edge of the grid
        neighbors = get_neighbor_charts(chart_grid, 0, 2)
        self.assertEqual(neighbors, {0, 1, 2, 3, 4, 5})
        
        # Test case 4: With hop=1
        neighbors = get_neighbor_charts(chart_grid, 2, 2, hop=1)
        self.assertEqual(neighbors, {1, 3, 4})
        
        # Test case 5: With hop=2
        neighbors = get_neighbor_charts(chart_grid, 2, 2, hop=2)
        self.assertEqual(neighbors, {0, 1, 2, 3, 4, 5})


    def test_infinite_loop(self):
        chart_grid = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 4, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 6, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        start_time = time.time()
        neighbors = get_neighbor_charts(chart_grid, 2, 2)
        end_time = time.time()
        
        # Assert that the function returns within a reasonable time (e.g., 1 second)
        self.assertLess(end_time - start_time, 0.01, "Function took too long to execute")


    def test_arc_case(self):
        input_data = [29, 29, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 6, 2, 2, 6, 2, 2, 6, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 2, 2, 6, 2, 2, 6, 2, 2, 6, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 2, 2, 7, 2, 2, 7, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 2, 2, 7, 2, 2, 7, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0]

        start_time = time.time()
        output = color_charts(input_data)
        end_time = time.time()
        
        # Assert that the function returns within a reasonable time (e.g., 1 second)
        self.assertLess(end_time - start_time, 0.01, "Function took too long to execute")


if __name__ == '__main__':
    unittest.main()