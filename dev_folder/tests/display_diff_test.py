import unittest
from io import StringIO
import sys
from src.utils.display_diff import compare_sequences, colorize, split_into_chunks

class TestDisplayDiff(unittest.TestCase):

    def setUp(self):
        self.held, sys.stdout = sys.stdout, StringIO()

    def tearDown(self):
        sys.stdout = self.held

    def test_colorize(self):
        self.assertEqual(colorize("test", "red"), "\033[91mtest\033[0m")
        self.assertEqual(colorize("test", "green"), "\033[92mtest\033[0m")
        self.assertEqual(colorize("test", "yellow"), "\033[93mtest\033[0m")

    def test_split_into_chunks(self):
        text = "line1 ROW_SEPARATOR line2 ROW_SEPARATOR line3"
        expected = ["line1", "line2", "line3"]
        self.assertEqual(split_into_chunks(text), expected)

    def test_compare_sequences_no_diff(self):
        expected = "line1 ROW_SEPARATOR line2"
        generated = "line1 ROW_SEPARATOR line2"
        compare_sequences(expected, generated)
        self.assertEqual(sys.stdout.getvalue().strip(), "line1\nline2")

    def test_compare_sequences_with_diff(self):
        expected = "line1 ROW_SEPARATOR line2 ROW_SEPARATOR line3"
        generated = "line1 ROW_SEPARATOR line2_modified ROW_SEPARATOR line4"
        compare_sequences(expected, generated)
        output = sys.stdout.getvalue().strip()
        self.assertIn("line1", output)
        self.assertIn("\033[91mline2\033[0m", output)
        self.assertIn("\033[92mline2_modified\033[0m", output)
        self.assertIn("\033[91mline3\033[0m", output)
        self.assertIn("\033[92mline4\033[0m", output)

    def test_original(self):
        expected = """START START_INPUT 0 0 0 0 0 0 0 0 0 ROW_SEPARATOR 0 0 0 0 0 0 0 0 0 ROW_SEPARATOR 0 0 3 0 0 6 0 0 0 ROW_SEPARATOR 0 0 0 2 2 0 0 0 0 ROW_SEPARATOR 0 0 0 2 2 0 0 0 0 ROW_SEPARATOR 0 0 8 0 0 7 0 0 0 ROW_SEPARATOR 0 0 0 0 0 0 0 0 0 ROW_SEPARATOR 0 0 0 0 0 0 0 0 0 ROW_SEPARATOR 0 0 0 0 0 0 0 0 0 ROW_SEPARATOR END_INPUT"""
        generated = """START START_INPUT 0 0 0 0 0 0 0 0 0 ROW_SEPARATOR 0 0 0 0 0 0 0 0 0 ROW_SEPARATOR 0 0 3 0 0 6 0 0 0 ROW_SEPARATOR 0 0 0 2 2 0 0 0 0 ROW_SEPARATOR 0 0 0 2 2 0 0 0 0 ROW_SEPARATOR 0 0 8 0 0 7 0 0 0 ROW_SEPARATOR 0 0 0 0 0 0 0 0 0 ROW_SEPARATOR 0 0 0 0 0 0 0 0 0 ROW_SEPARATOR 0 0 0 0 0 0 0 0 0 ROW_SEPARATOR END_INPUT"""
        compare_sequences(expected, generated)


if __name__ == '__main__':
    unittest.main()