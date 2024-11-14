# test_checkpoint_handler.py

import unittest
from src.checkpoint_handler import CheckpointHandler

class TestCheckpointHandler(unittest.TestCase):

    def setUp(self):
        self.handler = CheckpointHandler("dummy_dir")

    def test_quadratic_checkpoints(self):
        expected_checkpoints = [0, 50, 150, 350, 650, 1050, 1550, 2150, 2850]
        
        for epoch in range(3000):
            if epoch in expected_checkpoints:
                self.assertTrue(self.handler.is_quadratic_checkpoint(epoch), f"Expected checkpoint at epoch {epoch}")
            else:
                self.assertFalse(self.handler.is_quadratic_checkpoint(epoch), f"Unexpected checkpoint at epoch {epoch}")

    def test_non_checkpoints(self):
        non_checkpoints = [49, 51, 149, 151, 299, 301, 499, 501]
        for epoch in non_checkpoints:
            self.assertFalse(self.handler.is_quadratic_checkpoint(epoch), f"Unexpected checkpoint at epoch {epoch}")

    # def test_quadratic_checkpoints(self):        
    #     for epoch in range(3000):
    #         if self.handler.is_quadratic_checkpoint(epoch):
    #             print(f"{epoch} {self.handler.is_quadratic_checkpoint(epoch)}")

if __name__ == '__main__':
    unittest.main()