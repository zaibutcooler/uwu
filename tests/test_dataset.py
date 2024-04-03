
import unittest
import torch
# from uwu import Conversations

import numpy as np

class TestLoadingPretraining(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.data = self.load_data()
        # self.trainset = Conversations(self.data)
        
    def load_data(self):
        return np.array([1])
    
    
    def test_downloading(self):
        data = self.load_data()
        assert data is not None

    def test_dataset(self):
        # TODO make replace with actual data
        # desired_shape = 1
        
        # example = self.trainset[0]
        # assert example is not None
        # assert len(self.trainset[0]) >= desired_shape
        # print("asserting")
        assert 1 > 0
    
if __name__ == '__main__':
    unittest.main()
