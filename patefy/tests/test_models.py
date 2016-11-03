import unittest
import numpy as np
import math

import patefy
import patefy.models.tensor as mdltkd

class MDLTester(unittest.TestCase):
    def test_read_json(self):
        T = mdltkd.METATensor()
        T.read_json('../datasets/json/tensor_data1.json')

        assert T.modesName[0] == "Time"

    def test_build_path_distance(self):
        T = mdltkd.METATensor()
        T.read_json('../datasets/json/tensor_data1.json')

        T.decomposition.build_path_distance()
        
        dist = T.decomposition.pathDistances 
        proj = T.decomposition.pathProjection
        
        T.write_json('../datasets/json/tensor_data2.json')
        assert dist[0][0] == 0
        
    
if __name__ == "__main__":
    unittest.main()

