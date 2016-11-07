import unittest
import numpy as np
import math
import matplotlib.pyplot as plt

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
        ids = T.decomposition.pathProjection[0]
        ploj = T.decomposition.pathProjection[1]
        
        #T.write_json('../datasets/json/tensor_data2.json')
        fig, ax = plt.subplots()
        ax.scatter(ploj[:,0], ploj[:,1])
        
        fig.savefig("proj0.png")
        for i, txt in enumerate(ids):
            ax.annotate(txt, (ploj[i,0], ploj[i,1]), fontsize=8)
        
        fig.savefig("proj1.png")
        assert dist[0][0] == 0
    
if __name__ == "__main__":
    unittest.main()

