import unittest
import numpy as np
import math

import patefy
import patefy.methods.tucker as TKD

class MLATester(unittest.TestCase):
    def test_ALSNTD(self):
        I = [10, 50, 100]
        R = [5, 3, 4]

        T = np.random.rand( *I )
        fit = TKD.ALSNTD(T, R)
        fit( 100 )

        print "Erro ALSNTD:", fit.error()

    def test_ALTNTD(self):
        I = [10, 50, 100]
        R = [2, 3, 4]

        T = np.random.rand( *I )
        fit = TKD.ALTNTD(T, R)
        fit()

        print "Erro ALTNTD:", fit.error()

    def test_HOOI(self):
        I = [10, 50, 100]
        R = [2, 3, 4]

        T = np.random.rand(*I)
        fit = TKD.HOOI(T, R)
        fit( 100 )
        
        print "Erro HOOI :", fit.error()

    def test_HOSVD(self):
        I = [10, 50, 100]
        R = [2, 3, 4]

        T = np.random.rand(*I)
        fit = TKD.HOSVD(T, R)
        fit()

        print "Erro HOSVD :", fit.error()
        
if __name__ == "__main__":
    unittest.main()
