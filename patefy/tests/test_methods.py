import unittest
import numpy as np
import math

import patefy
import patefy.methods.tucker as TKD

class MLATester(unittest.TestCase):
	def test_tucker(self):
		I = [10, 50, 100]
		R = [2, 3, 4]
		constrB = [0, 0, 0]
		
		T = np.random.rand( *I)
		fit = TKD.Alttucker(T, R, constrB)
		fit()
		
		print "Erro :", fit.error()
		
if __name__ == "__main__":
	unittest.main()
