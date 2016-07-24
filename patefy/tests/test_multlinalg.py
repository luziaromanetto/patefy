import unittest
import numpy as np
import math

import patefy
import patefy.utils.multlinalg as MLA

class MLATester(unittest.TestCase):
	def test_inner_zeros(self):
		I = [2, 3, 4]
		
		A = np.zeros(I, dtype=float)
		B = np.zeros(I, dtype=float)
		
		assert MLA.inner(A,B) == 0.0
		
	def test_inner_ones(self):
		I = [2, 3, 4]
		
		A = np.ones(I, dtype=float)
		B = np.ones(I, dtype=float)
		
		assert MLA.inner(A,B) == 24.0
	
	def test_norm(self):
		I = [2, 3, 4]
		
		A = np.zeros(I, dtype=float)
		B = np.ones(I, dtype=float)
		
		assert MLA.norm(A) == 0.0
		assert MLA.norm(B) == math.sqrt(24.0)
		
	def test_tucker_operator_zeros(self):
		I = [10, 10, 10]
		R = [2, 3, 4]
		
		A = np.zeros( [I[0], R[0]], dtype=float)
		B = np.zeros( [I[1], R[1]], dtype=float)
		C = np.zeros( [I[2], R[2]], dtype=float)
		
		G = np.ones(R, dtype=float)
		
		assert np.sum(MLA.tucker_operator(G, [A, B, C])) == 0

	def test_tucker_operator_ones(self):
		I = [10, 10, 10]
		R = [2, 3, 4]
		
		A = np.ones( [I[0], R[0]], dtype=float)
		B = np.ones( [I[1], R[1]], dtype=float)
		C = np.ones( [I[2], R[2]], dtype=float)
		
		G = np.ones(R, dtype=float)
		
		assert np.sum(MLA.tucker_operator(G, [A, B, C])) == 24000
		
	def test_tucker_operator_rand(self):
		I = [10, 10, 10]
		R = [2, 3, 4]
		
		A = np.ones( [I[0], R[0]], dtype=float)
		B = np.ones( [I[1], R[1]], dtype=float)
		C = np.ones( [I[2], R[2]], dtype=float)
		
		G = np.random.rand( *R )
		
		assert abs( np.sum(MLA.tucker_operator(G, [A, B, C])) - np.sum(G)*1000 ) < 10e-8 
		
	def test_kruskal_operator_zeros(self):
		I = [10, 10, 10]
		R = 5
		
		A = np.ones( [I[0], R], dtype=float)
		B = np.ones( [I[1], R], dtype=float)
		C = np.ones( [I[2], R], dtype=float)
		
		G = np.zeros(R, dtype=float)
		
		assert np.sum(MLA.kruskal_operator(G, [A, B, C])) == 0

	def test_kruskal_operator_ones(self):
		I = [10, 10, 10]
		R = 5
		
		A = np.ones( [I[0], R], dtype=float)
		B = np.ones( [I[1], R], dtype=float)
		C = np.ones( [I[2], R], dtype=float)
		
		G = np.ones(R, dtype=float)
		
		assert np.sum(MLA.kruskal_operator(G, [A, B, C])) == 5000

	def test_kruskal_operator_rand(self):
		I = [10, 10, 10]
		R = 5
		
		A = np.ones( [I[0], R], dtype=float)
		B = np.ones( [I[1], R], dtype=float)
		C = np.ones( [I[2], R], dtype=float)
		
		G = np.random.rand(R)
		
		assert abs( np.sum(MLA.kruskal_operator(G, [A, B, C])) - np.sum(G)*1000) < 10e-8
						
if __name__ == "__main__":
	unittest.main()
