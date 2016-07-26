import unittest
import numpy as np
import math

import patefy
import patefy.utils.multlinalg as MLA
import patefy.methods.tucker as TKD

class MLATester(unittest.TestCase):	
	def test_refold_refold_3(self):
		I = tuple([2,3,4,5])
		
		T = np.arange(120).reshape(I)
		
		M = MLA.unfold2(T, 3)
		newT = MLA.refold2(M, 3, I)
		
	
	def test_unfold_refold_1(self):
		I = tuple([2,3,4,5])
		
		T = np.arange(120).reshape(I)
		
		M = MLA.unfold2(T, 1)
		newT = MLA.refold2(M, 1, I)
		
		dif = MLA.norm( T-newT )
		assert dif < 10e-6
		
	def test_unfold_refold_0(self):
		I = tuple([2,3,4,5])
		T = np.arange(120).reshape(I);
	
		M = MLA.unfold2(T, 0)
		newT2 = MLA.refold2(M, 0, I)
		
		dif = MLA.norm( T - newT2 )
		
		assert dif < 10e-6

	def test_unfold_refold_2(self):
		I = tuple([2,3,4,5])
		T = np.arange(120).reshape(I);
	
		M = MLA.unfold2(T, 2)
		newT2 = MLA.refold2(M, 2, I)
		
		dif = MLA.norm( T - newT2 )
		
		assert dif < 10e-6
		
	def test_swap(self):
		T = np.arange(120).reshape([2,3,4,5])
	
		newT = MLA.swap( T, [1, 0, 3, 2] )
	
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
		
	def test_khatri_rao(self):
		A = np.ones( [5, 4], dtype=float)
		B = np.ones( [10, 4], dtype=float)
		
		C = MLA.khatri_rao(A, B)
		
		assert np.sum(C) == 200
						
if __name__ == "__main__":
	unittest.main()
