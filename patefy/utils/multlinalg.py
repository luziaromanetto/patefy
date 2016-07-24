import numpy as np
import math

class TensorIterator:	
	def __init__(self, I):
		self.I = I
		self.N = len(I)
		self.idx = None
	
	def __iter__(self):
		return self
	
	def next(self):
		if self.idx is None :
			self.idx = np.zeros(self.N, dtype=int)
			return tuple(self.idx)
		
		for i in range(self.N-1,-1,-1) :
			if( self.idx[i] < (self.I[i]-1) ) :
				self.idx[i] +=1
				return tuple(self.idx)
			else :
				self.idx[i] = 0
		if( np.sum(self.idx) > 0 ):
			return tuple(self.idx)
		else :
			raise StopIteration()

def outer( vects ):
	N = len(vects)
	
	I = [len(v) for v in vects ]
	T = np.zeros(I, dtype=float)
	
	for i in TensorIterator(I):
		value = 1
		for n in range(N):
			value *= vects[n][i[n]]
		
		T[ i ] = value
	return T

def inner(X, Y):
	# TODO: check the dimension match
	return np.sum(np.multiply(X,Y))

def norm(X):
	# Frobenius norm for nway
	result = np.sum(np.multiply(X,X))	
	return math.sqrt(result)

def tucker_operator(core, facts):
	# TODO: check the dimension match
	R = core.shape
	N = len(R)
	I = [ facts[n].shape[0] for n in range(N) ]
	
	T = np.zeros(I, dtype=float)
	for r in TensorIterator(R):
		vects = [ facts[n][:,r[n]] for n in range(N) ]
		T += core[r]*outer(vects)
		
	return T
	
def kruskal_operator(sigma, facts):
	# TODO: check the dimension match
	N = len(facts)
	R = facts[0].shape[1]
	I = [ facts[n].shape[0] for n in range(N) ]
	
	if sigma is None:
		sigma = np.ones(R, dtype=float)
	
	T = np.zeros(I, dtype=float)
	for r in range(R):
		vects = [ facts[n][:,r] for n in range(N) ]
		T += sigma[r]*outer(vects)
		
	return T

def hadamard(A, B):
	# TODO: check the dimension match
	return np.multiply(A, B)
	
# -------------------------------------------------------------------- #
# TODO:
def kronecker(A, B):
	pass
	
def khatri_rao(A, B):
	pass
	

