import numpy as np
from numpy import linalg as LA
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
	return LA.norm(X)

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
	
def kron(A, B):
	# Produto de kroniker
	return np.kron(A, B)

def khatri_rao(A, B):
	# TODO: check the dimension match
	J = A.shape[1]
	
	T = []
	for j in range(J):
		colj = kron( A[:,j], B[:,j] )
		T.append(colj)

	return np.asarray(T).transpose()
	
def unfold(T, mod):
	I = T.shape
	order = len(I)

	N = I[mod];
	M = 1;

	for i in range(order):
		if not i==mod:
			M = M*I[i];
			
	C = np.zeros( (N,M) , dtype='float64')
	for idx in TensorIterator(I):
		i = idx[mod]
		j = getJ(idx, mod, I)
		
		C[i,j]=T[idx]
		
	return C
	
def refold(C, mod, I):
	order = len(I)
	T = np.zeros( I, dtype='float64')

	for idx in TensorIterator(I):
		i = idx[mod]
		j = getJ( idx, mod, I)
		T[idx] = C[i,j]
		
	return T
		
def getJ(idAct, mod, I):
	order = len(I)
	j=0;
	for k in range(order):
		if( k!= mod ):
			Jk = 1;
			for m in range(k):
				if( m != mod):
					Jk = Jk*I[m]
			j=j+idAct[k]*Jk;
	return j
