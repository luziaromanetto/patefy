import numpy as np
from numpy import linalg as LA
import math

# TODO : mudar a identacao de tab para espacos

def outer( vects ):
	N = len(vects)
	
	I = tuple([len(v) for v in vects ])
	T = np.zeros(I, dtype=float)
	
	for i in np.ndindex(I):
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

def tucker_operator2(core, facts):
	# TODO: check the dimension match
	R = tuple(core.shape)
	N = len(R)
	I = [ facts[n].shape[0] for n in range(N) ]
	
	T = np.zeros(I, dtype=float)
	for r in np.ndindex(R):
		#print r
		vects = [ facts[n][:,r[n]] for n in range(N) ]
		T += core[r]*outer(vects)
		
	return T

def tucker_operator(core, facts, order = None):
	# TODO: check the dimension match
	R = core.shape
	N = len(R)
	
	if order is None:
		order = range(N)			
			
	if len(facts) != len(order):
		raise ValueError("Invalid number of factors")
			
	for n in range(len(order)):
		if facts[n].shape[1] != R[order[n]] :
			raise ValueError("Invalid component number in factor "+str(n)+" - ("+str(facts[n].shape[1])+","+str(R[order[n]])+")")
	
	Tn = core
	Rn = list(R)
	
	for n in range(len(order)):
		Bn = facts[n]
		Cn = unfold(Tn, order[n])
		
		Mn = np.dot(Bn, Cn)
		Rn[order[n]] = facts[n].shape[0]
		
		Tn = refold(Mn, order[n], tuple(Rn))
			
	return Tn
	
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
	N = len(I)
	
	cmodes = [ v for v in range(N) if v != mod ]
	order = np.concatenate(([mod],cmodes))
	
	newT = swap(T, order)
	
	return newT.reshape( [I[mod] , np.asarray([I[m] for m in cmodes ]).prod() ])
	
def refold(C, mod, I):
	N = len(I)
	newC = C.copy()
	
	cmodes = [ i for i in range(N) if i != mod ] 
	order = list(np.concatenate(([mod],cmodes)))
	
	In = [ I[o] for o in order ]
	newT = np.reshape( np.asarray(newC), In )
	neworder = [ order.index(i) for i in range(N) ]	
	
	if mod != 0 :
		newT = swap(newT, neworder)
		
	return newT
	
def swap(T, order):
	I = T.shape
	N = len(I)
	
	if( len(order) != N ):
		raise ValueError("Invalid swap order")
	
	if( type(order) == list ):
		order = np.asarray(order)
		
	neworder = list(np.arange(N))
	newT = T.copy()
	
	
	for i in range(N-1):
		idx = neworder.index(order[i])
		
		newT = newT.swapaxes(i,idx)
		
		temp = neworder[i];
		neworder[i] = neworder[idx];
		neworder[idx] = temp;
	
	return newT

def refold_old(C, mod, I):
	order = len(I)
	T = np.zeros(I, dtype='float64')

	for idx in np.ndindex(I):
		i = idx[mod]
		j = getJ( idx, mod, I)
		T[idx] = C[i,j]
		
	return T
	
def unfold_old(T, mod):
	I = T.shape
	order = len(I)

	N = I[mod];
	M = 1;

	for i in range(order):
		if not i==mod:
			M = M*I[i];
			
	C = np.zeros( (N,M) , dtype='float64')
	for idx in np.ndindex(I):
		i = idx[mod]
		j = getJ(idx, mod, I)
		
		C[i,j]=T[idx]
		
	return C
	
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
