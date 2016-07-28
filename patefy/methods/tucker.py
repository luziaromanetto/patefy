import nimfa
import math

import numpy as np
from sklearn.decomposition import TruncatedSVD

import patefy
import patefy.utils.multlinalg as MLA
from patefy.models.tucker import TKD

from numpy import linalg as LA
	
class Alttucker(TKD):
		
	def __call__(self):
		TCn = self.T
		B = list()
		In = list(self.I)
		
		for n in range(self.N):
			constr = self.constrB[n]
			Rn = self.R[n]
			
			print("modo "+str(n))
			if constr >= 0 :
				print("Constr == "+str(constr)+" ; R_i == "+str(Rn)+"\n")
				print(".")
				Cn = MLA.unfold(TCn,n)
				print(".")
				if constr == 0:
					# Nonnegativity constraint
					In[n] = int(Rn)
					
					# Inicialization options random_vcol, random, random_c, nnsvd
					nmf = nimfa.Nmf(Cn, seed="random_vcol", max_iter=200, rank=Rn, update='euclidean', objective='fro')
					nmf_fit = nmf()

					Bi = nmf_fit.basis()
					H = nmf_fit.coef()

					for k in range(Rn): 
						bk = Bi[:,k]
						hk = H[k,:]

						nBk = LA.norm(bk, 1)
						Bi[:,k] = Bi[:,k]/nBk
						H[k,:] = H[k,:]*nBk

					B.append(Bi)
					print "."
					TCn = MLA.refold(H, n, tuple(In))

		self.B = B
		self.C = TCn
		self.err = None

class HOOI(TKD):
	
	def __call__(self, kmax = 10):
		T = self.T
		B = list()
		I = self.I
		R = self.R
		N = len(I)

		# Inicialization
		G = np.random.rand( *R )
		B = [ np.random.rand( *[I[n], R[n]] ) for n in range(N) ]
		
		for k in range(kmax):
			order = range(1,N)
			B_order = [ np.transpose(B[i]) for i in order ]
			for n in range(N):
				svd = TruncatedSVD(n_components=R[n], random_state=42)				
				
				X = MLA.tucker_operator( T, B_order, order )
				Y = MLA.unfold(X, n)
			
				svd.fit(np.transpose(Y))
				
				B[n] = np.transpose(svd.components_)
				if n < N-1 :
					order[n]=n
					B_order[n] = np.transpose(B[n])
				
			G = MLA.tucker_operator( T, [ np.transpose(b) for b in B] )
			
		self.C = G
		self.B = B
		self.err = None		

class HOSVD(TKD):
	# Original Tucker method
	def __call__(self):		
		T = self.T
		B = list()
		I = self.I
		R = self.R
		N = len(I)

		for n in range(N):			
			Y = MLA.unfold(T, n)
			
			svd = TruncatedSVD(n_components=R[n], random_state=42)
			svd.fit(np.transpose(Y))
			B.append(np.transpose(svd.components_))

		G = MLA.tucker_operator( T, [ np.transpose(b) for b in B] )
		self.C = G
		self.B = B
		self.err = None
