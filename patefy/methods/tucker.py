import nimfa
import math

import numpy as np

import patefy
import patefy.utils.multlinalg as MLA

from numpy import linalg as LA

class Alttucker(object):
	
	def __init__(self, T, facts, ConstrF):
		self.T = T
		self.C = None
		self.B = None
		self.R = facts
		self.I = T.shape # talvez mudar I e N para uma superclasse Tensor
		self.N = len(T.shape)
		self.constrB = ConstrF
		
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
					TCn = MLA.refold(H, n, tuple(In))

		self.B = B
		self.C = TCn

	def errornp(self):
		D = self.T - MLA.tucker_operator( self.C, self.B );
		return MLA.norm( D )/MLA.norm(self.T)
	
	def error(self):	
		I = self.I
		R = self.R
		N = self.N
		result = 0
		Tn = 0

		for idAct in MLA.TensorIterator(I):
			Til = 0
			for coreIdAct in MLA.TensorIterator(R):
				cr = self.C[coreIdAct]
				#print idAct
				#print coreIdAct
				for n in range(N):
					cr = cr*(self.B[n][ tuple([idAct[n] ,coreIdAct[n]]) ])
				
				Til = Til + cr;
			
			result += (Til-self.T[idAct])**2
			Tn += (self.T[idAct])**2

		Tn = math.sqrt(Tn)
		return math.sqrt(result)/Tn

	def pivotComponents(self, pivOrder, reverse=False):

		if( reverse == True ):
			pivOrder = [ p.reverse() for p in pivOrder ]

		for i in range(len(pivOrder)):
			Bi = np.array( self.B[i] , dtype='float64')
			Bpi = np.array( self.B[i] , dtype='float64')

			for j in range(len(pivOrder[i])):
				for k in range(len(self.B[i])):
					Bpi[k][j] = Bi[k][pivOrder[i][j]] 

			self.B[i]=Bpi

			Cp = np.zeros( self.R, dtype='float64')
			for idAct in MLA.TensorIterator(self.R):
				idActOrd = tuple([ pivOrder[i][idAct[i]] for i in range(len(pivOrder)) ])

				Cp[idAct]=self.C[idActOrd]
				
		self.C = Cp;
