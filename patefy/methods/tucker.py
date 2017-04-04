import nimfa
import math

import numpy as np
from sklearn.decomposition import TruncatedSVD

import patefy
import patefy.utils.multlinalg as MLA
from patefy.models.tucker import TKD

from numpy import linalg as LA

import warnings

class NTD(TKD):
    
    def __call__(self):
        N = self.N
        I = list(self.I)
        R = list(self.R)
        X = self.T
        
        # Initialization
        epsilon = 10e-18
        B = list()
        dB = list()
        C = np.random.rand(*R)*100
        dC = np.ones(list(R))*epsilon
        for n in range(N):
            B = B + [np.random.rand(I[n], R[n])]
            dB = dB + [np.ones([I[n], R[n]])*epsilon]
        
        # Multiplicative algorithm
        for k in range(100):
            Bn = list()
            # incrementing factors
            for n in range(N):
                order = [m for m in range(N) if m!=n]
                C_n = MLA.unfold(C, n)
                
                BtB_order = [np.dot(B[m].transpose(),B[m]) for m in order]
                
                SAA = MLA.tucker_operator( C, BtB_order, order)
                SAA_n = MLA.unfold(SAA, n)
                SAAS_n = np.dot(SAA_n, C_n.transpose())
                ASS_n = np.dot(B[n], SAAS_n)
                
                Bt_order = [B[m].transpose() for m in order]
                XS_n = MLA.tucker_operator( X, Bt_order, order)
                XS_n = np.dot(MLA.unfold(XS_n, n), C_n.transpose() )
                
                Bn = Bn + [B[n]*(XS_n/(ASS_n+dB[n]))]
            
            # increment core tensor
            XAA = MLA.tucker_operator(X, [b.transpose() for b in B])
            SAA = MLA.tucker_operator(C, [np.dot(b.transpose(),b) for b in B])
            
            Cn = C*XAA/(SAA+dC)
            
            B = Bn
            C = Cn
            
            self.B = Bn
            self.C = Cn
            
            print self.error()
            
        print self.C
            

class ALTNTD(TKD):

    def __call__(self):
        TCn = self.T
        B = list()
        In = list(self.I)

        for n in range(self.N):
            Rn = self.R[n]
            Cn = MLA.unfold(TCn,n)
            In[n] = int(Rn)

            # Inicialization options random_vcol, random, random_c, nndsvd
            nmf = nimfa.Nmf(Cn, seed="random", max_iter=600, rank=Rn, update='euclidean', objective='fro')
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
        self.err = None

class ALSNTD(TKD):

    def __call__(self, kmax = 10):
        T = self.T
        I = self.I
        R = self.R
        B = list()
        N = len(I)
        
        # ---------------  Inicialization of B and C   --------------- #
        for n in range(N):
            Cn = MLA.unfold(T, n)
            nmf = nimfa.Nmf(Cn, seed="random", max_iter=200, rank=R[n], update='euclidean', objective='fro')
            nmf_fit = nmf()
            B.append(nmf_fit.basis())

            for k in range(R[n]): 
                bk = B[n][:,k]
                nBk = LA.norm(bk, 1)
                B[n][:,k] = B[n][:,k]/nBk

        G = MLA.tucker_operator( T,[ LA.pinv(b) for b in B ] )
        G[ G < 0 ] = 0
        G0 = G
        # --------------- Run de alternating algorithm --------------- #
        for k in range(kmax):
            order = range(1,N)

            B_order = [ B[i] for i in order ]
            for n in range(N):
                X = MLA.tucker_operator( G, B_order, order)
                init_H = MLA.unfold(X, n)
                In = B[n].shape
                init_W = B[n]

                Cn = MLA.unfold(T, n)

                nmf = nimfa.Nmf(Cn, seed="fixed", W=np.asarray(init_W), H=np.asarray(init_H), max_iter=200, rank=R[n], update='euclidean', objective='fro')
                nmf_fit = nmf()

                B[n] = nmf_fit.basis()

                for k in range(R[n]): 
                    bk = B[n][:,k]
                    nBk = LA.norm(bk, 1)

                    if nBk == float('Inf'):
                        warnings.warn("Metodo divergiu.\n")
                        self.err = float('Inf')
                        return
                    elif nBk > 1e-8:
                        B[n][:,k] = B[n][:,k]/nBk

                if n < N-1 :
                    order[n]=n
                    B_order[n] = B[n]

            G = MLA.tucker_operator( T,[ LA.pinv(b) for b in B ] )
            G[ G < 0 ] = 0

            G0 = G
            if LA.norm( G - G0 )/LA.norm(G) < 10e-8:
                break

        self.C = G
        self.B = B
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
            Y = np.transpose(Y)
            svd.fit(Y)
            B.append(np.transpose(svd.components_))

        G = MLA.tucker_operator( T, [ np.transpose(b) for b in B] )
        self.C = G
        self.B = B
        self.err = None
