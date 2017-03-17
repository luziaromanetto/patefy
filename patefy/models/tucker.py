import numpy as np

import patefy.utils.multlinalg as MLA
import patefy.utils.tsne as tsne
from sklearn.manifold import TSNE

class TKD(object):
    def __init__(self, T=[], facts=[]):
        self.T = T
        self.C = None
        self.B = None
        self.R = facts
        self.I = T.shape if T!=[] else [] 
        self.N = len(T.shape) if T!=[] else 0 
        self.err = None
        self.uniquePaths = None
        self.pathDistances = None
        self.pathProjection = None
        
    def error(self):
        self.err = MLA.norm(self.T - MLA.tucker_operator( self.C, self.B ));
        self.err = self.err/MLA.norm(self.T)
            
        return self.err

    def pivotComponents(self, pivOrder, reverse=False):
        # TODO : considerar os novos campos neste metodo; prod e distancia

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
            for idAct in np.ndindex(tuple(self.R)):
                idActOrd = tuple([ pivOrder[i][idAct[i]] for i in range(len(pivOrder)) ])

                Cp[idAct]=self.C[idActOrd]
                
        self.C = Cp

