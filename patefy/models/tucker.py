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
        if self.err is None:
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

    def track_unique_paths(self):
        R = self.R
        C = self.C
        norm = np.sum(C)
        
        pathMap = dict()
        for Ri in np.ndindex( tuple(R) ):
            if C[Ri]/norm > 10e-3 :
                pathMap[ Ri[1:] ] = pathMap.get(Ri[1:], [])+ [ Ri ]
            else: 
                C[Ri] = 0
        
        self.uniquePaths = []
        for key in pathMap:
            if len(pathMap[key]) == 1 :
                self.uniquePaths.append(pathMap[key][0])
            
    def build_path_distance(self):
        R = self.R
        C = self.C
        N = self.N
        norm = np.sum(C)
        
        i = j = 0
        prodR = np.asarray(R).prod();
        dist = np.zeros( [prodR, prodR] )
        
        position = list()
        for id_i in np.ndindex( tuple(R) ):
            position.append( list(id_i) )
            for id_j in np.ndindex( tuple(R) ):
                for m in range(N):
                    slc1 = [ slice(None,None,None) ]*N
                    slc1[ m ] = id_i[m]
                    slc2 = [ slice(None,None,None) ]*N
                    slc2[ m ] = id_j[m]
                    
                    C1 = C[slc1]
                    C2 = C[slc2]
                    
                    dist[i,j] += np.sum(abs(C1-C2))
                
                dist[i,j] = dist[i,j]/(N*norm)
                j += 1
            i += 1; j = 0
            
        self.pathDistances = dist
        model = TSNE(n_components=2, random_state=42, metric='precomputed')
        np.set_printoptions(suppress=True)
        proj = model.fit_transform(dist)
        
        pathProj = [ ( position[i] , proj[i] ) for i in range(len(proj)) ]
        
        self.pathProjection = pathProj

