import numpy as np
from numpy import linalg as LA
import math

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
    if X.shape != Y.shape:
        raise ValueError("Invalid matrix dimension, can't multiply "+str(X.shape )+" x "+str(Y.shape))
    return np.sum(np.multiply(X,Y))

def norm(X):
    # Frobenius norm for nway
    return LA.norm(X)

def tucker_operator2(core, facts):
    R = tuple(core.shape)
    N = len(R)
    I = [ facts[n].shape[0] for n in range(N) ]
    
    if R != [ facts[n].shape[1] for n in range(N) ]:
        raise ValueError("The dimensions between core and factors don't match")

    T = np.zeros(I, dtype=float)
    for r in np.ndindex(R):
        vects = [ facts[n][:,r[n]] for n in range(N) ]
        T += core[r]*outer(vects)

    return T

def tucker_operator(core, facts, order = None):
    R = core.shape
    N = len(R)

    if order is None:
        order = range(N)

    if len(facts) != len(order):
        raise ValueError("Invalid number of factors", len(facts), len(order))
    
    if [R[n] for n in order] != [ facts[n].shape[1] for n in range(len(order)) ]:
        raise ValueError("The dimensions between core and factors don't match")

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
    N = len(facts)
    R = facts[0].shape[1]
    I = [ facts[n].shape[0] for n in range(N) ]

    if sigma is not None and len(sigma)!=R:
        raise ValueError("Size of sigma and components don't match")

    if [ facts[n].shape[1] for n in range(N) ] != [R]*N :
        raise ValueError("The number of components in factors don't match")

    if sigma is None:
        sigma = np.ones(R, dtype=float)

    T = np.zeros(I, dtype=float)
    for r in range(R):
        vects = [ facts[n][:,r] for n in range(N) ]
        T += sigma[r]*outer(vects)

    return T

def hadamard(A, B):
    if A.shape != B.shape:
        raise ValueError("The matrices have diferent sizes")

    return np.multiply(A, B)

def kron(A, B):
    # Produto de kroniker
    return np.kron(A, B)

def khatri_rao(A, B):
    if A.shape[1]!= B.shape[1]:
        raise ValueError("Number of colunms between matrices don't match")

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
