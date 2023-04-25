import numpy as np
import scipy as sp
import scipy.sparse as sps
from scipy.sparse import find, identity, coo_matrix
import graph_tool as gt
from graph_tool import centrality, inference
import graph_tool.correlations as gtcorr
import graph_tool.clustering as gtclust


def leading_eigenv_approx(A, max_iter=1000, tol=1e-6, cval=None):
    # Initialize the starting vector x as a random vector
    A=A*0.85
    n = A.shape[0]
    x = np.random.rand(n)
    if cval==None:
    	cval = (1-0.85)/(n)
    # Iterate until convergence
    for i in range(max_iter):
        # Compute y = (A + C) x
        y = A@x + np.ones(n)*x.sum()*cval

        # Normalize y to obtain the new vector x
        x_new = y / np.linalg.norm(y)

        # Compute the change in x and check for convergence
        delta_x = np.linalg.norm(x_new - x)
        if delta_x < tol:
            #print("Reached convergence")
            break

        # Update x for the next iteration
        x = x_new

    # Compute the dominant eigenvalue and eigenvector
    eigenvalue = x.T@A@x + (np.ones(x.shape)*x.sum()*cval).T @ x
    eigenvector = x
    
    return [eigenvalue, eigenvector]

def katz_eigenvalue_approx(A, alpha=0, max_iter=1000, tol=1e-6):
    _, k1 = leading_eigenv_approx(A, cval=0)
    if alpha==0:
        alpha = 1/k1
        alpha = alpha-(0.00001*alpha)
    else:
        if alpha>1/k1:
            print("Warning: alpha>1/k1")
    
    x = np.random.randn(A.shape[0])
    for i in range(max_iter):
        xnew = alpha* A@x + np.ones(A.shape[0])
        # Compute the change in x and check for convergence
        delta_x = np.linalg.norm(x_new - x)
        if delta_x < tol:
            print("Reached convergence")
            break
        # Update x for the next iteration
        x = x_new
    eigenvalue = x.T@A@x + (np.ones(x.shape)*x.sum()*cval).T @ x
    eigenvector = x
    
    return [eigenvalue, eigenvector]