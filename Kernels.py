# KERNEL


# IMPORT
import numpy as np


# KERNELS


#######################################
#######################################
############# POLYNOMIAL ##############
#######################################
#######################################


class polynomial():
    """
    Compute the matrix of the Polynomial Kernel : 
    
    (K)_ij=K(X_i,Y_j)=(<X_i,Y_j>)^d the Polynomial Kernel of degree d evaluated between the ith data and jth data

    Parameters : 
    X : 2d array size (n,p) 
        the Data matrix with n the number of data and p the size of data
    Y : 2d array size (q,p) 
        the Data matrix with q the number of data and p the size of data
    d : Integer
            Degree of the polynomial kernel

    Outputs :
    2d array size (n,q)
    Matrix of the Gaussian Kernel
    """
    def __init__(self,d=2):
        self.d=d

    def kernel(self,X,Y,intercept=False):
        if not(intercept):
            return np.dot(X,Y.T) ** self.d
        else : 
            X_intercept = np.concatenate((X,np.ones(X.shape[0]).reshape(-1,1)), axis=1)
            Y_intercept= np.concatenate((Y,np.ones(Y.shape[0]).reshape(-1,1)), axis=1)
            return np.sum(X_intercept * Y_intercept[:,None,:], axis=-1) ** self.d




#######################################
#######################################
############## GAUSSIAN ###############
#######################################
#######################################


# initial RBF kernel computation which is not optimal
    
class RBF_so:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        return  np.exp(-np.sum((X[:,None]-Y)**2,axis=-1)/(2*self.sigma**2)) ## Matrix of shape NxM
    


# RBF kernel computation which is optimal
    

class RBF():
    """
    Compute the matrix of the Gaussian Kernel : 
    
    (K)_ij=K(X_i,Y_j)=exp( 1/2*sigma^2 * ||X_i-Y_j||^2 ) the Gaussian Kernel evaluated between the ith data and jth data

    Parameters : 
    X : 2d array size (n,p) 
        the Data matrix with n the number of data and p the size of data
    Y : 2d array size (q,p) 
        the Data matrix with q the number of data and p the size of data
    sigma : float 
             Variance of the GaussianKernel 

    Outputs :
    2d array size (n,q)
    Matrix of the Gaussian Kernel
    """   
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        diff2 = np.sum(X**2, axis=1)[:, None] + np.sum(Y**2, axis=1)[None, :] - 2 * np.dot(X, Y.T)
        return  np.exp(-diff2/(2*self.sigma**2)) ## Matrix of shape NxM