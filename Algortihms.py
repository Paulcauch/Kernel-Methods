#### MAIN ALGOS


# IMPORT 

import numpy as np 
from scipy.optimize import fmin_l_bfgs_b 
from cvxopt import matrix, solvers
import pickle as pkl
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve



# UTILS 

#######################################
#######################################
############# KERNEL PCA ##############
#######################################
#######################################



class KernelPCA:
    
    def __init__(self,kernel, r=2):                             
        self.kernel = kernel         
        self.alpha = None # Matrix of shape N times d representing the d eingenvectors alpha corresp
        self.lmbda = None # Vector of size d representing the top d eingenvalues
        self.support = None 
        self.r =r  # Number of principal components

    def compute_PCA(self, X):
        
        self.support = X
        K=self.kernel(X,X)

        #Center
        U=np.ones(K.shape)
        I=np.eye(K.shape[0])
        K=(I-U)@K@(I-U)
        
        valp,vectp=np.linalg.eigh(K)

        #we take the last r eingenvalues
        self.lmbda=valp[-self.r:]
        inverse=1/np.sqrt(self.lmbda)

        self.alpha = inverse*vectp[:,-self.r:]


        
    def transform(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K=self.kernel(x,self.support)
        return K@self.alpha



#######################################
#######################################
########### NYMSTROM APPRO ############
#######################################
#######################################
    

# Nymstrom approximation via Kernel PCA
    

class NystromKPCA():
    """
    Compute an approximation of kernel with the PCA techniques, 
    taking m data points, projecting on p directions (p<=m)

    """
    def __init__(self,kernel,p,m) :
       self.kernel=kernel
       self.m=m
       self.p=p
       self.alphas=None
       self.big_approximated_kernel=None
       self.big_approximated_repr=None
       self.X_subset=None

    def fit_PCA(self, X):

        set=self.choose_subset(X)

        K=self.kernel(self.X_subset,self.X_subset)

        
        U=np.ones(K.shape)
        I=np.eye(K.shape[0])
        K=(I-U)@K@(I-U)
        
        valp,vectp=np.linalg.eigh(K)

        #we take the last r eingenvalues
        lmbda=valp[-self.p:]
        inverse=1/np.sqrt(lmbda)

        self.alphas=(inverse*vectp[:,-self.p:]).T
    

    def approximated_repr(self,x):
        return self.alphas@self.kernel(self.X_subset,x)

    def appro_kernel(self,X,Y):
        repr_X=self.alphas@self.kernel(self.X_subset,X)
        repr_Y=self.alphas@self.kernel(self.X_subset,Y)
        return repr_X.T@repr_Y

    def choose_subset(self,X):
        self.n=X.shape[0]
        set=np.random.choice(self.n,self.m)
        self.X_subset=X[set]
        return set







# SOLVERS



#######################################
#######################################
########### KERNEL RIDGE ##############
#######################################
#######################################


class KernelRR:
    
    def __init__(self,kernel,lmbda):
        self.lmbda = lmbda                    
        self.kernel = kernel    
        self.alpha = None 
        self.b = None
        self.support = None
        self.type='ridge'
        
    def fit(self, X, y):
        N=len(y)
        self.support = X
        ones=np.ones((N,1))
        K=self.kernel(X,X)
        K_prime=np.block([[K, ones], [ones.T, np.ones((1, 1))]])
        y_prime=np.append(y,[0])

        mat=K_prime+(N/2)*self.lmbda*np.identity(N+1)

        alpha_prime=np.linalg.solve(mat,y_prime)  
        self.alpha = alpha_prime[:-1]  
        self.b=alpha_prime[-1]     
        

    def regression_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        K=self.kernel(x,self.support)
        return K@self.alpha


    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        return self.regression_function(X)+self.b
    


#######################################
#######################################
########## KERNEL LOGISTIC ############
#######################################
#######################################
    

# Kernel Logistic Regression with the optimize package

class KernelLogisticRegression_optimize():
    """
    Class which comput the solution to the Kernel logistic Regression with regularization parameter lambda (lmbda)
    Using the Optimize package
    """

    def __init__(self,lmbda,alpha0):
        self.lmbda=lmbda
        self.alpha0=alpha0

    def obj(self,K,y,alpha):
        n=len(K)
        return np.sum(np.log(1+np.exp(-y*(K@alpha))))/n+self.lmbda*alpha.T@K@alpha/2
    
    def sigmoid(self,u):
        return 1 / (1 + np.exp(-u))

    def logistic(self,u):
        return np.log(1 + np.exp(-u))


    def logistic_prime(self,u):
        return -self.sigmoid(-u)


    def logistic_prime2(self,u):
        return self.sigmoid(u) * self.sigmoid(-u)


    def derivative(self,K,y,alpha):
        n=len(K)
        P=-np.diag(1/(1+np.exp(y*(K@alpha))))
        return K@P@y/n+self.lmbda*K@alpha
    
    def second_derivative(self,K,y,alpha):
       n=len(K)
       W=np.diag(self.logistic_prime2(y*(K@alpha)))
       return (1/n)* K@W@K + self.lmbda*K

    def train(self,K,y):
        ob= lambda alpha : self.obj(K=K,y=y,alpha=alpha)
        der=lambda alpha : self.derivative(K=K,y=y,alpha=alpha)
        secder=lambda alpha : self.second_derivative(K=K,y=y,alpha=alpha)
        
        optRes = optimize.minimize(fun=ob,
                                   x0=np.ones(len(K)), 
                                   method='SLSQP', 
                                   jac=der,hess=secder
                                   ,tol=1e-5)
        self.alpha = optRes.x
    
    def fit(self,K):
        return self.sigmoid(K@self.alpha)


# Kernel Logistic Regression with iteratively reweighted least-square (IRLS)
    
class KernelLogisticRegression:
    """
    Class which comput the solution to the Kernel logistic Regression with regularization parameter lambda (lmbda)
    Using the IRLS
    """

    def __init__(self, kernel, reg_param=0, epsilon=1e-8):
        self.alpha = None
        self.reg_param = reg_param
        self.beta = None
        self.kernel = kernel
        self.eps = epsilon
        self.support=None

    def fit(self, X, y):
        N = X.shape[0]
        self.support=X

        k = self.kernel(X,X)
        alpha = np.zeros(N)
        alpha_old = alpha + np.inf
        sig = np.vectorize(self.sigmoid)
        logpp = np.vectorize(self.logistic_prime2)
        i=0
        while (np.abs(alpha - alpha_old) > self.eps).any():
            # Update coefs
            m = k @ alpha
            W = np.diag(logpp(y * m))
            z = m + y / sig(y * m)

            # Solve Weighted KRR
            sqrt_W = np.sqrt(W)

            alpha_old = alpha
            alpha = sqrt_W @ np.linalg.inv(
                sqrt_W @ k @ sqrt_W + N * self.reg_param * np.eye(N)
            ) @ sqrt_W @ z
            print(f'{i}ème iteration, epsilon :{np.max(np.abs(alpha - alpha_old))}')
            i+=1
            if i == 100 :
                break

        self.alpha = alpha

        return self.sigmoid(np.einsum('i, ij->j', self.alpha, k))

    def predict(self, X):

        K_Xx = self.kernel(X, self.support)
        predictions = self.sigmoid(np.einsum('i, ij->j', self.alpha, K_Xx.T))
        return predictions  


    def logistic(self,u):
        return np.log(1 + np.exp(-u))


    def logistic_prime(self,u):
        return -self.sigmoid(-u)


    def logistic_prime2(self,u):
        return self.sigmoid(u) * self.sigmoid(-u)


    def sigmoid(self,u):
        return 1 / (1 + np.exp(-u))



#######################################
#######################################
############# KERNEL SVM ##############
#######################################
#######################################


# Kernel SVM classifier with the optimize package

class KernelSVC_optimize:

    """
    Class which train SVM models with Kernels methods takes a dataset with labels in {-1,1} 
    Using the optimize package
    """
    
    def __init__(self, C, kernel, epsilon = 1e-1):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon 
        self.norm_f = None
       
    
    def fit(self, X, y):
        N = len(y)
        K=self.kernel(X,X)
        print('kernel computed')
        diag=np.diag(y)

        # Lagrange dual problem
        def loss(alpha):
            return  (1/2)*(diag@alpha).T@K@(diag@alpha)-np.sum(alpha) 

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return diag@K@diag@alpha-np.ones_like(alpha) 



        fun_eq = lambda alpha: alpha.T@y        
        jac_eq = lambda alpha: y   
        fun_ineq = lambda alpha: np.concatenate((alpha,self.C-alpha))     
        jac_ineq = lambda alpha:  np.concatenate((np.identity(len(alpha)),-np.identity(len(alpha)))) 

        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})
        print('begin opti :')
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints,tol=self.epsilon)
        self.alpha = optRes.x
        print('end opti')

        ## Attributes
        indice_sv=np.where(np.abs(self.alpha)>1e-5)[0]
        self.support=X[indice_sv]
        self.alpha_support=self.alpha[indice_sv]
        self.beta=diag@self.alpha
        self.beta_support=self.beta[indice_sv]
        margin_indices = np.where((self.alpha > 1e-5) & (self.alpha < self.C-1e-5))[0]
        self.margin_points = X[margin_indices] 
        self.b = np.mean(y[indice_sv]-(K@self.beta)[indice_sv])  
        self.norm_f = self.beta[indice_sv].T@(K@self.beta)[indice_sv]

 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x,self.support)@self.beta_support
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1
    


# Kernel SVM classifier with the cvxopt package
    

class KernelSVC:
    """
    Class which train SVM models with Kernels methods takes a dataset with labels in {-1,1} 
    Using the cvxopt package
    """
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon 
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        K=self.kernel(X,X)
        diag=np.diag(y)

        P = diag@K@diag
        q = -np.ones(N)

        G1 = -np.identity(N)
        h1 = np.zeros(N)

        # Pour alpha <= C
        G2 = np.identity(N)
        h2 = np.ones(N) * self.C

        # Combiner en matrices G et h pour cvxopt
        G = np.vstack([G1, G2])
        h = np.hstack([h1, h2])
        
        
        self.alpha = solvers.qp(P=matrix(P),q=matrix(q),G=matrix(G),h=matrix(h))['x']
        self.alpha = np.array(self.alpha).reshape(-1)
        
        ## Assign the required attributes

        indice_sv=np.where(np.abs(self.alpha)>1e-4)[0]
        #print(indice_sv)
        #print(np.array(self.alpha).reshape(-1))
        
        self.support=X[indice_sv]
        self.alpha_support=self.alpha[indice_sv]

        self.beta=diag@self.alpha
        self.beta_support=self.beta[indice_sv]

        margin_indices = np.where((self.alpha > 1e-5) & (self.alpha < self.C-1e-5))[0]
        self.margin_points = X[margin_indices]  #'''------------------- A matrix with each row corresponding to a point that falls on the margin ------------------'''
        
        self.b = np.mean(y[indice_sv]-(K@self.beta)[indice_sv])  #''' -----------------offset of the classifier------------------ '''

        self.norm_f = self.beta[indice_sv].T@(K@self.beta)[indice_sv]# '''------------------------RKHS norm of the function f ------------------------------'''


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x,self.support)@self.beta_support
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1



