"""
Packages
"""
import numpy as np
from numpy import linalg as LA
import pickle
import math
import time
import sys
import os
import io
import scipy.io as sio
import pickle

"""
o Author: Bo-Hong (Jay) Cho, jaycho2007@gmail.com
          Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
          Ching-Yao (Jason) Chou, endpj@access.ntu.edu.tw
          github :
o This file is for utility functions and hyper-parameter configuration

"""
"""
Standardize each Column of Matrix
"""
def myStandardize(Matrix):
    from sklearn import preprocessing
    MatrixMod = preprocessing.scale(Matrix)
    for col in range(MatrixMod.shape[1]):
        MatrixMod[:,col] = MatrixMod[:,col] / np.linalg.norm(MatrixMod[:,col], 2)

    return MatrixMod

def myNormalize(Z, axis = 0):
    if axis == 0: # normalize each column
        for i in range(Z.shape[1]):
            Z[:,i] /= LA.norm(Z[:,i],2)
    else:
        for i in range(Z.shape[0]):
            Z[i,:] /= LA.norm(Z[i,:],2)
    return Z

def myNormalize_v2(Zref, Z, axis = 1):
    if axis == 1: # normalize each row
        Z_mean = Zref.mean(axis = 1).reshape((-1,1))
        Z_std = Zref.std(axis = 1).reshape((-1,1))
        Z_sub = Z - Z_mean
        Z_nor = Z_sub / Z_std
    else: # normalize each column
        Z_mean = Zref.mean(axis = 0).reshape((1,-1)) # mean of each column
        Z_std = Zref.std(axis = 0).reshape((1,-1))   # std of each column
        Z_sub = Z - Z_mean
        Z_nor = Z_sub / Z_std
    return Z_nor

"""
customize loading data function
"""
def loadData(fileName, ratio, nClasses):
    data = sio.loadmat(fileName)

    if fileName == "../../data/resample.mat":
        matDict = {}
        matDict['YArr'] = data['X_r_test'].T
        labelTest = data['Y_test'].reshape((1, -1))
        labelTest[labelTest == -1] = 2
        matDict['ttls'] = labelTest.astype(int)
        TrainTmp = data['X_r_train'].T
        TrLabelTmp = data['Y_train'].reshape((1,-1))
        TrLabelTmp[TrLabelTmp == -1] = 2

        if ratio != 1.0:
            n = TrainTmp.shape[0]
            N0 = int(TrainTmp.shape[1])
            N = int(TrainTmp.shape[1] * ratio)
            Nc0 = int(N0 / nClasses)
            Nc = int(N / nClasses)

            matDict['trls'] = np.zeros((1,N)).astype(int)
            matDict['XArr'] = np.zeros((n,N))
            for c in range(nClasses):
                permut = np.random.permutation(Nc0)
                matDict['XArr'][:,c*Nc:(c+1)*Nc] = TrainTmp[:, permut[:Nc] + c*Nc0]
                matDict['trls'][:,c*Nc:(c+1)*Nc]  = TrLabelTmp[:, permut[:Nc] + c*Nc0]
        else:
            labelTrain = data['Y_train'].reshape((1, -1))
            labelTrain[labelTrain == -1] = 2
            matDict['trls'] = labelTrain.astype(int)
            matDict['XArr'] = data['X_r_train'].T
            n = TrainTmp.shape[0]
            N = TrainTmp.shape[1]

        matDict['N'] = N
        matDict['n'] = [n]
        return matDict

"""
For dictionary projection
"""
def projectionDic(D):
    for i in range(D.shape[1]):
        norm_col_squared = np.sum(np.square(D[:,i]))
        if norm_col_squared > 1:
            D[:,i] /= np.sqrt(norm_col_squared)
    return D    

def fista(fn_grad, Xinit, L, alambda, opts, fn_calc_F, show_progress = False):
    """
    * A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse
            Problems.
    * Solve the problem: `X = arg min_X F(X) = f(X) + lambda||X||_1` where:
       - `X`: variable, can be a matrix.
       - `f(X)` is a smooth convex function with continuously differentiable
       with Lipschitz continuous gradient `L(f)`
    * This is modified from https://github.com/tiepvupsu/DICTOL_python.git
       - Termination condition
    * Syntax: `[X, iter] = fista(grad, Xinit, L, lambda, opts, calc_F)` where:
       - INPUT:
            + `grad`: a _function_ calculating gradient of `f(X)` given `X`.
            + `Xinit`: initial guess.
            + `L`: the Lipschitz constant of the gradient of `f(X)`.
            + `lambda`: a regularization parameter, can be either positive a
                    scalar or a weighted matrix.
            + `opts`: a _structure_ variable describing the algorithm.
              * `opts.max_iter`: maximum iterations of the algorithm.
                    Default `300`.
              * `opts.tol`: a tolerance, the algorithm will stop if difference
                    between two successive `X` is smaller than this value.
                    Default `1e-8`.
              * `opts.show_progress`: showing `F(X)` after each iteration or
                    not. Default `false`.
            + `calc_F`: optional, a _function_ calculating value of `F` at `X`
                    via `feval(calc_F, X)`.
      - OUTPUT:
        + `X`: solution.
        + `iter`: number of iterations.
        + `cost_info` : cost function of each iteration
    """
    Linv = 1/L
    lambdaLiv = alambda/L
    x_old = Xinit.copy()
    y_old = Xinit.copy()
    t_old = 1
    it = 0
    cost_old = float("inf") # the positive infinity number
    cost_info = np.zeros((opts.max_iter))
    while it < opts.max_iter:
        it += 1
        x_new = np.real(shrinkage(y_old - Linv*fn_grad(y_old), lambdaLiv))
        t_new = 0.5*(1 + math.sqrt(1 + 4*t_old**2))
        y_new = x_new + (t_old - 1)/t_new * (x_new - x_old)
        cost_new = fn_calc_F(x_new)
        e = cost_old - cost_new
        cost_old = cost_new
        cost_info[it-1] = cost_new
        #e = norm1(x_new - x_old)/x_new.size
        if e < opts.tol:
            cost_info = cost_info[:it-1]
            if opts.verbose:
                print(cost_info.shape)
            break
        x_old = x_new.copy()
        t_old = t_new
        y_old = y_new.copy()
        if opts.verbose:
            print ('iter = '+str(it)+', cost = %.6f' % cost_new)
        if not opts.verbose and show_progress and it%10 == 0:
            str0 = progress_str(it, opts.max_iter, 50)
            sys.stdout.write("\r%s %.2f%%" % (str0, (it*100.0)/opts.max_iter ))
            sys.stdout.flush()
    if not opts.verbose and show_progress and it%10 == 0:
        print ('')
    return (x_new, it, cost_info)   
        
"""
For hyper-parameter configuration
"""
class TDDLOpts(object):
    def __init__(self,
        d = 44,
        lambda1 = 0.01,
        iterUnsupDic = 20,
        iterSupDic = 20,
        batchSize = 100,
        init = True,
        initModel = "",
        validation = False,
        verbose = False,
        show_result = True):

        self.d = d
        self.lambda1 = lambda1
        self.iterUnsupDic = iterUnsupDic
        self.iterSupDic = iterSupDic
        self.batchSize = batchSize
        self.init = init
        self.initModel = initModel
        self.validation = validation
        self.verbose = verbose
        self.show_result = show_result

class ClassifierOpts(object):
    def __init__(self,
        t0 = 0.2,
        threshold = 1e-8,
        nuQuad = 1e-8,
        rhoQuad = 0.25,
        nClasses = 2,
        intercept = True):

        self.t0 = t0
        self.threshold = threshold
        self.nuQuad = nuQuad
        self.rhoQuad = rhoQuad
        self.nClasses = nClasses
        self.intercept = intercept


"""
Source code below is adapted from
    DICTOL - A Discriminative dictionary Learning Toolbox for Classification (Python 2.7 version).
and upgraded to Python 3.6 version
github repo: https://github.com/tiepvupsu/DICTOL_python.git
"""

#################################
#               ODL             #
#################################
def ODL(Y, k, lambda1, opts, method = 'fista'):
    """
    * Solving the following problem:
     (D, X) = \arg\min_{D,X} 0.5||Y - DX||_F^2 + lambda1||X||_1
    * Syntax: `(D, X) = ODL(Y, k, lambda1, opts)`
      - INPUT:
        + `Y`: collection of samples.4/7/2016 7:35:39 PM
        + `k`: number of atoms in the desired dictionary.
        + `lambda1`: norm 1 regularization parameter.
        + `opts`: option.
        + `sc_method`: sparse coding method used in the sparse coefficient update. Possible values:
          * `'fista'`: using FISTA algorithm. See also [`fista`](#fista).
          * `'spams'`: using SPAMS toolbox [[12]](#fn_spams).
      - OUTPUT:
        + `D, X`: as in the problem.
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 4/7/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """

    Y_range = np.array([0, Y.shape[1]])
    D_range = np.array([0, k])
    D = pickDfromY(Y, Y_range, D_range)  # D Init by input data
    X = np.zeros((D.shape[1], Y.shape[1]))
    if opts.verbose:
        print ('Initial cost: %5.4f' % ODL_cost(Y, D, X, lambda1))
    it = 0
    optsX = Opts(max_iter = 300)
    optsD = Opts(max_iter = 200, tol = 1e-8)
    while it < opts.max_iter:
        it += 1
        # Sparse coding
        X = lasso_fista(Y, D, X, lambda1, optsX)[0]
        # X = X0[0]
        if opts.verbose:
            costX = ODL_cost(Y, D, X, lambda1)
            print ('iter: %3d' % it, '| costX = %4.4f' % costX )#'it: ', itx
        # Dictionary update
        F = np.dot(X, X.T)
        E = np.dot(Y, X.T)
        D = ODL_updateD(D, E, F, optsD)
        if opts.verbose:
            costD = ODL_cost(Y, D, X, lambda1)
            print ('          | costD = %4.4f' % costD) #'it: ', itd
            if abs(costX - costD) < opts.tol:
                break

        else:
            if opts.processbar:
                str0 = progress_str(it, opts.max_iter, 50)
                sys.stdout.write("\r%s %.2f%%" % (str0, (it*100.0)/opts.max_iter ))
                sys.stdout.flush()
    if opts.verbose:
        print ('Final cost: %4.4f' % ODL_cost(Y, D, X, lambda1))
    else:
        if opts.processbar:
            print("")
    return (D, X)

def ODL_cost(Y, D, X, lambda1):
    """
    cost = 0.5* ||Y - DX||_F^2 + lambda1||X||_1
    """

    return 0.5*LA.norm(Y - np.dot(D, X), 'fro')**2 + lambda1*LA.norm(X, 1)

def ODL_updateD(D, E, F, opts):
    """
    * The main algorithm in ODL.
    * Solving the optimization problem:
      `D = arg min_D -2trace(E'*D) + trace(D*F*D')` subject to: `||d_i||_2 <= 1`,
         where `F` is a positive semidefinite matrix.
    * Syntax `[D, iter] = ODL_updateD(D, E, F, opts)`
      - INPUT:
        + `D, E, F` as in the above problem.
        + `opts`. options:
          * `opts.max_iter`: maximum number of iterations.
          * `opts.tol`: when the difference between `D` in two successive
            iterations less than this value, the algorithm will stop.
      - OUTPUT:
        + `D`: solution.
        + `iter`: number of run iterations.
    -----------------------------------------------
    Author: Tiep Vu, thv102@psu.edu, 04/07/2016
            (http://www.personal.psu.edu/thv102/)
    -----------------------------------------------
    """
    def calc_cost(D):
        return -2*np.trace(np.dot(E, D.T)) + np.trace(np.dot(np.dot(F, D.T), D))

    D_old = D.copy()
    it = 0
    sizeD = D.size
    # print opts.tol
    while it < opts.max_iter:
        it = it + 1
        for i in range(D.shape[1]):
            if F[i,i] != 0:
                a = 1.0/F[i,i] * (E[:, i] - D.dot(F[:, i])) + D[:, i]
                D[:,i] = a/max(LA.norm(a, 2), 1)
        if opts.verbose:
            print('iter = %3d | cost = %.4f | tol = %f' %(it, calc_cost(D), LA.norm(D - D_old, 'fro')/sizeD ))

        if LA.norm(D - D_old, 'fro')/sizeD < opts.tol:
            break
        D_old = D.copy()
    return D


#################################
#              FISTA            #
#################################
def lasso_fista(Y, D, Xinit, alambda, opts, show_progress = False):
    """
    * Syntax: `[X, iter] = lasso_fista(Y, D, Xinit, lambda, opts)`
    * Solving a Lasso problem using FISTA [[11]](#fn_fista):
        `X, = arg min_X 0.5*||Y - DX||_F^2 + lambda||X||_1`.
        Note that `lambda` can be either a positive scalar or a matrix with
        positive elements.
      - INPUT:
        + `Y, D, lambda`: as in the problem.
        + `Xinit`: Initial guess
        + `opts`: options. See also [`fista`](#fista)
      - OUTPUT:
        + `X`: solution.
        + `iter`: number of fistat iterations.
        + `cost_info` : cost function of each iteration
    """
    if Xinit.size == 0:
        Xinit = np.zeros((D.shape[1], Y.shape[1]))

    def calc_f(X):
        return 0.5*normF2(Y - np.dot(D, X))

    def calc_F(X):
        if isinstance(alambda, np.ndarray):
            return calc_f(X) + alambda*abs(X) # element-wise multiplication
        else:
            return calc_f(X) + alambda*norm1(X)

    DtD = np.dot(D.T, D)
    DtY = np.dot(D.T, Y)
    def grad(X):
        g =  np.dot(DtD, X) - DtY
        return g
    L = np.max(LA.eig(DtD)[0])
    (X, it, cost_info) = fista(grad, Xinit, L, alambda, opts, calc_F, show_progress)
    return (X, it, cost_info)

def shrinkage(U, alambda):
    """
    * Soft thresholding function.
    * Syntax: ` X = shrinkage(U, lambda)`
    * Solve the following optimization problem:
    `X = arg min_X 0.5*||X - U||_F^2 + lambda||X||_1`
    where `U` and `X` are matrices with same sizes. `lambda` can be either
    positive a scalar or a positive matrix (all elements are positive) with
    same size as `X`. In the latter case, it is a weighted problem.
    """
    return np.maximum(0, U - alambda) + np.minimum(0, U + alambda)

#################################
#              norm             #
#################################

def norm1(X):
    # pass
    if X.shape[0]*X.shape[1] == 0:
        return 0
    return abs(X).sum()
    # return LA.norm(X, 1)

def normF2(X):
    # pass
    if X.shape[0]*X.shape[1] == 0:
        return 0
    return LA.norm(X, 'fro')**2

def progress_str(cur_val, max_val, total_point=50):
    p = int(math.ceil(float(cur_val)*total_point/ max_val))
    return '|' + p*'#'+ (total_point - p)*'.'+ '|'

def pickDfromY(Y, Y_range, D_range):
    """
    randomly pick k_c samples from Y_c
    """
    C = Y_range.size - 1
    D = np.zeros((Y.shape[0], D_range[-1]))
    for c in range(C):
        Yc = get_block_col(Y, c, Y_range)
        N_c = Yc.shape[1]
        ids = randperm(N_c)
        range_Dc = get_range(D_range, c)
        kc = D_range[c+1] - D_range[c]
        D[:, D_range[c]:D_range[c+1]] = Yc[:, np.sort(ids[:kc])]
    return D

def get_block_col(M, C, col_range):
    """
    * Syntax: `Mc = get_block_col(M, c, col_range)`
    * Extract a block of columns from a matrix.
        - `M`: the big matrix `M = [M_1, M_2, ...., M_C]`.
        - `C`: blocks indices (start at 0).
        - `col_range`: range of samples, see `Y_range` and `D_range` above.
    * Example: `M` has 25 columns and `col_range = [0, 10, 25]`, then
    `get_block_col(M, 1, col_range)` will output the first block of `M`,
    i.e. `M(:, 1:10)`.
    """
    if isinstance(C, int):
        return M[:, col_range[C]: col_range[C+1]]
    if isinstance(C, list) or isinstance(C, (np.ndarray, np.generic)):
        ids = []
        for c in C:
            ids = ids + range(col_range[c], col_range[c+1])
        return M[:, ids]

def randperm(n):
    return np.random.permutation(range(n))

def get_range(arange, c):
    return range(arange[c], arange[c+1])

class Opts:
    """
    parameters options. Store regularization parameters and algorithm stop
    criteria
    """
    def __init__(self, tol = 1e-8, max_iter = 100, show_cost = False,\
        test_mode = False, lambda1 = None, lambda2 = None, lambda3 = None, \
        eta = None, check_grad = False, verbose = False, processbar = False):
        self.tol        = tol
        self.max_iter   = max_iter
        self.show_cost  = show_cost
        self.test_mode  = test_mode
        self.check_grad = check_grad
        self.lambda1    = lambda1
        self.lambda2    = lambda2
        self.lambda3    = lambda3
        self.verbose    = verbose
        self.processbar = processbar
    def copy(self):
        opts = Opts()
        opts.tol        = self.tol
        opts.max_iter   = self.max_iter
        opts.show_cost  = self.show_cost
        opts.test_mode  = self.test_mode
        opts.check_grad = self.check_grad
        opts.lambda1    = self.lambda1
        opts.lambda2    = self.lambda2
        opts.lambda3    = self.lambda3
        opts.verbose    = self.verbose
        opts.processbar = self.processbar
        return opts