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
o Author : Bo-Hong (Jay) Cho, jaycho2007@gmail.com
           Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
           Ching-Yao (Jason) Chou, endpj@access.ntu.edu.tw
  GitHub : https://github.com/kevin71104/ECG-Telemonitoring/tree/master/Eigenspace-aided_Compressed_Analysis/src
  Date   : 2018.06.14
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

    if fileName == "../data/resample.mat":
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