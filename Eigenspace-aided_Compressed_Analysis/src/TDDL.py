from utils import *
from supDL_linear import *
import numpy as np
import scipy.io as sio

def TDDL(XArr, YArr, trls, ttls, wholeOpts, Classifieropts, fista_opts, ODL_opts, timing):
    """
    o Solving the following problem:
      (D, W) = \arg\min_{D,W} E{||y - W alpha(x,D)||_2^2} + nu/2 ||W||_2^2
    o Syntax: `(Model, Acc) = TDDL(XArr, YArr, trls, ttls, wholeOpts, Classifieropts, fista_opts, ODL_opts)`
      - INPUT:
        + `XArr`: collection of training samples, has dimension of (N,n)
        + `YArr`: collection of testing samples
        + `trls`: collection of training labels
        + `ttls`: collection of testing labels
        + `wholeOpts` --- option for whole TDDL
          * `d`            : the number of atoms in dictionary
          * `lambda1`      : regularization term for l_1 norm
          * `iterUnsupDic` : iterations of unsupervised (used in ODL)
          * `iterSupDic`   : iterations of TDDL
          * `batchSize`    : size of each batch
        + `Classifieropts` --- option for W.
          * `t0`        : learning rate will keep constant until iterations exceeds this ratio
          * `threshold` : entry in alpha will be considered effective only if it's bigger than threshold
          * `nuQuad`    : regularization coefficients of W
          * `rhoQuad`   : initial learning rate
          * `nClasses`  : number of classes
          * `intercept` : use constant term in classifier or not
        + `fista_opts` --- option for FISTA [13]
          * `lambda1`  : regularization term for l_1 norm
          * `max_iter` : the maximum iterations in FISTA
        + `ODL_opts` --- option for ODL [13]
          * `lambda1`  : regularization term for l_1 norm
          * `max_iter` : the maximum iterations in ODL (theory in [11] and code in [14])
        + `timing` : if (true) record timing information
      - OUTPUT:
        + `Model`
          * `D` : supervised sparse representation learning layer
          * `W` : classifier
          * `b` : bias term
        + `Acc` : a list consisting of TrainAcc and TestAcc
    -----------------------------------------------
    o Author: Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
              Bo-Hong (Jay) Cho, jaycho2007@gmail.com
              Ching-Yao (Jason) Chou, endpj@access.ntu.edu.tw
              github :
    -----------------------------------------------
    o Reference: [8] J. Mairal, F. Bach and J. Ponce, “Task-Driven Dictionary Learning,”
                     in IEEE Trans. Pattern Anal. Mach. Intell.,
                     vol. 34, no. 4, pp. 791 - 804, Apr. 2012.
    """

    #==== PARAMETERS ====
    n  = trls.shape[1]
    ntt = ttls.shape[1]
    d  = wholeOpts.d
    lambda1 = wholeOpts.lambda1
    nClasses  = Classifieropts.nClasses
    intercept = Classifieropts.intercept
    time_info = np.zeros(4)
    if timing:
        import time

    #==== VALIDATION SET ====
    if wholeOpts.validation:
        print('Validation Set...')
        permut = np.random.permutation(n)
        ntr = int(n*0.8)
        Xval = XArr[:,permut[ntr:]]
        Yval = trls[:,permut[ntr:]]
        Xtrain = XArr[:,permut[:ntr]]
        Ytrain = trls[:,permut[:ntr]]
        show_valid = True
    else:
        ntr = n
        Xval = []
        Yval = []
        Xtrain = XArr
        Ytrain = trls
        show_valid = False
    if timing:
        init_time_0 = time.time()
    if wholeOpts.init:
        print ('Start Initializing Unsupervised Dictionary...', end='')
        DUnsup, _ = ODL(Xtrain, d, lambda1, ODL_opts)
        DUnsup = projectionDic(DUnsup)
        print ('Init Done')

        #==== A: Sparse Coefficients feeded into classifiers as feature vector ====
        Atr, _, _ = lasso_fista(Xtrain, DUnsup, np.array([]), wholeOpts.lambda1, fista_opts )
        Att, _, _ = lasso_fista(YArr,   DUnsup, np.array([]), wholeOpts.lambda1, fista_opts )

        #==== ONE-HOT ENCODING ====
        outputVectorTrain = np.zeros((nClasses, ntr)).astype(int)
        for j in range(ntr):
            outputVectorTrain[Ytrain[0, j] - 1, j] = 1  # use index start with 1

        #==== LINEAR REGRESSION ====
        from sklearn.linear_model import LinearRegression, Ridge
        # clf = LinearRegression()
        # alpha = C^-1, Larger values specify stronger regularization
        clf = Ridge(alpha=10.0, fit_intercept = intercept)

        clf.fit(Atr.T, outputVectorTrain.T)
        W = clf.coef_.T
        if intercept:
            b = clf.intercept_.reshape((nClasses,1))

        modelInit = {}
        modelInit['W'] = W
        modelInit['b'] = b
        modelInit['D'] = DUnsup
    else:
        modelInit = sio.loadmat('../../model/' + wholeOpts.initModel +'.mat')
        print ('Load Init Model Done')
        Atr = np.zeros((d, ntr))
        Atr, _, _ = lasso_fista(XArr, modelInit['D'], np.array([]), wholeOpts.lambda1, fista_opts )
        Att = np.zeros((d, ntt))
        Att, _, _ = lasso_fista(YArr, modelInit['D'], np.array([]), wholeOpts.lambda1, fista_opts )
    if timing:
        init_time_1 = time.time()
        init_time = init_time_1 - init_time_0
    D_init = modelInit['D']
    W_init = {}
    W_init['W'] = modelInit['W']
    W_init['b'] = modelInit['b']

    if wholeOpts.show_result:
        modelOutTrain = np.zeros((nClasses, ntr))
        modelOutTest  = np.zeros((nClasses, ntt))
        trainPred = np.zeros((ntr, 1))
        testPred = np.zeros((ntt, 1))

        if intercept:
            modelOutTrain = np.matmul(modelInit['W'].T, Atr) + np.tile(modelInit['b'], (1, ntr))
            modelOutTest = np.matmul(modelInit['W'].T, Att) + np.tile(modelInit['b'], (1, ntt))
        else:
            modelOutTrain = np.matmul(modelInit['W'].T, Atr)
            modelOutTest = np.matmul(modelInit['W'].T, Att)

        trainPred = np.argmax(modelOutTrain, axis = 0)
        trainAcc = float(np.sum(trainPred == (Ytrain.reshape(ntr) - 1))) / ntr * 100
        testPred = np.argmax(modelOutTest, axis = 0)
        testAcc = float(np.sum(testPred == (ttls.reshape(ntt) - 1))) / ntt * 100
        print ('Unsupervised Classification Result...')
        print ('TrainUnSup : %4.2f %%' %(trainAcc))
        print ('TestUnSup  : %4.2f %%' %(testAcc))

    #==== SUPERVISED TRAINING (TDDL) ====
    print("Start supervised dictionary learning(TDDL)")
    DSup, modelQuadSup, TDDL_info = supDL_linear(Xtrain, Ytrain, Xval, Yval,
                                                 wholeOpts, Classifieropts, fista_opts,
                                                 D_init, W_init, show_valid = show_valid, timing=timing)
    if timing:
        TDDL_info_sum = np.sum(TDDL_info, axis = 0)
        time_info[0] = init_time
        time_info[1] = TDDL_info_sum[0]
        time_info[2] = TDDL_info_sum[1]
        time_info[3] = TDDL_info_sum[2]
    print("Dictionary done")

    W = modelQuadSup['W']
    b = modelQuadSup['b']
    if wholeOpts.show_result:
        Atr, _, _ = lasso_fista(Xtrain, DSup, np.array([]), lambda1, fista_opts )
        Att, _, _ = lasso_fista(YArr,   DSup, np.array([]), lambda1, fista_opts )

        modelOutTrain = np.zeros((nClasses, ntr))
        modelOutTest  = np.zeros((nClasses, ntt))
        trainPred = np.zeros((ntr, 1))
        testPred = np.zeros((ntt, 1))

        if intercept:
            modelOutTrain = np.matmul(W.T, Atr) + np.tile(b, (1, ntr))
            modelOutTest = np.matmul(W.T, Att) + np.tile(b, (1, ntt))
        else:
            modelOutTrain = np.matmul(W.T, Atr)
            modelOutTest = np.matmul(W.T, Att)

        trainPred = np.argmax(modelOutTrain, axis = 0)
        trainAcc = float(np.sum(trainPred == (Ytrain.reshape(ntr) - 1))) / ntr * 100

        testPred = np.argmax(modelOutTest, axis = 0)
        testAcc = float(np.sum(testPred == (ttls.reshape(ntt) - 1))) / ntt * 100

        print ('TDDL Results...')
        print ('TrainSup : %4.2f %%' %(trainAcc))
        print ('TestSup  : %4.2f %%' %(testAcc))
    else:
        trainAcc = 0
        testAcc = 0

    model = {}
    model['W'] = W
    model['b'] = b
    model['D'] = DSup
    acc = [trainAcc, testAcc]
    return model, acc, time_info
