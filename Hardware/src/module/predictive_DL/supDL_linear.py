from ..utils import *
from ..sparse_coding.FISTA import *
import numpy as np

def supDL_linear(XArr, Y, Xval, Yval, wholeOpts, Classifieropts, fista_opts,
                 D_init = None, W_init = None, show_train = False,
                 show_valid = False, timing = False):
    """
    o Main algorithm in TDDL
    o Solving the following problem:
      (D, W) = \arg\min_{D,W} E{||y - W alpha(x,D)||_2^2} + nu/2 ||W||_2^2
    o Syntax: `(D, Model) = supDL_linear(XArr, Y, Xval, Yval, wholeOpts,
                                         Classifieropts, fista_opts, D_init, W_init)`
      - INPUT:
        + `XArr`: collection of training samples, has dimension of (N,n)
        + `Y`   : collection of training labels
        + `wholeOpts` --- option for whole TDDL
          * `d`            : the number of atoms in dictionary
          * `lambda1`      : regularization term for l_1 norm
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
        + `D_init` : initial dictionary if given
        + `W_init` : initial classifier if given
          * `W` : coefficients
          * `b` : bias term
        + `timing` : if (true) record timing information
      - OUTPUT:
        + `D` : supervised sparse representation learning layer
        + `Model`
          * `W` : classifier
          * `b` : bias term
    -----------------------------------------------
    o Author : Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
               Bo-Hong (Jay) Cho, jaycho2007@gmail.com
               Ching-Yao (Jason) Chou, endpj@access.ntu.edu.tw
      GitHub : https://github.com/kevin71104/ECG-Telemonitoring/tree/master/Eigenspace-aided_Compressed_Analysis/src
      Date   : 2018.06.14
    -----------------------------------------------
    o Reference: [8] J. Mairal, F. Bach and J. Ponce, “Task-Driven Dictionary Learning,”
                     in IEEE Trans. Pattern Anal. Mach. Intell.,
                     vol. 34, no. 4, pp. 791 - 804, Apr. 2012.
    """

    #==== INIT ====
    N = XArr.shape[0]
    n = XArr.shape[1]
    d = wholeOpts.d
    it = wholeOpts.iterSupDic
    lambda1     = wholeOpts.lambda1
    batchSize   = wholeOpts.batchSize

    threshold   = Classifieropts.threshold
    intercept   = Classifieropts.intercept     # whether to have bias
    nu          = Classifieropts.nuQuad
    rho         = Classifieropts.rhoQuad       # initial stepsize
    nClasses    = Classifieropts.nClasses

    #==== check if dictionary is provided ====
    if D_init is None:
        # Initialize D using randomly taken train samples from all classes
        D = np.zeros((N, d))
        numAtomPerClass = np.floor(d / nClasses).astype(int)

        temp = 0
        for i in range(nClasses):
            tempIndex = np.nonzero(Y == np.unique(Y)[i])[1]
            permut2 = np.random.permutation(tempIndex.shape[0]) # shuffle data
            D[:, temp: temp + numAtomPerClass] = XArr[:, tempIndex[0: numAtomPerClass]]
            temp += numAtomPerClass
        #==== fill the rest with random train samples ====
        if d > nClasses * numAtomPerClass:
            permut = np.random.permutation(n)                   # shuffle data
            D[:, temp:d] = XArr[:, permut[0 : d - temp + 1]]
    else:
        D = D_init

    #==== check if parameter is provided ====
    if W_init is None:
        modelQuad = {}
        W = np.random.standard_normal((d, nClasses))
        b = np.zeros((nClasses, 1))
        modelQuad['W'] = W
        modelQuad['b'] = b
    else:
        modelQuad = W_init
        W = modelQuad['W']
        b = modelQuad['b']

    t0 = n * it * Classifieropts.t0
    permut = np.random.permutation(n)
    XArr = XArr[:, permut]
    Y = Y[:, permut]

    #==== ONE-HOT ENCODING ====
    Y_onehot = np.zeros((nClasses, Y.shape[1]))
    for j in range(Y.shape[1]):
        Y_onehot[Y[0, j] - 1, j] = 1  # for index start with 1

    #==== OPTIMIZATION ====
    step = 0
    if timing:
        import time
        tmp = int((n - n % batchSize) / batchSize + 1)
        tmp = tmp * it
        timing_info = np.zeros((tmp, 3))
    else:
        timing_info = []
    for iters in range(it):
        for t in range(0, n - n % batchSize, batchSize):
            step += 1
            learnRate = rho*t0/step if t0 < step else rho # changeable learning rate

            A = np.zeros((d, batchSize))
            B = np.zeros((d, batchSize))

            gradD = np.zeros((N, d))
            gradW = np.zeros((d, nClasses))
            gradb = np.zeros((nClasses, 1))
            gradTemp = np.zeros((nClasses, batchSize))

            xt = XArr[:, t:t+batchSize].reshape((N,batchSize))
            yt = Y_onehot[:,t:t+batchSize].reshape((nClasses,batchSize))

            #==== Sparse Coding ====
            if timing:
                FISTA_time_0 = time.time()
            A = lasso_fista(xt, D, np.array([]), wholeOpts.lambda1, fista_opts)[0]
            if timing:
                FISTA_time_1 = time.time()
                FISTA_time = FISTA_time_1 - FISTA_time_0

            if timing:
                gradient_time_0 = time.time()
            for j in range(batchSize):
                beta = np.zeros((d, 1))
                alphaAbs = np.absolute(A[:,j]).reshape(d,1)
                act_row = []
                for row in range(d):
                    if alphaAbs[row, 0] > threshold:
                        act_row.append(row)
                # default setting if actrow is empty
                if len(act_row) == 0:
                    act_row = np.arange(D.shape[1])
                num_act = len(act_row)

                #==== Gradient of Ls respect to active set of alpha ====
                W_act = W[act_row, :]
                y = yt[:,j].reshape((nClasses, 1))
                alpha = A[:,j].reshape((d, 1))
                gradLs_actAlphaVec = np.zeros((num_act, 1))

                tempAlpha = np.matmul(W_act.T, alpha[act_row,0]).reshape((nClasses,1)) + b - y
                gradLs_actAlphaVec = np.matmul(W_act, tempAlpha)
                gradTemp[:,j] = tempAlpha.reshape(nClasses)

                #==== Calculate beta ====
                #print np.matmul(D[:,act_row].T, D[:,act_row])
                Cholesky = np.linalg.inv(np.matmul(D[:,act_row].T, D[:,act_row]))
                beta[act_row,0] = np.matmul(Cholesky,gradLs_actAlphaVec).reshape(num_act)
                B[:, j] = beta.reshape(d)
            #==== Gradient of Ls respect to W and b ====
            gradW = np.matmul(A, gradTemp.T) / batchSize + nu * W
            if intercept:
                gradb = np.sum(gradTemp, axis = 1).reshape((nClasses, 1)) / batchSize + nu * b

            #==== Gradient of Ls respect to D ====
            tempD1 = np.matmul(D,B)
            tempD2 = xt - np.matmul(D,A)
            gradDtmp = -np.matmul(tempD1, A.T) + np.matmul(tempD2, B.T)
            gradDtmp = gradDtmp / batchSize
            if timing:
                gradient_time_1 = time.time()
                gradient_time = gradient_time_1 - gradient_time_0

            #==== Update ====
            if timing:
                update_time_0 = time.time()
            W -= learnRate * gradW
            if intercept:
                b -= learnRate * gradb
            Dold = D
            D = projectionDic(Dold - learnRate * gradDtmp)
            if timing:
                update_time_1 = time.time()
                update_time = update_time_1 - update_time_0
                timing_info[step,:] = [FISTA_time, gradient_time, update_time]
        if wholeOpts.verbose:
            if show_train:
                permut = np.random.permutation(n)
                Xtr = XArr[:,permut[:50]]
                Ytr = Y[:,permut[:50]]
                Ntr = 50
                Atr, _, _ = lasso_fista(Xtr, D, np.array([]), lambda1, fista_opts )
                Out = np.matmul(W.T, Atr) + np.tile(b, (1, Ntr))
                Pred = np.argmax(Out, axis = 0)
                Acc1 = float(np.sum(Pred == Ytr.reshape(Ntr) - 1)) / Ntr * 100
            if show_valid:
                Nval = Xval.shape[1]
                Aval, _, _ = lasso_fista(Xval, D, np.array([]), lambda1, fista_opts )
                Out = np.matmul(W.T, Aval) + np.tile(b, (1, Nval))
                Pred = np.argmax(Out, axis = 0)
                Acc2 = float(np.sum(Pred == Yval.reshape(Nval) - 1)) / Nval * 100
            if show_train:
                if show_valid:
                    print ('Iter %3.3d : %4.2f %% / %4.2f %%' %(iters+1, Acc1, Acc2))
                else:
                    print ('Iter %3.3d : %4.2f %% / -----' %(iters+1, Acc1))
            else:
                if show_valid:
                    print ('Iter %3.3d : ----- / %4.2f %%' %(iters+1, Acc2))
        else:
            str0 = progress_str(iters+1, it, 50)
            sys.stdout.write("\r%s %.2f%%" % (str0, ((iters+1)*100.0)/it ))
            sys.stdout.flush()
    print ('')
    modelQuad['W'] = W
    modelQuad['b'] = b

    return D, modelQuad, timing_info
