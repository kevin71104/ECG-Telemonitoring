from utils import *
from Indep_PCA import *
from TDDL import *

import argparse
import scipy.io as sio
import numpy as np
import time

"""
o Author: Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
          Bo-Hong (Jay) Cho, jaycho2007@gmail.com
          Ching-Yao (Jason) Chou, endpj@access.ntu.edu.tw
          github :
o Script:
  - python3 CA-E.py <-f datapath> <-c #classes> <-nr data ratio> [-nor]
                    <-P CS type> <-cr compressed ratio>
                    [-im initmodel] <-a #atoms> <-l lambda1> [-i]
                    <-r learning rate> <-to constant iter ratio> <-b batchsize> <-it #iters> <-nu regularize> [-val] [-int]
                    <-tn #tests> [-s]
  - Ex:  python3 CA-E.py -i -int -nr 0.3 -b 50 -tn 1
"""

def preprocess(X_hat, mu_hat, DECODING):
    S = np.matmul(DECODING, X_hat - mu_hat)
    return S

def main():
    #################################
    #        Argument Parser        #
    #################################
    parser = argparse.ArgumentParser(description = "Eigenspace-aided Compressed Analysis (CA-E)")

    #==== DATA-RELATED ====
    parser.add_argument("-f",  "--file",        type = str,   help = "datapath",               default = "../../data/resample.mat")
    parser.add_argument("-c",  "--classes",     type = int,   help = '# classes',              default = 2)
    parser.add_argument("-nr", "--number",      type = float, help = 'data ratio',             default = 0.5)
    parser.add_argument("-nor", "--normalize",  action="store_true")
    #==== CS-RELATED ====
    parser.add_argument("-P",  "--PHI",         type = str,   help = 'CS matrix type',         default = "Ber")
    parser.add_argument("-cr", "--compress",    type = float, help = 'compressed ratio',       default = 0.25)
    #==== DICTIONARY-RELATED ====
    parser.add_argument("-im", "--initModel",   type = str,   help = "init model",             default = "CA-E")
    parser.add_argument("-a",  "--atom",        type = int,   help = 'atom size',              default = 50)
    parser.add_argument("-l",  "--lambda1",     type = float, help = 'lambda',                 default = 0.5)
    parser.add_argument("-i",  "--init",        action="store_true")
    #==== TRAINING-RELATED ====
    parser.add_argument("-r",   "--rho",        type = float, help = 'learning rate',          default = 0.1)
    parser.add_argument("-t",   "--t0",         type = float, help = 'Iter ratio to drop rho', default = 0.2)
    parser.add_argument("-b",   "--batch",      type = int,   help = 'batch size',             default = 100)
    parser.add_argument("-it",  "--iteration",  type = int,   help = 'iteration',              default = 150)
    parser.add_argument("-nu",  "--nu",         type = float, help = 'W regularizer',          default = 1e-5)
    parser.add_argument("-int", "--intercept",  action="store_true")
    parser.add_argument("-val", "--validation", action="store_true")
    #==== TESTING-RELATED ====
    parser.add_argument("-tn",  "--testNum",    type = int,   help = 'number of tests',        default = 100)
    #==== PCA-RELATED ====
    parser.add_argument("-s", "--semi",         action="store_true", help = 'semi-PCA')
    args = parser.parse_args()

    #################################
    #             OPTS              #
    #################################
    wholeOpts = TDDLOpts(lambda1 = args.lambda1, iterUnsupDic = 50, iterSupDic = args.iteration,
                         batchSize = args.batch, d = args.atom, init = args.init, initModel = args.initModel,
                         validation = False, verbose = False, show_result = False)

    Classifieropts = ClassifierOpts(threshold = 1e-8, nuQuad = args.nu,rhoQuad = args.rho, t0 = args.t0,
                                    nClasses = args.classes, intercept = args.intercept)

    fista_opts = Opts(lambda1 = args.lambda1, max_iter = 500, verbose = False)
    fista_opts_2 = Opts(lambda1 = args.lambda1, max_iter = 500, verbose = True)

    ODL_opts = Opts(lambda1 = args.lambda1, max_iter = wholeOpts.iterUnsupDic, verbose = False)

    print ("=========== Eigenspace-aided Compressed Analysis (CA-E) ===========")
    print ("Apply CA-E on " + args.file[11:] + " with parameters:")
    print ("(Data Ratio, BatchSize) = ({0}, {1})".format(args.number, args.batch))
    print ("(Number of Atoms, lambda) = ({0}, {1})".format(args.atom, args.lambda1))
    print ("(nu, rho) = ({0}, {1})".format(args.nu, args.rho))
    print ("Compressed Ratio = {0}".format(args.compress))
    print ("Number of tests: {0}".format(args.testNum))
    print ("Iterations: {0} and start to drop at {1}".format(args.iteration, int(args.iteration*args.t0)))
    print ("===================================================================")

    #################################
    #           MAIN CODE           #
    #################################

    #==== LOAD DATA ====
    print ("==== START PCA ====")
    train_PCA_0 = time.time()
    dataDict, _ = Indep_PCA(args, normalize = args.normalize, semi = args.semi)
    train_PCA_1 = time.time()
    X_train = dataDict['X_train']
    X_test = dataDict['X_test']

    T_train = dataDict['T_train']
    S_test = dataDict['S_test']
    trls = dataDict['trls'].astype(int)
    ttls = dataDict['ttls'].astype(int)

    if args.compress < 1:
        PHI = dataDict['PHI']
    PSI = dataDict['PSI']
    x_mean = dataDict['x_mean']

    n  = X_train.shape[1]
    nt = X_test.shape[1]
    N = X_train.shape[0] # 512
    cr = args.compress
    M = int(cr*N)
    r = S_test.shape[0]
    d = args.atom
    testNum = args.testNum
    cost = False # `True`: get cost process of FISTA
    seq = False # `True`: sequentially inference
    print ("Dimension Change {0} -> {1} -> {2}".format(N, M, r))

    #==== TDDL on T ====
    print ("==== START TDDL on T ====")
    train_TDDL_0 = time.time()
    model, _, time_info = TDDL(T_train, S_test, trls, ttls, wholeOpts, Classifieropts, fista_opts, ODL_opts, timing=True)
    train_TDDL_1 = time.time()
    train_time = train_TDDL_1 - train_TDDL_0 + train_PCA_1 - train_PCA_0
    D = model['D']
    W = model['W']
    b = model['b']

    accMtx = np.zeros((args.testNum,2))
    TestTimeList = np.zeros((args.testNum,1))
    LASSOTimeList = np.zeros((args.testNum,2))
    LASSOIterList = np.zeros((args.testNum,2))
    #==== TEST on S ====
    print("==== Start Simulations -- {0} Simulations ====".format(args.testNum))
    for i in range(testNum):
        if args.compress < 1:
            from module import CS_API
            PHI = CS_API.PHI_generate(M, N, 'BERNOULLI')
            PHI = myNormalize(PHI)
            THETA = np.matmul(PHI,PSI)
            DECODING = LA.pinv(THETA)
            mu_hat = np.matmul(PHI, x_mean)
            X_hat_train = np.matmul(PHI, X_train)
            X_hat_test = np.matmul(PHI, X_test)

            test_0 = time.time() # START TO COUNT TIME
            S_train = preprocess(X_hat_train, mu_hat, DECODING)
            S_test = preprocess(X_hat_test, mu_hat, DECODING)
        else:
            S_train = np.matmul(PSI.T, (X_train - x_mean))
            S_test = np.matmul(PSI.T, (X_test - x_mean))
            test_0 = time.time() # START TO COUNT TIME
            
        t_i_1 = time.time()
        if seq:
            iter_tr = 0
            Atr1 = np.zeros((d,n))
            for idx in range(n):
                s = S_train[:,idx].reshape((r,1))
                alpha, iter_cnt, _ = lasso_fista(s, D, np.array([]), args.lambda1, fista_opts )
                Atr1[:,idx] = alpha.reshape(d)
                iter_tr += iter_cnt
        else:
            Atr1, iter_tr, _ = lasso_fista(S_train, D, np.array([]), args.lambda1, fista_opts )
        t_i_2 = time.time()

        if seq:
            iter_tt = 0
            Att1 = np.zeros((d,nt))
            for idx in range(nt):
                s = S_test[:,idx].reshape((r,1))
                if idx == 100 and cost:
                    alpha, iter_cnt, cost_info = lasso_fista(s,  D, np.array([]), args.lambda1, fista_opts_2 )
                else:
                    alpha, iter_cnt, _ = lasso_fista(s,  D, np.array([]), args.lambda1, fista_opts )
                Att1[:,idx] = alpha.reshape(d)
                iter_tt += iter_cnt
        else:
            Att1, iter_tt, _ = lasso_fista(S_test, D, np.array([]), args.lambda1, fista_opts )
        t_i_3 = time.time()

        if cost:
            cost_mtx = {}
            cost_mtx['cost_info'] = cost_info
            sio.savemat('cost.mat', cost_mtx)

        if i == 0:
            model['Atr_x_hat'] = Atr1
            model['Att_x_hat'] = Att1
            if args.compress < 1:
                model['PHI'] = PHI
            sio.savemat('../../model/' + args.initModel + '_F.mat', model)

        modelOutTrain_1 = np.matmul(W.T, Atr1) + np.tile(b, (1, n))
        trainPred_1 = np.argmax(modelOutTrain_1, axis = 0)
        modelOutTest = np.matmul(W.T, Att1) + np.tile(b, (1, nt))
        testPred = np.argmax(modelOutTest, axis = 0)

        test_1 = time.time()
        trainAcc1 = float(np.sum(trainPred_1 == (trls.reshape(n) - 1))) / n * 100
        testAcc = float(np.sum(testPred == (ttls.reshape(nt) - 1))) / nt * 100

        accMtx[i,:] = [trainAcc1, testAcc]
        TestTimeList[i,:] = [test_1 - test_0]
        LASSOTimeList[i,:] = [t_i_2 - t_i_1, t_i_3 - t_i_2]
        LASSOIterList[i,:] = [iter_tr, iter_tt]
        #==== TESTING PROCESS BAR ====
        str0 = progress_str(i+1, args.testNum, 50)
        sys.stdout.write("\r%s %.2f%%" % (str0, ((i+1)*100.0)/args.testNum))
        sys.stdout.flush()

    print ("\n==== The result of CA_IE_TDDL ====")
    accMean = np.mean(accMtx, axis = 0)
    accStd = np.std(accMtx, axis = 0)
    print (np.array_str(accMean, precision=2, suppress_small=True))
    print (np.array_str(accStd,  precision=3, suppress_small=True))

    print ("\n==== TIMING INFORMATION ====")
    print ("--- Overall Time")
    print ("Training Time: {:4.3f}".format(train_time))
    test_time = np.sum(TestTimeList) * 1000
    if seq:
        test_time_aver = test_time /(n+nt) /testNum
    else:
        test_time_aver = test_time /testNum
    print ("Testing Time: {:4.3f}/{:4.3f}".format(test_time, test_time_aver))
    #print (np.array_str(TestTimeList*1000, precision=3, suppress_small=True))

    print ("--- LASSO FISTA Time")
    LASSO_time = np.sum(LASSOTimeList, axis = 0) # get the average of each column
    LASSO_iter = np.sum(LASSOIterList, axis = 0) # get the average of each column
    lasso_train_time = LASSO_time[0] / LASSO_iter[0]
    lasso_test_time = LASSO_time[1] / LASSO_iter[1]
    if seq:
        lasso_time_mean = np.sum(LASSO_time) / (n+nt) / testNum
        lasso_iter_mean = np.sum(LASSO_iter) / (n+nt) / testNum
    else:
        lasso_time_mean = np.sum(LASSO_time) / testNum
        lasso_iter_mean = np.sum(LASSO_iter) / testNum
    lasso = lasso_time_mean / lasso_iter_mean
    print ("Train: {:4.3f}; Test: {:4.3f}".format(lasso_train_time * 1e6, lasso_test_time * 1e6))
    print ("Total Time: {:4.3f}; Total Iter: {:.2f}; Time/Iter: {:4.3f}".format(lasso_time_mean * 1e6, lasso_iter_mean, lasso * 1e6))
    #print (np.array_str(LASSOTimeList, precision=3, suppress_small=True))
    #print (np.array_str(LASSOIterList, precision=3, suppress_small=True))
    print ("--- TDDL Time")
    print ("Init Time: {:.3f}; FISTA Time: {:.3f}".format(time_info[0], time_info[1]))
    print ("Gradient Time: {:.3f}; Update Time: {:.3f}".format(time_info[2], time_info[3]))
    #print (np.array_str(time_info, precision=3, suppress_small=True))

if __name__ == '__main__':
    main()
