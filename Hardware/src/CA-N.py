import argparse
import scipy.io as sio
import numpy as np
import time

from module.utils import *
from module.predictive_DL.TDDL import *
from module.sparse_coding.FISTA import *
from module.CS_API.CS_API import *

"""
o Author : Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
           Bo-Hong (Jay) Cho, jaycho2007@gmail.com
           Ching-Yao (Jason) Chou, endpj@access.ntu.edu.tw
  GitHub : https://github.com/kevin71104/ECG-Telemonitoring/tree/master/Eigenspace-aided_Compressed_Analysis/src
  Date   : 2018.06.14
o Script:
  - python3 CA-N.py <-f datapath> <-c #classes> <-nr data ratio> [-nor]
                    <-P CS type> <-cr compressed ratio>
				    [-im initmodel] <-a #atoms> <-l lambda1> [-i]
				    <-r learning rate> <-to constant iter ratio> <-b batchsize> <-it #iters> <-nu regularize> [-val] [-int]
				    <-tn #tests>
  - Ex: python3 CA-N.py -i -int -a 50 -l 0.5 -nr 0.5 -b 100 -tn 1
"""

def main():
    parser = argparse.ArgumentParser(description = "Naive Compressed Analysis (CA-N)")

    #==== DATA-RELATED ====
    parser.add_argument("-f",  "--file",        type = str,   help = "datapath",               default = "../data/resample.mat")
    parser.add_argument("-c",  "--classes",     type = int,   help = '# classes',              default = 2)
    parser.add_argument("-nr", "--number",      type = float, help = 'data ratio',             default = 0.5)
    parser.add_argument("-nor", "--normalize",  action="store_true")
    #==== CS-RELATED ====
    parser.add_argument("-P",  "--PHI",         type = str,   help = 'CS matrix type',         default = "Ber")
    parser.add_argument("-cr", "--compress",    type = float, help = 'compressed ratio',       default = 0.25)
    #==== DICTIONARY-RELATED ====
    parser.add_argument("-im", "--initModel",   type = str,   help = "init model",             default = "CA-N")
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
    args = parser.parse_args()

    #==== LOAD DATA ====
    dataDict = loadData(args.file, args.number, args.classes)
    XOri = dataDict['XArr']
    YOri = dataDict['YArr']
    if args.normalize:
        XArr_mean = XOri.mean(axis=1).reshape((-1,1))
        XArr = myNormalize(XOri - XArr_mean)
        YArr = myNormalize(YOri - XArr_mean)
    else:
        XArr = XOri
        YArr = YOri

    trls = dataDict['trls']
    ttls = dataDict['ttls']
    #==== PARAMETERS ====
    n  = trls.shape[1]
    nt = ttls.shape[1]
    N  = XArr.shape[0]
    cr = args.compress # compressed ratio
    M = int(N * cr)
    d  = args.atom
    nClasses = args.classes
    testNum = args.testNum
    cost = True

    #==== OPTS ====
    wholeOpts = TDDLOpts(lambda1 = args.lambda1, iterUnsupDic = 50, iterSupDic = args.iteration,
                         batchSize = args.batch, d = args.atom, init = args.init,
                         initModel = args.initModel, validation = False,
                         verbose = False, show_result = False)

    Classifieropts = ClassifierOpts(threshold = 1e-8, nuQuad = args.nu,
                                    rhoQuad = args.rho, t0 = args.t0)

    fista_opts = Opts(lambda1 = args.lambda1, max_iter = 500, verbose = False)
    fista_opts_2 = Opts(lambda1 = args.lambda1, max_iter = 50, verbose = True)

    ODL_opts = Opts(lambda1 = args.lambda1, max_iter = wholeOpts.iterUnsupDic, verbose = False)

    print ("================= Naive Compressed Analysis (CA-N) ================")
    print ("Apply CA-N on " + args.file[11:] + " with parameters:")
    print ("(Data Ratio, BatchSize) = ({0}, {1})".format(args.number, args.batch))
    print ("(Number of Atoms, lambda) = ({0}, {1})".format(args.atom, args.lambda1))
    print ("(nu, rho) = ({0}, {1})".format(args.nu, args.rho))
    print ("Compressed Ratio = {0}".format(args.compress))
    print ("Number of tests: {0}".format(args.testNum))
    print ("Iterations: {0} and start to drop at {1}".format(args.iteration, int(args.iteration*args.t0)))
    print ("===================================================================")

    train_TDDL_0 = time.time()
    model, _, time_info = TDDL(XArr, YArr, trls, ttls, wholeOpts, Classifieropts, fista_opts, ODL_opts, True)
    train_TDDL_1 = time.time()
    train_time = train_TDDL_1 - train_TDDL_0

    W = model['W']
    b = model['b']
    D = model['D']
    sio.savemat('../model/' + args.initModel + '_F.mat', model)

    accMtx = np.zeros((args.testNum,2))
    TestTimeList = np.zeros((args.testNum,1))
    LASSOTimeList = np.zeros((args.testNum,2))
    LASSOIterList = np.zeros((args.testNum,2))

    for i in range(testNum):
        if args.PHI == 'Ber':
            #print ("Using Bernoulli CS Matrix")
            PHI = PHI_generate(M, N, 'BERNOULLI')
        else:
            #print ('Using Normal CS Matrix')
            PHI = np.random.normal(0, 1.0, (M,N))
        PHI = myNormalize(PHI)

        #==== SIMULATE CS DATA ====
        PSI = np.matmul(PHI, D)
        Xmod = np.matmul(PHI, XOri)
        Ymod = np.matmul(PHI, YOri)
        if args.normalize:
            Xmod_mean = Xmod.mean(axis=1).reshape((-1,1))
            Xmod = myNormalize(Xmod - Xmod_mean)
            Ymod = myNormalize(Ymod - Xmod_mean)

        test_0 = time.time() # START TO COUNT TIME
        #==== SPARSE CODING by LASSO FISTA ====
        t_i_1 = time.time()
        iter_tr = 0
        Atr2 = np.zeros((d,n))
        for idx in range(n):
            s = Xmod[:,idx].reshape((M,1))
            alpha, iter_cnt, _ = lasso_fista(s, PSI, np.array([]), args.lambda1, fista_opts )
            Atr2[:,idx] = alpha.reshape(d)
            iter_tr += iter_cnt
        t_i_2 = time.time()
        iter_tt = 0
        Att2 = np.zeros((d,nt))
        for idx in range(nt):
            s = Ymod[:,idx].reshape((M,1))
            if idx == 100 and cost:
                alpha, iter_cnt, cost_info = lasso_fista(s, PSI, np.array([]), args.lambda1, fista_opts_2 )
            else:
                alpha, iter_cnt, _ = lasso_fista(s, PSI, np.array([]), args.lambda1, fista_opts )
            Att2[:,idx] = alpha.reshape(d)
            iter_tt += iter_cnt
        t_i_3 = time.time()
        if cost:
            cost_mtx = {}
            cost_mtx['cost_info'] = cost_info
            sio.savemat('cost.mat', cost_mtx)

        #==== PREDICT by LINEAR REGRESSION ====
        modelOutTrain = np.zeros((nClasses, n))
        modelOutTest  = np.zeros((nClasses, nt))
        trainPred = np.zeros(n)
        testPred  = np.zeros(nt)

        modelOutTrain[:,:] = np.matmul(W.T, Atr2) + np.tile(b, (1, n))
        modelOutTest[:,:]  = np.matmul(W.T, Att2) + np.tile(b, (1, nt))
        trainPred[:] = np.argmax(modelOutTrain, axis = 0)
        testPred[:] = np.argmax(modelOutTest, axis = 0)

        test_1 = time.time()
        trainAcc2 = float(np.sum(trainPred == (trls - 1))) / n * 100
        testAcc2 = float(np.sum(testPred == (ttls - 1))) / nt * 100

        accMtx[i,:] = [trainAcc2, testAcc2]
        TestTimeList[i,:] = [test_1 - test_0]
        LASSOTimeList[i,:] = [t_i_2 - t_i_1, t_i_3 - t_i_2]
        LASSOIterList[i,:] = [iter_tr, iter_tt]

        #==== TESTING PROCESS BAR ====
        str0 = progress_str(i+1, args.testNum, 50)
        sys.stdout.write("\r%s %.2f%%" % (str0, ((i+1)*100.0)/args.testNum))
        sys.stdout.flush()

    print ("\nThe result of CA_TDDL: ")
    accMean = np.mean(accMtx, axis = 0)
    accStd = np.std(accMtx, axis = 0)
    print (np.array_str(accMean, precision=2, suppress_small=True))
    print (np.array_str(accStd,  precision=3, suppress_small=True))

    print ("\n==== TIMING INFORMATION ====")
    print ("--- Overall Time")
    print ("Training Time: {:4.3f}".format(train_time))
    test_time = np.sum(TestTimeList) * 1000
    test_time_aver = test_time /(n+nt) /testNum
    print ("Testing Time: {:4.3f}/{:4.3f}".format(test_time, test_time_aver))
    #print (np.array_str(TestTimeList*1000, precision=3, suppress_small=True))

    print ("--- LASSO FISTA Time")
    LASSO_time = np.sum(LASSOTimeList, axis = 0) # get the average of each column
    LASSO_iter = np.sum(LASSOIterList, axis = 0) # get the average of each column
    lasso_train_time = LASSO_time[0] / LASSO_iter[0]
    lasso_test_time = LASSO_time[1] / LASSO_iter[1]
    lasso_time_mean = np.sum(LASSO_time) / (n+nt) / testNum
    lasso_iter_mean = np.sum(LASSO_iter) / (n+nt) / testNum
    lasso = lasso_time_mean / lasso_iter_mean
    print ("Train: {:4.3f}; Test: {:4.3f}".format(lasso_train_time * 1e6, lasso_test_time * 1e6))
    print ("Total Time: {:4.3f}; Total Iter: {:.2f}; Time/Iter: {:4.3f}".format(lasso_time_mean * 1e6, lasso_iter_mean, lasso * 1e6))
    #print (np.array_str(LASSOTimeList, precision=3, suppress_small=True))
    #print (np.array_str(LASSOIterList, precision=3, suppress_small=True))
    print ("--- TDDL Time")
    print ("Init Time: {:.3f}; FISTA Time: {:.3f}".format(time_info[0], time_info[1]))
    print ("Gradient Time: {:.3f}; Update Time: {:.3f}".format(time_info[2], time_info[3]))

if __name__ == '__main__':
    main()
