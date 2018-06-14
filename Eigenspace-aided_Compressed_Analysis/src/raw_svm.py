"""
Packages
"""
import argparse
import scipy.io as sio
import numpy as np
from utils import *
from module import CS_API
"""
o Author : Bo-Hong (Jay) Cho, jaycho2007@gmail.com
           Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
           Ching-Yao (Jason) Chou, endpj@access.ntu.edu.tw
  GitHub : https://github.com/kevin71104/ECG-Telemonitoring/tree/master/Eigenspace-aided_Compressed_Analysis/src
  Date   : 2018.06.14
o Script:
  - python3 raw_svm.py <-f datapath> <-c #classes> <-nr data ratio> 
                    <-g Gamma>  <-C C> <-tn #tests> <-cr compressed ratio>
				    [-p PHI] <-m SVM kernel mode> [-v verbose]
  - Ex: python3 python3.6 raw_svm.py -f ../../data/resample.mat -nr 0.1 -c 2 -g 0.2 -C 500 -t 3 -cr 0.25
"""        
def main():
    #==== Arguments ====
    parser = argparse.ArgumentParser(description="task-driven dictionary learning")
    
    parser.add_argument("-f",   "--file", type = str, help = "fileName", default = "../../data/resample.mat")
    parser.add_argument("-c",   "--classes", type = int, help = 'number of classes', default = 2)
    parser.add_argument("-nr",  "--number", type = float, help = 'ratio of train data', default = 0.5)
    parser.add_argument("-g",   "--gamma", type = float, help = 'gamma', default = 0.1)
    parser.add_argument("-C",   "--C", type = float, help = 'C', default = 1000)
    parser.add_argument("-tn", "--time", type = int, help = 'experiment times', default = 3)
    parser.add_argument("-cr", "--cr", type = float, help = 'compress ratio', default = 0.25)
    parser.add_argument("-p", "--phi", action = 'store_true')
    parser.add_argument("-m", "--mode", type=str, help='linear or rgb', default='rgb')
    parser.add_argument("-v", '--verbose', action='store_true')
    args = parser.parse_args()

    #==== data loading ====
    dataDict = loadData(args.file, args.number, args.classes)
    XArr = dataDict['XArr']
    YArr = dataDict['YArr']
    trls = dataDict['trls']
    ttls = dataDict['ttls']
    n = dataDict['n']
    
    if args.phi:
        m = int(n[0] * args.cr)
        #==== PRODUCE CS MATRIX ====
        PHI = CS_API.PHI_generate(m, n[0], 'BERNOULLI')
        PHI = myNormalize(PHI)
        #==== SIMULATE CS DATA ====
        Xmod = np.matmul(PHI, XArr).T
        Ymod = np.matmul(PHI, YArr).T
    else:
        m = int(n[0])
        Xmod = XArr.T
        Ymod = YArr.T
    trls = trls.reshape((-1, ))
    ttls = ttls.reshape((-1, ))
    recordMat = np.zeros((args.time, 3))

    print ("================= SVM baseline Model ==============================")
    print ("Apply SVM on " + args.file[11:] + " with parameters:")
    print ("(Data Ratio) = ({0})".format(args.number))
    print ("(gamma, C) = ({0}, {1})".format(args.gamma, args.C))
    print ("Compressed Ratio = {0}".format(args.cr))
    print ("Number of tests: {0}".format(args.time))
    print ("===================================================================")

    #====== START EXPERIMENT ====
    for i in range(args.time):
        #==== Validation set  ====
        trVal = int(Xmod.shape[0] * 0.2)
        permut = np.random.permutation(Xmod.shape[0])
        X = Xmod[permut, :]
        Y = trls[permut]
        X_train = X[trVal:, :]
        X_val = X[:trVal, :]
        Y_train = Y[trVal:]
        Y_val = Y[:trVal]
        
        from sklearn import svm
        #==== TRAIN ON SVM Kernel ====
        if args.mode == 'rgb':
            clf = svm.SVC(C=args.C, gamma = args.gamma)
            clf.fit(X_train, Y_train)
        #==== TRAIN ON LINEAR SVM ====
        else:
            clf = svm.LinearSVC(C=args.C)
            clf.fit(X_train, Y_train)
        if args.verbose:
            print('# of support vector: %d' %(clf.support_vectors_.shape[0]))
        trainAcc = clf.score(X_train, Y_train)
        valAcc = clf.score(X_val, Y_val)
        testAcc = clf.score(Ymod, ttls)
        #===== RECORD RESULT ====
        recordMat[i, 0] = trainAcc * 100
        recordMat[i, 1] = valAcc * 100
        recordMat[i, 2] = testAcc * 100
        print ('SVM kernel Result, Exeperiment: ', str(i+1))
        print ('Training accuracy : %4.2f %%' %(trainAcc*100))
        print ('Validation accuracy : %4.2f %%' %(valAcc*100))
        print ('Testing  accuracy : %4.2f %%' %(testAcc*100))
    #====== SAVE FILE ======
    mean = np.mean(recordMat, axis = 0)
    std  = np.std(recordMat, axis = 0)
    print("SVM total result...")
    print ('Training accuracy : %4.2f %%' %(mean[0]), 'with std :%4.2f %%' %(std[0]))
    print ('Validation accuracy : %4.2f %%' %(mean[1]), 'with std :%4.2f %%' %(std[1]))
    print ('Testing  accuracy : %4.2f %%' %(mean[2]), 'with std :%4.2f %%' %(std[2]))
    
if __name__ == '__main__':
    main()
