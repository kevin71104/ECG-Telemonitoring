from utils import *
from numpy import linalg as LA
import scipy.io as sio
import numpy as np

def Indep_PCA(args, normalize = False, semi = True, KNN = False):
    """
    o Learned Principal Component Analysis (PCA) on X
    o Syntax: `(dataMtx, Acc) = Indep_PCA(args, normalize = False, semi = True)`
      - INPUT:
        + `args`
          * `file`     : datapath
          * `compress` : compressed ratio
          * `number`   : data ratio
          * `class`    : number of classes
        + `normalize`  : normalize original data
        + `semi`       : learn PCA on all data
      - OUTPUT:
        + `dataMtx`
          * `PHI`     : sensing matrix
          * `PSI`     : PCA matrix
          * `x_mean`  : mean vector of X_hat_test
          * `T_train` : coefficients of train data on PSI by origin SVD
          * `S_train` : coefficients of train data on PSI by THETA
          * `S_test`  : coefficients of test data on PSI by THETA
          * `trls`    : label of train dta
          * `ttls`    : label of test dta
    -----------------------------------------------
    o Author : Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
               Bo-Hong (Jay) Cho, jaycho2007@gmail.com
               Ching-Yao (Jason) Chou, endpj@access.ntu.edu.tw
      GitHub : https://github.com/kevin71104/ECG-Telemonitoring/tree/master/Eigenspace-aided_Compressed_Analysis/src
      Date   : 2018.06.14
    """
    data = sio.loadmat(args.file)
    X_tmp      = data['X_r_train']
    X_test     = data['X_r_test']
    label_tmp  = data['Y_train'].astype(int)
    label_test = data['Y_test'].astype(int)
    N = X_tmp.shape[0] # 5000
    n = X_tmp.shape[1] # 512
    cr = args.compress
    M = int(cr*n)
    r = 83

    #==== PARTIAL DATA ====
    if args.number != 1.0:
        if semi:
            print('Using all Training Data')
            #==== Signal space learning (PCA): W_pca ====
            from sklearn.decomposition import PCA
            pca = PCA(n_components=r) # PCA will automatically subtract mean
            pca.fit(X_tmp)
            W_pca = pca.components_.T # components_ of size as n_components * n_features (r * n)
            PSI = W_pca
        N_new = int(N * args.number)
        Nc = int(N / args.classes)
        N_new_c = int(N_new / args.classes)
        X_train = np.zeros((N_new,n))
        label_train = np.zeros((N_new,1))
        for c in range(args.classes):
            permut = np.random.permutation(Nc)
            sta = c*N_new_c
            fin = (c+1)*N_new_c
            X_train[sta : fin, :] = X_tmp[permut[:N_new_c] + c*Nc, :]
            label_train[sta : fin, :] = label_tmp[permut[:N_new_c] + c*Nc, :]
        N = N_new
        if not semi:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=r) # PCA will automatically subtract mean
            pca.fit(X_train)
            PSI = pca.components_.T # components_ of size as n_components * n_features (r * n)
    else:
        X_train = X_tmp
        label_train = label_tmp
        from sklearn.decomposition import PCA
        pca = PCA(n_components=r) # PCA will automatically subtract mean
        pca.fit(X_train)
        PSI = pca.components_.T # components_ of size as n_components * n_features (r * n)

    x_mean = X_train.mean(axis=0) # shift to average as original (axis = 0 means get average of each column)
    if args.compress < 1:
        #==== CS DATA ====
        from module import CS_API
        PHI = CS_API.PHI_generate(M, n, 'BERNOULLI')
        PHI = myNormalize(PHI)
        X_hat_train = (X_train - x_mean).dot(PHI.T)
        X_hat_test = (X_test - x_mean).dot(PHI.T)

        #=== DECODING MATRIX ====
        THETA = PHI.dot(PSI)
        D = LA.pinv(THETA)

        #==== DECODING TO EIGENSPACE ====
        T_train = (X_train - x_mean).dot(PSI)
        S_train = X_hat_train.dot(D.T)
        S_test  = X_hat_test.dot(D.T)
    else:
        T_train = (X_train - x_mean).dot(PSI)
        S_train = T_train
        S_test  = (X_test - x_mean).dot(PSI)

    #==== TEST BY KNN ====
    if KNN:
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=1)

        neigh.fit(T_train, label_train.reshape(-1,))
        Y_predict = neigh.predict(T_train)
        acc1 = float(np.sum(Y_predict == label_train.reshape(-1,))) / Y_predict.shape[0]

        Y_predict = neigh.predict(S_test)
        acc2 = float(np.sum(Y_predict == label_test.reshape(-1,))) / Y_predict.shape[0]
        print('Performance of KNN : {:.2%} / {:.2%}'.format(acc1,acc2))
    else:
        acc1 = 0.0
        acc2 = 0.0

    dataMtx = {}
    if args.compress < 1:
        dataMtx['PHI'] = PHI

    dataMtx['PSI'] = PSI
    dataMtx['x_mean'] = x_mean.reshape((-1,1))

    dataMtx['X_train'] = X_train.T
    dataMtx['X_test']  = X_test.T

    dataMtx['T_train'] = T_train.T
    dataMtx['S_train'] = S_train.T
    dataMtx['S_test']  = S_test.T

    label_test[label_test == -1] = 2
    label_train[label_train == -1] = 2
    dataMtx['trls'] = label_train.T
    dataMtx['ttls'] = label_test.T
    sio.savemat('../../data/Indep_PCA_%d.mat'%N,dataMtx)

    return dataMtx, [acc1, acc2]
