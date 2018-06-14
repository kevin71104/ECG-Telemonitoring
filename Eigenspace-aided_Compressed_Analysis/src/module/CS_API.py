from numpy import linalg as LA
from sklearn import preprocessing
import numpy as np

def PHI_generate(m, n, type):
    """
    o Produce sensing matrix used in compressed sensing model
    o Syntax: `PHI = PHI_generate(m, n, type)`
      - INPUT:
        + `m`    : output dimension
        + `n`    : input dimension
        + `type` : type of sensing matrix, PHI
          *  'GAUSSIAN'  : entries from i.i.d. Gaussian(0,1)
          *  'BERNOULLI' : entries from i.i.d. Bernoulli(0.5)
          *  'Uniform'   : entries from i.i.d. Uniform([0,1])
          *  'TOEPLITZ'  : Toeplitz structred
          *  'CIRCULANT' : Circulant structured
      - OUTPUT:
        + `PHI` : sensing matrix of dimension (m,n)
    -----------------------------------------------
    o Author : Ching-Yao (Jason) Chou, endpj@access.ntu.edu.tw
               Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
               Bo-Hong (Jay) Cho, jaycho2007@gmail.com
      GitHub : https://github.com/kevin71104/ECG-Telemonitoring/tree/master/Eigenspace-aided_Compressed_Analysis/src
      Date   : 2018.06.14
    """
    assert m < n, "dimension m > n is not correct!"
    m = int(m)
    if type is 'GAUSSIAN':
        PHI = np.random.randn(m,n)
        
    if type is 'BERNOULLI':
        PHI = np.sign(np.random.rand(m,n) - 0.5)
        PHI[PHI == 0] = 1    
        
    if type is 'UNIFORM':
        PHI = 2*np.random.rand(m,n) - 1
        PHI = preprocessing.normalize(PHI, norm='l2',axis=0)
        
    if type is 'TOEPLITZ': # structured matrix
        PHI = np.zeros((m,n))
        PHI[0,:] = np.random.rand(1,n)
        for i, row in enumerate(PHI[:-1]):
            PHI[i+1] = np.concatenate((np.random.rand(1), PHI[i][:-1]), axis=0)
            
    if type is 'CIRCULANT':
        PHI = np.zeros((m,n))
        PHI[0,:] = np.random.rand(1,n)
        for i, row in enumerate(PHI[:-1]):
            PHI[i+1] = np.roll(PHI[i], 1)
    return PHI

def calPRD(X_rec, X_orig):
    """
    o Calculate Percentage Root-Mean-Square Deviation (PRD)
    o Syntax: `PRD = calPRD(X_rec, X_orig)`
      - INPUT:
        + `X_rec`  : reconstructed data matrix, in dimension of (N,n)
        + `X_orig` : original data matrix, in dimension of (N,n)
      - OUTPUT:
        + `PRD` : Percentage Root-Mean-Square Deviation
    -----------------------------------------------
    o Author: Ching-Yao (Jason) Chou, 
              Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
              Bo-Hong (Jay) Cho, jaycho2007@gmail.com
              github :
    """
    ERR = X_rec - X_orig
    prd = np.average(LA.norm(ERR,axis=1)/LA.norm(X_orig,axis=1))*100
    return prd  