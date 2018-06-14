"""
Packages
"""
import argparse
import scipy.io as sio
import numpy as np
import tensorflow as tf
from utils import *
"""
o Author : Bo-Hong (Jay) Cho, jaycho2007@gmail.com
           Kai-Chieh (Kevin) Hsu, kevin71104@gmail.com
           Ching-Yao (Jason) Chou, endpj@access.ntu.edu.tw
  GitHub : https://github.com/kevin71104/ECG-Telemonitoring/tree/master/Eigenspace-aided_Compressed_Analysis/src
  Date   : 2018.06.14
o Script:
    - python3 raw_nn.py <-f Data path> <-b Batch size> <-nr Number ratio of data> <-cr Compress ratio>
                  <-c # of classes> <-it Iterations> <-l Learning rate> <-m Model> 
                  <-tn # of testing times> [-p on compressed data]
    - Ex. python3 raw_nn.py -b 50 -nr 0.5 -cr 0.25 -it 1500 -m 5 -tn 2 -p
"""
"""
Model constructed by Tensorflow 1.6 API
"""
#====================== MODEL =======================#
class NN(object):
    def __init__(self,batchSize,input_dim,output_dim,learning_rate,model,):
        #===== parameter initialization ====#
        self.batchSize = batchSize
        self.lr = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        #==== in/ out ====
        self.xs = tf.placeholder(dtype = tf.float32, shape = [None, self.input_dim], name = 'xs')
        self.ys = tf.placeholder(dtype = tf.float32, shape = [None, self.output_dim], name = 'ys')
        self.keep_prob = tf.placeholder(dtype = tf.float32, name = 'keep_prob')
        #===== model selection ======#
        if model == 0:
            self.build_net0()
            ## model 0:
            ## (8, 16, 32)
        elif model == 1:
            self.build_net1()
            ## model 1:
            ## (16, 32, 64)
        elif model == 2:
            self.build_net2()
            ## model 2:
            ## (32, 64, 128)
        elif model == 3:
            self.build_net3()
            ## model 3:
            ## (16, 32)
        elif model == 4:
            self.build_net4()
            ## model 4:
            ## (32, 64)
        elif model == 5:
            self.build_net5()
            ## model 5:
            ## (64, 128)
        elif model == 6:
            self.build_net4()
            ## model 6:
            ## (128, 256)
        else:
            print("Error model name Input!!")
        #===== session initialization ======#
        self.sess = tf.Session()
        
    #===== MODEL 0 (8, 16, 32) ======#
    def build_net0(self):
        #==== layer =====
        with tf.variable_scope('fc1'):
            w_fc1 = tf.get_variable('w_fc1', shape = [self.input_dim, 8], )
            b_fc1 = tf.get_variable('b_fc1', shape = [8], )
            h_fc1 = tf.nn.relu(tf.matmul(self.xs, w_fc1) + b_fc1)
            h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob = self.keep_prob)
        with tf.variable_scope('fc2'):
            w_fc2 = tf.get_variable('w_fc2', shape = [8, 16], )
            b_fc2 = tf.get_variable('b_fc2', shape = [16], )
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)
            h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob = self.keep_prob)
        with tf.variable_scope('fc3'):
            w_fc3 = tf.get_variable('w_fc3', shape = [16, 32], )
            b_fc3 = tf.get_variable('b_fc3', shape = [32], )
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2_dropout, w_fc3) + b_fc3)
            h_fc3_dropout = tf.nn.dropout(h_fc3, keep_prob = self.keep_prob)  
        with tf.variable_scope('output'):
            w_fc4 = tf.get_variable('w_fc4', shape = [32, self.output_dim], )
            b_fc4 = tf.get_variable('b_fc4', shape = [self.output_dim], )
            self.pred = tf.nn.softmax(tf.matmul(h_fc3_dropout, w_fc4) + b_fc4)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.pred + 1e-20), reduction_indices=[1]))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    #===== MODEL 1 (16, 32, 64) ======#
    def build_net1(self):
        with tf.variable_scope('fc1'):
            w_fc1 = tf.get_variable('w_fc1', shape = [self.input_dim, 16])
            b_fc1 = tf.get_variable('b_fc1', shape = [16])
            h_fc1 = tf.nn.relu(tf.matmul(self.xs, w_fc1) + b_fc1)
            h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob)
        with tf.variable_scope('fc2'):
            w_fc2 = tf.get_variable('w_fc2', shape = [16, 32], )
            b_fc2 = tf.get_variable('b_fc2', shape = [32], )
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)
            h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob = self.keep_prob)
        with tf.variable_scope('fc3'):
            w_fc3 = tf.get_variable('w_fc3', shape = [32, 64], )
            b_fc3 = tf.get_variable('b_fc3', shape = [64], )
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2_dropout, w_fc3) + b_fc3)
            h_fc3_dropout = tf.nn.dropout(h_fc3, keep_prob = self.keep_prob)  
        with tf.variable_scope('output'):
            w_fc4 = tf.get_variable('w_fc4', shape = [64, self.output_dim], )
            b_fc4 = tf.get_variable('b_fc4', shape = [self.output_dim], )
            self.pred = tf.nn.softmax(tf.matmul(h_fc3_dropout, w_fc4) + b_fc4)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.pred + 1e-20), reduction_indices=[1]))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    #===== MODEL 2 (32, 64, 128) ======#
    def build_net2(self):
        with tf.variable_scope('fc1'):
            w_fc1 = tf.get_variable('w_fc1', shape = [self.input_dim, 32])
            b_fc1 = tf.get_variable('b_fc1', shape = [32])
            h_fc1 = tf.nn.relu(tf.matmul(self.xs, w_fc1) + b_fc1)
            h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob)
        with tf.variable_scope('fc2'):
            w_fc2 = tf.get_variable('w_fc2', shape = [32, 64], )
            b_fc2 = tf.get_variable('b_fc2', shape = [64], )
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)
            h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob = self.keep_prob)
        with tf.variable_scope('fc3'):
            w_fc3 = tf.get_variable('w_fc3', shape = [64, 128], )
            b_fc3 = tf.get_variable('b_fc3', shape = [128], )
            h_fc3 = tf.nn.relu(tf.matmul(h_fc2_dropout, w_fc3) + b_fc3)
            h_fc3_dropout = tf.nn.dropout(h_fc3, keep_prob = self.keep_prob)  
        with tf.variable_scope('output'):
            w_fc4 = tf.get_variable('w_fc4', shape = [128, self.output_dim], )
            b_fc4 = tf.get_variable('b_fc4', shape = [self.output_dim], )
            self.pred = tf.nn.softmax(tf.matmul(h_fc3_dropout, w_fc4) + b_fc4)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.pred + 1e-20), reduction_indices=[1]))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    #===== MODEL 3 (16, 32) ======#
    def build_net3(self):
        with tf.variable_scope('fc1'):
            w_fc1 = tf.get_variable('w_fc1', shape = [self.input_dim, 16])
            b_fc1 = tf.get_variable('b_fc1', shape = [16])
            h_fc1 = tf.nn.relu(tf.matmul(self.xs, w_fc1) + b_fc1)
            h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob)
        with tf.variable_scope('fc2'):
            w_fc2 = tf.get_variable('w_fc2', shape = [16, 32], )
            b_fc2 = tf.get_variable('b_fc2', shape = [32], )
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)
            h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob = self.keep_prob)
        with tf.variable_scope('output'):
            w_fc3 = tf.get_variable('w_fc3', shape = [32, self.output_dim], )
            b_fc3 = tf.get_variable('b_fc3', shape = [self.output_dim], )
            self.pred = tf.nn.softmax(tf.matmul(h_fc2_dropout, w_fc3) + b_fc3)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.pred + 1e-20), reduction_indices=[1]))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    #===== MODEL 4 (32, 64) ======#
    def build_net4(self):
        with tf.variable_scope('fc1'):
            w_fc1 = tf.get_variable('w_fc1', shape = [self.input_dim, 32])
            b_fc1 = tf.get_variable('b_fc1', shape = [32])
            h_fc1 = tf.nn.relu(tf.matmul(self.xs, w_fc1) + b_fc1)
            h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob)
        with tf.variable_scope('fc2'):
            w_fc2 = tf.get_variable('w_fc2', shape = [32, 64], )
            b_fc2 = tf.get_variable('b_fc2', shape = [64], )
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)
            h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob = self.keep_prob)
        with tf.variable_scope('output'):
            w_fc3 = tf.get_variable('w_fc3', shape = [64, self.output_dim], )
            b_fc3 = tf.get_variable('b_fc3', shape = [self.output_dim], )
            self.pred = tf.nn.softmax(tf.matmul(h_fc2_dropout, w_fc3) + b_fc3)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.pred + 1e-20), reduction_indices=[1]))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    #===== MODEL 5 (64, 128) ======#
    def build_net5(self):
        with tf.variable_scope('fc1'):
            w_fc1 = tf.get_variable('w_fc1', shape = [self.input_dim, 64])
            b_fc1 = tf.get_variable('b_fc1', shape = [64])
            h_fc1 = tf.nn.relu(tf.matmul(self.xs, w_fc1) + b_fc1)
            h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob)
        with tf.variable_scope('fc2'):
            w_fc2 = tf.get_variable('w_fc2', shape = [64, 128], )
            b_fc2 = tf.get_variable('b_fc2', shape = [128], )
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)
            h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob = self.keep_prob)
        with tf.variable_scope('output'):
            w_fc3 = tf.get_variable('w_fc3', shape = [128, self.output_dim], )
            b_fc3 = tf.get_variable('b_fc3', shape = [self.output_dim], )
            self.pred = tf.nn.softmax(tf.matmul(h_fc2_dropout, w_fc3) + b_fc3)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.pred + 1e-20), reduction_indices=[1]))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    #===== MODEL 6 (128, 256) ======#
    def build_net6(self):
        with tf.variable_scope('fc1'):
            w_fc1 = tf.get_variable('w_fc1', shape = [self.input_dim, 128])
            b_fc1 = tf.get_variable('b_fc1', shape = [128])
            h_fc1 = tf.nn.relu(tf.matmul(self.xs, w_fc1) + b_fc1)
            h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob=self.keep_prob)
        with tf.variable_scope('fc2'):
            w_fc2 = tf.get_variable('w_fc2', shape = [128, 256], )
            b_fc2 = tf.get_variable('b_fc2', shape = [256], )
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_dropout, w_fc2) + b_fc2)
            h_fc2_dropout = tf.nn.dropout(h_fc2, keep_prob = self.keep_prob)
        with tf.variable_scope('output'):
            w_fc3 = tf.get_variable('w_fc3', shape = [256, self.output_dim], )
            b_fc3 = tf.get_variable('b_fc3', shape = [self.output_dim], )
            self.pred = tf.nn.softmax(tf.matmul(h_fc2_dropout, w_fc3) + b_fc3)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.pred + 1e-20), reduction_indices=[1]))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    #======= TRAIN PROCESS =======#
    def train(self, epoch, data, label, X_val, Y_val, verbose):
        self.sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            for j in range(0, data.shape[0], self.batchSize):
                batch_x = data[j: j+self.batchSize, :]
                batch_y = label[j: j+self.batchSize, :]
                loss, _ = self.sess.run(
                    [self.loss, self.train_op],
                    feed_dict = {
                        self.xs: batch_x,
                        self.ys: batch_y,
                        self.keep_prob: 0.7
                    }
                )
            if i % 10 == 0 and verbose:
                val_acc = self.test(X_val, Y_val)
                print("Iteration:" ,i+1, " training_loss: ", loss)
                print("Validation Accuracy: ", val_acc)
    #======== TEST PROCESS ========#
    def test(self, data, label):
        y_pred = self.sess.run(
            self.pred,
            feed_dict = {
                self.xs: data,
                self.keep_prob: 1
            }
        )
        y_acc = tf.equal(tf.argmax(y_pred, 1), tf.argmax(label, 1))
        acc = tf.reduce_mean(tf.cast(y_acc, tf.float32))
        result = self.sess.run(
            acc, 
            feed_dict = {
                self.xs: data,
                self.ys: label,
                self.keep_prob: 1,
            }
        )
        return result

def main():
    #==== Arguments ====
    parser = argparse.ArgumentParser(description="task-driven dictionary learning")
    
    parser.add_argument("-f",   "--file", type = str, help = "fileName", default = "../../data/resample.mat")
    parser.add_argument("-b",   "--batch", type = int, help = 'batch size', default = 200)
    parser.add_argument("-nr",  "--number", type = float, help = 'ratio of train data', default = 0.5)
    parser.add_argument("-cr", "--ratio", type = float, help = "Compressed Ratio", default = 0.25)
    parser.add_argument("-c",   "--classes", type = int, help = 'number of classes', default = 2)
    parser.add_argument("-it",  "--iteration", type = int, help = 'iteration', default = 1000)
    parser.add_argument("-l",  "--learningRate", type = float, help = 'lr', default = 1e-3)
    parser.add_argument("-m", "--model", type = int, help = "model name", default = 0)
    parser.add_argument("-tn", "--times", type = int, help = 'experiment times', default = 1)
    parser.add_argument("-p", "--phi", action = 'store_true')
    parser.add_argument("-v", "--verbose", action = 'store_true')

    args = parser.parse_args()

    #==== data loading ====
    dataDict = loadData(args.file, args.number, args.classes)
    XArr = dataDict['XArr']
    YArr = dataDict['YArr']
    trls = dataDict['trls']-1
    ttls = dataDict['ttls']-1
    n = dataDict['n']
    nClasses = args.classes
    if args.phi:
        m = int(n[0] * args.ratio)
        #=== sensing data ====
        from module import CS_API
        PHI = CS_API.PHI_generate(m, n[0], 'BERNOULLI')
        PHI = myNormalize(PHI)
        ## X_train, Y_trian = (n_samples, n_features)
        X_train = np.matmul(PHI, XArr).T
        X_test = np.matmul(PHI, YArr).T
    else:
        m = int(n[0])
        X_train = XArr.T
        X_test = YArr.T
    #=== one hot encoding ===
    Y_train = np.zeros((trls.shape[1], nClasses))
    Y_test  = np.zeros((ttls.shape[1], nClasses))
    for index, i in enumerate(trls[0]):
        Y_train[index, i] = 1
    for index, i in enumerate(ttls[0]):
        Y_test[index, i] = 1
    #==== Model initializeing ====
    clf = NN(
        args.batch,
        m,
        args.classes,
        args.learningRate,
        args.model,
    )
    #=== Experiment ====

    print ("================= DNN baseline Model ==============================")
    print ("Apply DNN on " + args.file[11:] + " with parameters:")
    print ("(Data Ratio) = ({0})".format(args.number))
    print ("(lr, iteration, model) = ({0}, {1}, {2})".format(args.learningRate, args.iteration, args.model))
    print ("Compressed Ratio = {0}".format(args.ratio))
    print ("Number of tests: {0}".format(args.times))
    print ("===================================================================")

    recordMat = np.zeros((args.times, 2))
    for i in range(args.times): 
        ## validaiton_split: X_val, X_train; Y_val, Y_train
        permut_train = np.random.permutation(X_train.shape[0])
        permut_test  = np.random.permutation(X_test.shape[0])
        X_train = X_train[permut_train, :]
        Y_train = Y_train[permut_train, :]
        X_test  = X_test[permut_test, :]
        Y_test  = Y_test[permut_test, :]

        trainVal = int(X_train.shape[0] * 0.2)
        X_val = X_train[:trainVal, :]
        X_train_ = X_train[trainVal:, :]
        Y_val = Y_train[:trainVal, :]
        Y_train_ = Y_train[trainVal:, :]
        #==== training ====#
        clf.train(
            args.iteration,
            X_train_,
            Y_train_,
            X_val,
            Y_val,
            verbose = args.verbose
        )
        #===== testing =====#
        trainAcc = clf.test(
            X_train_,
            Y_train_, 
        )
        testAcc = clf.test(
            X_test,
            Y_test,
        )
        #===== result recording =====#
        print("Experiment time: ", str(i+1))
        print("training Accuracy: ", trainAcc)
        print("testing Accuracy: ", testAcc)
        recordMat[i, 0] = trainAcc * 100
        recordMat[i, 1] = testAcc * 100
    
    mean = np.mean(recordMat, axis = 0)
    std  = np.std(recordMat, axis = 0)
    print("NN total result...")
    print ('Training accuracy : %4.2f %%' %(mean[0]), 'with std :%4.2f %%' %(std[0]))
    print ('Training accuracy : %4.2f %%' %(mean[1]), 'with std :%4.2f %%' %(std[1]))
    

if __name__ == '__main__':
    main()
