# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 01:33:35 2019

@author: owner
"""

import numpy as np
import h5py # for load .mat file
from sklearn.preprocessing import minmax_scale # for minmax normalization
from skimage.transform import resize # for data image resize
import matplotlib.pyplot as plt # for figure plot
from sklearn.metrics import confusion_matrix # for making confusion matrix
import itertools # for plot confusion matrix
import random # for random shuffle for minibatch sampling
import tensorflow as tf # for network
import time # for training time and test time memorization
import matplotlib.pyplot as plt
import os

import scipy.io as sio
from sklearn.model_selection import train_test_split # for train,test,split
# In[2]:


# functions


def get_minibatch_RNN(data_X, data_Y, batch_size):
    """
    Description:
    Using the preprocessed image datset, this function samples
    randomly according to mini batch size."""    
    data_size = np.shape(data_X)
    
    ind_sample = random.sample(range(0,data_size[0]), batch_size)
    X_sample = data_X[ind_sample,:,:]
    Y_sample = data_Y[ind_sample,:]
    
    return X_sample, Y_sample

def get_minibatch_AE(data_X, batch_size):
    """
    Description:
    Using the preprocessed image datset, this function samples
    randomly according to mini batch size."""    
    data_size = np.shape(data_X)
    
    ind_sample = random.sample(range(0,data_size[0]), batch_size)
    X_sample = data_X[ind_sample,:]
    
    return X_sample

def k1_divergence(p,q):
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))

# In[3]:


# Data load
filepath_data = 'D://7. Research/1. People Counting/2. People Counting based on DNN/0. Data/signal_LG_ncor.mat' # Signal data path

#f_rawsignal = h5py.File(filepath_data, 'r')
#data_training = f_rawsignal.get('signal_LG_ncor')
#data_training = np.array(data_training) # Change training data to np.array type

mat_data = sio.loadmat(filepath_data)
mat_signal = mat_data['signal_LG_ncor']
mat_signal = np.array(mat_signal)


# 
# variables
num_people = 10
data_total = [[],[],[],[],[],[],[],[],[],[],[]]
data_total_resize = [[],[],[],[],[],[],[],[],[],[],[]]

sig_min = 200
sig_max = 1480

frame_len = 25
frame_overlap = 15

rand_seed = 42 # random number seed

len_data = np.shape(mat_signal)[0]

# In[4]:

#
for n in range(len_data): # 0부터 10명까지
    d1_sig_temp = mat_signal[n,:]
    num_people = int(d1_sig_temp[-1])
    d1_sig = d1_sig_temp[sig_min-1:sig_max-1]
    
    data_total[num_people].append(d1_sig)

# frame 형성
for n in range(num_people+1):
    d2_sig_temp = data_total[n]
    d2_sig_temp = np.array(d2_sig_temp)
    
    (size_x,size_y) = np.shape(d2_sig_temp)
    size_dataset_temp = np.floor((size_x-frame_len)/(frame_len-frame_overlap))+1
    size_dataset_temp = int(size_dataset_temp)
    
    d2_dataset_temp = np.empty((size_dataset_temp, frame_len, size_y, 1), dtype=np.float32)
    
    for m in range(size_dataset_temp):
        d2_frame = d2_sig_temp[(frame_len-frame_overlap)*m : (frame_len-frame_overlap)*m+(frame_len)]
        d2_frame_resize = np.resize(d2_frame,[frame_len, size_y, 1])
        
        d2_dataset_temp[m,:,:,:] = d2_frame_resize
        
    data_total_resize[n] = d2_dataset_temp
        
    
# In[5]:   
# training,  test dataset(one-hot encoding) 형성
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
X_train = np.array([])
X_test = np.array([])
X_val = np.array([])
y_train = np.array([])
y_test = np.array([])
y_val = np.array([])

for n in range(num_people+1):
    d2_temp = data_total_resize[n]
    d1_target = np.zeros((np.shape(d2_temp)[0],num_people+1), dtype=np.uint8)
    d1_target[:,n] = 1
    
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(d2_temp,d1_target, test_size=val_ratio+test_ratio, random_state=42)
    X_val_temp, X_test_temp, y_val_temp, y_test_temp = train_test_split(X_test_temp,y_test_temp, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
    
    if n==0:
        X_train = X_train_temp
        X_test = X_test_temp
        X_val = X_val_temp
        y_train = y_train_temp
        y_test = y_test_temp
        y_val = y_val_temp
    else:
        X_train = np.concatenate((X_train,X_train_temp), axis=0)
        X_test = np.concatenate((X_test,X_test_temp), axis=0)
        X_val = np.concatenate((X_val,X_val_temp), axis=0)
        y_train = np.concatenate((y_train,y_train_temp), axis=0)
        y_test = np.concatenate((y_test,y_test_temp), axis=0)
        y_val = np.concatenate((y_val,y_val_temp), axis=0) 


# In[6]: AE용 train, test
X_train = np.reshape(X_train, [np.shape(X_train)[0], np.shape(X_train)[1], np.shape(X_train)[2]])
X_val = np.reshape(X_val, [np.shape(X_val)[0], np.shape(X_val)[1], np.shape(X_val)[2]])
X_test = np.reshape(X_test, [np.shape(X_test)[0], np.shape(X_test)[1], np.shape(X_test)[2]])


# In[7]: AE용 train, test

tf.reset_default_graph() # 텐서들 전부 flush

MODEL_PATH = './model/'


new_saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH, 'trained_AE_v1.ckpt.meta')) # 저장된 tensor들 load



class Model_BILSTM_AE:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        
    def _build_net(self,num_hidden,noise_level,m_idx):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
            self.is_training = tf.placeholder(tf.bool) # dropout확인 위해 사용
            activation = tf.nn.elu
            weight_initializer = tf.keras.initializers.he_normal()
            
            n_AE_hidden1 = num_hidden[0]
            n_AE_hidden2 = num_hidden[1]
            n_AE_hidden3 = num_hidden[2]
                     
            n_input_steps = 25
            n_input_len = 1280
            n_class = 11
            n_hidden = 80

            
            # input place holders
            self.W1 = tf.get_default_graph().get_collection('W1')[m_idx]
            self.W2 = tf.get_default_graph().get_collection('W2')[m_idx]
            self.W3 = tf.get_default_graph().get_collection('W3')[m_idx]
            
            
            ## Biases
            self.B1 = tf.get_default_graph().get_collection('B1')[m_idx]
            self.B2 = tf.get_default_graph().get_collection('B2')[m_idx]
            self.B3 = tf.get_default_graph().get_collection('B3')[m_idx]


            # Network
            self.X = tf.placeholder(tf.float32, [None, n_input_steps, n_input_len]) # Matrix form data is entered into each BiLSTM input
            X_noisy = self.X + noise_level*tf.random_normal(tf.shape(self.X))
            self.Y = tf.placeholder(tf.int32, [None, n_class])
            
            W_init = weight_initializer([n_hidden*2, n_class])
            self.W = tf.Variable(W_init, dtype=tf.float32)
            self.B = tf.Variable(tf.truncated_normal(shape=[n_class]), dtype=tf.float32)
            
            self.keep_prob = tf.placeholder(tf.float32)
            
            self.W1 = tf.reshape(self.W1,[1,-1,n_AE_hidden1])
            self.W2 = tf.reshape(self.W2,[1,-1,n_AE_hidden2])
            self.W3 = tf.reshape(self.W3,[1,-1,n_AE_hidden3])            
            self.B1 = tf.reshape(self.B1,[1,n_AE_hidden1])
            self.B2 = tf.reshape(self.B2,[1,n_AE_hidden2])
            self.B3 = tf.reshape(self.B3,[1,n_AE_hidden3])
            
            self.W1_stack = tf.tile(self.W1,[n_input_steps,1,1])
            self.W2_stack = tf.tile(self.W2,[n_input_steps,1,1])
            self.W3_stack = tf.tile(self.W3,[n_input_steps,1,1])
            self.B1_stack = tf.tile(self.B1,[n_input_steps,1])
            self.B2_stack = tf.tile(self.B2,[n_input_steps,1])
            self.B3_stack = tf.tile(self.B3,[n_input_steps,1])
            
            for i in range(n_input_steps):
                Node_hidden1 = activation(tf.matmul(X_noisy[:,i,:],self.W1_stack[i,:,:])+self.B1_stack[i,:])
                Node_hidden2 = activation(tf.matmul(Node_hidden1,self.W2_stack[i,:,:])+self.B2_stack[i,:])
                
                Node_hidden2_stop = tf.stop_gradient(Node_hidden2) ### frozen
                
                Node_hidden3 = tf.matmul(Node_hidden2_stop,self.W3_stack[i,:,:])+self.B3_stack[i,:] # activation 떼어야하나?? 애매하다/...
                
                Node_hidden1 = tf.reshape(Node_hidden1, [-1,1,n_AE_hidden1])
                Node_hidden2 = tf.reshape(Node_hidden2, [-1,1,n_AE_hidden2])
                Node_hidden3 = tf.reshape(Node_hidden3, [-1,1,n_AE_hidden3])
                
                if i==0:
                    self.Node_hidden1_stack = Node_hidden1
                    self.Node_hidden2_stack = Node_hidden2
                    self.LSTM_X= Node_hidden3
                else:
                    self.Node_hidden1_stack = tf.concat([self.Node_hidden1_stack,Node_hidden1],axis = 1)
                    self.Node_hidden2_stack = tf.concat([self.Node_hidden2_stack,Node_hidden2],axis = 1)
                    self.LSTM_X = tf.concat([self.LSTM_X,Node_hidden3],axis = 1)
            
            # Bi-LSTM network
#            self.LSTM_X = tf.Variable(self.Node_hidden3_stack, dtype=tf.float32, name='LSTM_X')
            self.LSTM_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True, name='LSTM_fw_cell')
#            self.LSTM_fw_cell = tf.nn.rnn_cell.DropoutWrapper(self.LSTM_fw_cell, output_keep_prob = self.keep_prob)
            
            self.LSTM_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=n_hidden, state_is_tuple=True, name='LSTM_bw_cell')
#            self.LSTM_bw_cell = tf.nn.rnn_cell.DropoutWrapper(self.LSTM_bw_cell, output_keep_prob = self.keep_prob)
            
            self.Node_output,_ = tf.nn.bidirectional_dynamic_rnn(self.LSTM_fw_cell, self.LSTM_bw_cell, self.LSTM_X, dtype = tf.float32)
            
            output_fw = tf.transpose(self.Node_output[0],[1,0,2])
            output_bw = tf.transpose(self.Node_output[1],[1,0,2])
            
            output_concat = tf.concat([output_fw[-1], output_bw[-1]], axis=1)
            
            self.logit = tf.matmul(output_concat,self.W) + self.B
            self.prob = tf.nn.softmax(self.logit)
        
        with tf.variable_scope(self.name):
            self.Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logit, labels = self.Y))
            self.optimizer = tf.train.AdamOptimizer(Learning_rate).minimize(self.Loss)
            correct_prediction = tf.equal(tf.argmax(self.logit, 1), tf.argmax(self.Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                          

#    def temp(self, x_test, drop_rate=0):
#        rate = 1.0 - drop_rate
#        return self.sess.run(self.Node_hidden3_ed, feed_dict={self.X: x_test, self.keep_prob: rate})
    
    def predict(self, x_test, drop_rate=0):
        rate = 1.0 - drop_rate
        return self.sess.run(self.prob, feed_dict={self.X: x_test, self.keep_prob: rate})
    
    def test(self, x_test, y_test, drop_rate=0):
        rate = 1.0 - drop_rate
        return self.sess.run(self.Loss, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: rate})

    def train(self, x_training, y_training, drop_rate):
        rate= 1.0 - drop_rate
        return self.sess.run([self.Loss, self.optimizer], feed_dict={self.X: x_training, self.Y: y_training, self.keep_prob: rate})
    
    def get_accuracy(self, x_test, y_test, drop_rate=0):
        rate = 1.0 - drop_rate
        return self.sess.run(self.accuracy, feed_dict={self.X: x_test, self.Y: y_test, self.keep_prob: rate})
       
## Hyperparameter
Learning_rate = 0.0001
EPOCH = 300
MINIBATCH_SIZE = 200
Drop_rate = 0.5

      
sess = tf.Session()

d1_node_hidden1 = np.array([600,250,100])
list_hidden = [d1_node_hidden1]

#writer = tf.summary.FileWriter("./logs/PC_DNN_SSAERNN_load_v1")
#writer.add_graph(sess.graph)

RNN_models = []
num_models = len(tf.get_default_graph().get_collection('W1'))

for m in range(num_models):
    RNN_models.append(Model_BILSTM_AE(sess, "RNNmodel" + str(m)))

# network 생성
for m_idx, m in enumerate(RNN_models):
    m._build_net(list_hidden[0], 0, m_idx)
    
sess.run(tf.global_variables_initializer())
new_saver.restore(sess, os.path.join(MODEL_PATH, 'trained_AE_v1.ckpt'))

print('Learning Started!')

# train
for i in range(EPOCH):
    d1_avg_cost = np.zeros(len(RNN_models))
    d1_avg_cost_test = np.zeros(len(RNN_models))
    d1_avg_acc = np.zeros(len(RNN_models))
    d1_avg_acc_test = np.zeros(len(RNN_models))
    
    total_batch = int(np.shape(X_train)[0]/MINIBATCH_SIZE)
    
    for j in range(total_batch):
        batch_xs, batch_ys = get_minibatch(X_train, y_train, MINIBATCH_SIZE)
    
    
        #train each model
    for m_idx,m in enumerate(RNN_models):              
        c,_ = m.train(batch_xs, batch_ys, Drop_rate)
        acc = m.get_accuracy(batch_xs, batch_ys)
        d1_avg_cost[m_idx] += c/total_batch
            
    for m_idx, m in enumerate(RNN_models):
        c_test = m.test(X_val, y_val)
        acc = m.get_accuracy(X_val, y_val)
        d1_avg_cost_test[m_idx] += c_test
            
    print('[Training] Epoch:', '%04d' % (i +1), 'cost =', d1_avg_cost)
    print('[Test] Epoch:', '%04d' % (i +1), 'cost =', d1_avg_cost_test)
    print('[Training] Epoch:', '%04d' % (i +1), 'Accuracy =', d1_avg_acc)
    print('[Test] Epoch:', '%04d' % (i +1), 'Accuracy =', d1_avg_acc_test)
    
print('Learning Finished!')







#sess = tf.InteractiveSession()
#new_saver.restore(sess,os.path.join(MODEL_PATH, 'trained_AE_v1.ckpt'))
#
#for op in tf.get_default_graph().get_collection('model1'):
#    print(op.name)
