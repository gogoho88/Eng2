#!/usr/bin/env python
# coding: utf-8

# ### Peoplecounting based on Stacked Sparse Denoising AutoEncoder (SSDAE) + Bidirectional LSTM
# #### Made by Jae-Ho Choi, REMS(Radar and ElectoMagnetic Signal processing Lab), EE, POSTECH.
# 
# Author: Jaeho Choi
# 
# Date Created: Jun. 20, 2019
# 
# Date Last Modified: Dec. 27, 2019


## v1->v2 설명
# 네트워크 파라티머 save하는 부분도 추가

## v2-v3 설명
# Input Dataset을 Preprocessed signal받아서 수행하도록 설정(Dataset_LG, Dataset_Atlas_ver...)
# Preprocessed(MATLAB)에서 이미 signal cut(coupling) 수행하였으므로 python에서 수행할 필요 없음
# MF, Denoise(Wavelet), AGC에 따른 결과 차이 확인하기 위해 코드 전반적으로 수정
# In[1]:


# Import
import numpy as np
import h5py # for load v7.3 mat file
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

tf.reset_default_graph() # 텐서들 전부 flush
# In[2]:


# functions


def get_minibatch(data_X, data_Y, batch_size):
    """
    Description:
    Using the preprocessed image datset, this function samples
    randomly according to mini batch size."""    
    data_size = np.shape(data_X)
    
    ind_sample = random.sample(range(0,data_size[0]), batch_size)
    X_sample = data_X[ind_sample,:,:,:]
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
filepath_data = 'D://7. Research/1. People Counting/2. People Counting based on DNN/0. Data/Dataset_LG.mat' # Signal data path

f_i = 0 # 0 일시 Wavelet_denoise x, 1은 level 2, 2는 level 4, 3은 level 6
f_j = 0 # 0일시 corr x, 1일시 corr o
f = h5py.File(filepath_data, 'r')
mat_signal = np.array(f[f['Dataset_LG_total'][0][0]])



#mat_data = sio.loadmat(filepath_data)
#mat_signal = mat_data['signal_LG_ncor']
#mat_signal = np.array(mat_signal)


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

# In[4]: Preprocessing

#
for n in range(len_data): # 0부터 10명까지
    d1_sig_temp = mat_signal[n,:]
    num_people = int(d1_sig_temp[-1])   # signal 맨 끝에 있는 사람 수 확인   
    d1_sig = d1_sig_temp[0:-1]          # 사람 수 제외한 진짜 signal 형성 
    
    data_total[num_people].append(d1_sig)   # 각 사람 수 tag 별로 signal 정리

# frame 형성
for n in range(num_people+1):
    d2_sig_temp = data_total[n]         # 사람 수에 해당하는 dataset signal 선택
    d2_sig_temp = np.array(d2_sig_temp) 
    
    #min-max normalization (0~1 사이로)
    for p in range(np.shape(d2_sig_temp)[0]):
        d1_temp = d2_sig_temp[p,:]
        d1_norm = minmax_scale(d1_temp)
        d2_sig_temp[p,:] = d1_norm
    
    #AGC도 여기에 추가
    
    
    ####
    
    (size_x,size_y) = np.shape(d2_sig_temp)
    size_dataset_temp = np.floor((size_x-frame_len)/(frame_len-frame_overlap))+1    # frame 길이 및 overlap 고려하여 dataset 길이 결정
    size_dataset_temp = int(size_dataset_temp)
    
    d2_dataset_temp = np.empty((size_dataset_temp, frame_len, size_y, 1), dtype=np.float32)
    
    # DNN Input 형태에 맞게 dataset 조정
    for m in range(size_dataset_temp):
        d2_frame = d2_sig_temp[(frame_len-frame_overlap)*m : (frame_len-frame_overlap)*m+(frame_len)]   # pulse를 묶어 frame 형성
        d2_frame_resize = np.resize(d2_frame,[frame_len, size_y, 1])
        
        d2_dataset_temp[m,:,:,:] = d2_frame_resize
    
    
    data_total_resize[n] = d2_dataset_temp      # 각 사람 수 별로 형성
        
    
# In[5]:   
# Preprocessed dataset으로부터 training,  test dataset(one-hot encoding) 형성
Flag_last = 1;  # train,test를 random으로 선택할지, 혹은 맨 마지막 pulse로부터 선택할지

train_ratio = 0.8
val_ratio = 0.0
test_ratio = 0.2
X_train = np.array([])
X_test = np.array([])
X_val = np.array([])
y_train = np.array([])
y_test = np.array([])
y_val = np.array([])

for n in range(num_people+1):
    d2_temp = data_total_resize[n]
    d1_target = np.zeros((np.shape(d2_temp)[0],num_people+1), dtype=np.uint8)
    d1_target[:,n] = 1  # vector one-hot encoding
    
    # 맨 뒤에서부터 sampling
    if Flag_last==1:
        s1_temp = np.shape(d2_temp)[0]
        ind_flag = np.floor(s1_temp*0.8)
        ind_flag = int(ind_flag)
        
        X_train_temp = d2_temp[0:ind_flag,:,:,:]
        X_test_temp = d2_temp[ind_flag:,:,:,:]
        y_train_temp = d1_target[0:ind_flag,:]
        y_test_temp = d1_target[ind_flag:,:]
    
    # Random sampling
    else:
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
X_train_size = np.shape(X_train)
X_val_size = np.shape(X_val)
X_train_AE = X_train.reshape(X_train_size[0]*X_train_size[1],X_train_size[2])    
X_val_AE = X_val.reshape(X_val_size[0]*X_val_size[1],X_val_size[2])

[s1,s2,s3,s4] = np.shape(X_train)

# In[7]: Stacked autoencoder

###### hyperparameters
reg_parameter = 0.0002 # 0으로할 시 no regularization
droprate_input = 0 # 0으로할 시 no drop-out, 즉 몇 %를 drop시킬건지
droprate_hidden = 0 # 0으로할 시 no drop-out
Learning_rate = 0.0002
noise_level = 1.0

EPOCH = 800
MINIBATCH_SIZE = 200


class Model_DSAE:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        
    def _build_net(self,num_hidden,droprate,noise_level):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
            self.is_training = tf.placeholder(tf.bool)
            activation = tf.nn.elu
            weight_initializer = tf.keras.initializers.he_normal()
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=reg_parameter)
                     
            n_inputs = s3
            n_hidden1 = num_hidden[0]   # 1st layer
            n_hidden2 = num_hidden[1]   # 2nd layer
            n_hidden3 = num_hidden[2]   # 3rd layer
            n_hidden4 = n_hidden2       # decoder
            n_hidden5 = n_hidden1       # decoder
            n_hidden6 = n_inputs        # decoder
            
            ## Network initialization
            W1_init = weight_initializer([n_inputs, n_hidden1])
            W2_init = weight_initializer([n_hidden1, n_hidden2])
            W3_init = weight_initializer([n_hidden2, n_hidden3])
            
            # input place holders
            self.W1 = tf.Variable(W1_init, dtype=tf.float32, name='W1')
            self.W2 = tf.Variable(W2_init, dtype=tf.float32, name='W2')
            self.W3 = tf.Variable(W3_init, dtype=tf.float32, name='W3')
            
            ## Decoder weights
            W4 = tf.transpose(self.W3)
            W5 = tf.transpose(self.W2)
            W6 = tf.transpose(self.W1)
            
            ## Biases
            self.B1 = tf.Variable(tf.truncated_normal(shape=[n_hidden1]), dtype=tf.float32, name='B1')
            self.B2 = tf.Variable(tf.truncated_normal(shape=[n_hidden2]), dtype=tf.float32, name='B2')
            self.B3 = tf.Variable(tf.truncated_normal(shape=[n_hidden3]), dtype=tf.float32, name='B3')
            self.B4 = tf.Variable(tf.truncated_normal(shape=[n_hidden4]), dtype=tf.float32, name='B4')
            self.B5 = tf.Variable(tf.truncated_normal(shape=[n_hidden5]), dtype=tf.float32, name='B5')
            self.B6 = tf.Variable(tf.truncated_normal(shape=[n_hidden6]), dtype=tf.float32, name='B6')

            # Network Input
            self.X = tf.placeholder(tf.float32, shape=[None, n_inputs])
            
#            self.Node_input = self.X
            X_noisy = self.X + noise_level*tf.abs(tf.random_normal(tf.shape(self.X)))
            Node_input_drop = tf.layers.dropout(inputs=X_noisy, rate=droprate[0], training=self.is_training)    
            
            Node_hidden1 = activation(tf.matmul(Node_input_drop,self.W1)+self.B1)
            Node_hidden1_drop = tf.layers.dropout(inputs=Node_hidden1, rate=droprate[1], training=self.is_training)
            
            Node_hidden2 = activation(tf.matmul(Node_hidden1_drop,self.W2)+self.B2)
            Node_hidden2_drop = tf.layers.dropout(inputs=Node_hidden2, rate=droprate[1], training=self.is_training)
            
            Node_hidden3 = activation(tf.matmul(Node_hidden2_drop,self.W3)+self.B3)
            
            Node_hidden4 = activation(tf.matmul(Node_hidden3,W4)+self.B4)
            
            Node_hidden5 = activation(tf.matmul(Node_hidden4,W5)+self.B5)
            
            self.Node_output = tf.matmul(Node_hidden5,W6)+self.B6            
     

        # define cost/loss & optimizer
        with tf.variable_scope(self.name):
            reconstruction_loss = tf.losses.mean_squared_error(labels = self.X, predictions = self.Node_output)
            reg_loss = l2_regularizer(self.W1) + l2_regularizer(self.W2) + l2_regularizer(self.W3)
#            self.Loss = reconstruction_loss 
            self.Loss = reconstruction_loss + reg_loss
#            
            self.optimizer = tf.train.AdamOptimizer(Learning_rate).minimize(self.Loss)
            
        self.train_vars = {'W1': self.W1, 'W2': self.W2, 'W3': self.W3,
                           'B1': self.B1, 'B2': self.B2, 'B3': self.B3,
                           'B4': self.B4, 'B5': self.B5, 'B6': self.B6}
        
        for key, var in self.train_vars.items():
            tf.add_to_collection(key, var)
       
        
    def predict(self, x_test, is_training=False):
        return self.sess.run(self.Node_output, feed_dict={self.X: x_test, self.is_training: is_training})

    def test(self, x_test, is_training=False):
        return self.sess.run(self.Loss, feed_dict={self.X: x_test, self.is_training: is_training})

    def train(self, x_training, is_training=True):
        return self.sess.run([self.Loss, self.optimizer], feed_dict={self.X: x_training, self.is_training: is_training})







sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

d1_node_hidden1 = np.array([600,250,100])
d1_node_hidden2 = np.array([600,300,150])
d1_node_hidden3 = np.array([800,400,200])
d1_node_hidden4 = np.array([900,600,300])
droprate1 = np.array([0,0])
droprate2 = np.array([0.25,0.25])
droprate3 = np.array([0.5,0.5])
d1_noiselevel = np.array([0.2,0.1,0.05,0.01])

list_hidden = [d1_node_hidden1,d1_node_hidden2,d1_node_hidden3,d1_node_hidden4]
list_drop = [droprate1,droprate2,droprate3]
list_noiselevel = d1_noiselevel

models = []
num_models = len(list_hidden)*len(list_drop)*len(list_noiselevel)

#for i in range(len(list_hidden)):
#    for j in range(len(list_drop)):
#        for k in range(len(list_noiselevel)):
#            m = len(list_drop)*len(list_noiselevel)*i + len(list_noiselevel)*j + k
#            models.append(Model_DSAE(sess, "model" + str(m)))

for m in range(num_models):
    models.append(Model_DSAE(sess, "model" + str(m)))

# network 생성
for m_idx, m in enumerate(models):
    i = int(m_idx/(len(list_drop)*len(list_noiselevel)))
    j = int((m_idx%(len(list_drop)*len(list_noiselevel)))/len(list_noiselevel))
    k = m_idx%len(list_noiselevel)
    m._build_net(list_hidden[i],list_drop[j],list_noiselevel[k])    # parameter에 따른 network 형성

        
        
sess.run(tf.global_variables_initializer())

print('Learning Started!')
# train
for i in range(EPOCH):
    d1_avg_cost = np.zeros(len(models))
    d1_avg_cost_test = np.zeros(len(models))
    
    total_batch = int(np.shape(X_train_AE)[0]/MINIBATCH_SIZE)
    
    for j in range(total_batch):
        batch_xs = get_minibatch_AE(X_train_AE,MINIBATCH_SIZE)
        
        #train each model
        for m_idx,m in enumerate(models):              
            c,_ = m.train(batch_xs)
            d1_avg_cost[m_idx] += c/total_batch
            
    for m_idx, m in enumerate(models):
        c_test = m.test(X_val_AE)
        d1_avg_cost_test[m_idx] += c_test
            
    print('[Training] Epoch:', '%04d' % (i +1), 'cost =', d1_avg_cost)
    print('[Test] Epoch:', '%04d' % (i +1), 'cost =', d1_avg_cost_test)
    
print('Learning Finished!')

# Test model and check accuracy

# saver
MODEL_PATH = './model/'
saver = tf.train.Saver()
save_path = saver.save(sess,os.path.join(MODEL_PATH, 'trained_AE_v1.ckpt'))  
