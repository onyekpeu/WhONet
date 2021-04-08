# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 03:29:06 2020

@author: onyekpeu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 30 04:44:25 2020

@author: onyekpeu
"""

import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, History, LearningRateScheduler
from tensorflow.keras import regularizers
# from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers 
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from vincenty import vincenty
import tensorflow as tf
import abc
import math
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops

def GRU_model(x,y, input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps):
    start=time.time()
    regressor = Sequential()
    regressor.add(GRU(units =h2,input_shape = (x.shape[1], input_dim), activation="tanh", recurrent_activation="softmax", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_) , return_sequences = False))
#    regressor.add(Dropout(dropout))
#    regressor.add(GRU(units = h2, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_)))
    adamax=optimizers.Adam(lr=learning_rate)#, beta_1=0.9, beta_2=0.99)     
    regressor.add(Dense(units = output_dim, activation='linear'))
    regressor.compile(optimizer = adamax, loss = 'mean_absolute_error')
    def exp_decay(epoch):
        lrate=learning_rate*np.exp(-decay_rate*epoch)
        return lrate
    lr_rt=LearningRateScheduler(exp_decay)
    loss_history=History()
    callbacks_list=[loss_history, lr_rt]    
    print(regressor.summary())
    history = regressor.fit(x, y, epochs = num_epochs, callbacks=callbacks_list, batch_size = batch_size, validation_split=0.005) #iterates 50 times    
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.savefig('GRU_LOSS'+ str(nfr))
    plt.show()
    end=time.time()
    Computation_time=end-start
    return Computation_time, regressor

def RNN_model(x,y, input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps):
#, kernel_initializer="glorot_uniform"
    start=time.time()
    regressor = Sequential()
    regressor.add(SimpleRNN(units =h2,input_shape = (x.shape[1], input_dim), activation="tanh", use_bias=True, recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_) , return_sequences = False))
#    regressor.add(Dropout(dropout))
#    regressor.add(SimpleRNN(units = h2, activation="tanh", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_)))
    adamax=optimizers.Adam(lr=learning_rate)#, beta_1=0.9, beta_2=0.99)     
    regressor.add(Dense(units = output_dim, activation='linear'))
    regressor.compile(optimizer = adamax, loss = 'mean_absolute_error')
    def exp_decay(epoch):
        lrate=learning_rate*np.exp(-decay_rate*epoch)
        return lrate
    lr_rt=LearningRateScheduler(exp_decay)
    loss_history=History()
    callbacks_list=[loss_history, lr_rt]    
    print(regressor.summary())
    history = regressor.fit(x, y, epochs = num_epochs, callbacks=callbacks_list, batch_size = batch_size, validation_split=0.005) #iterates 50 times    
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.savefig('RNN_LOSS'+ str(nfr))
    plt.show()
    end=time.time()
    Computation_time=end-start
    return Computation_time, regressor
#    regressor.add(LSTM(units =h2,input_shape = (x.shape[1], input_dim), activation="tanh", recurrent_activation="softmax", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=0, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_) , return_sequences = True))
#    regressor.add(Dropout(dropout))
#    regressor.add(LSTM(units = h2, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=0, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_), return_sequences = True))
#    regressor.add(Dropout(dropout))
#kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_) , 
def LSTM_model(x,y, input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps):
    start=time.time()
    regressor = Sequential()
    regressor.add(LSTM(units =h2,input_shape = (x.shape[1], input_dim), activation="tanh", recurrent_activation="softmax", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_) , return_sequences = False))
#    regressor.add(Dropout(dropout))
#    regressor.add(LSTM(units = h2, activation="tanh", recurrent_activation="softmax", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=0, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_)))
    adamax=optimizers.Adam(lr=learning_rate)#, beta_1=0.9, beta_2=0.99)     
    regressor.add(Dense(units = output_dim, activation='linear'))
    regressor.compile(optimizer = adamax, loss = 'mean_absolute_error')
    def exp_decay(epoch):
        lrate=learning_rate*np.exp(-decay_rate*epoch)
        return lrate
    lr_rt=LearningRateScheduler(exp_decay)
    loss_history=History()
    callbacks_list=[loss_history, lr_rt]    
    print(regressor.summary())
    history = regressor.fit(x, y, epochs = num_epochs, callbacks=callbacks_list, batch_size = batch_size, validation_split=0.005) #iterates 50 times    
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.savefig('LSTM_LOSS'+ str(nfr))
    plt.show()
    end=time.time()
    Computation_time=end-start
    return Computation_time, regressor

def BGRU_model(x,y, input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps):
    start=time.time()
    regressor = Sequential()
    regressor.add(Bidirectional(GRU(units =h2,input_shape = (x.shape[1], input_dim), activation="tanh", recurrent_activation="softmax", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_) , return_sequences = True)))
    regressor.add(Dropout(dropout))
    regressor.add(Bidirectional(GRU(units = h2, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_))))
    adamax=optimizers.Adamax(lr=learning_rate)#, beta_1=0.9, beta_2=0.99)     
    regressor.add(Dense(units = output_dim, activation='linear'))
    regressor.compile(optimizer = adamax, loss = 'mean_absolute_error')
    def exp_decay(epoch):
        lrate=learning_rate*np.exp(-decay_rate*epoch)
        return lrate
    lr_rt=LearningRateScheduler(exp_decay)
    loss_history=History()
    callbacks_list=[loss_history, lr_rt]    
    
    history = regressor.fit(x, y, epochs = num_epochs, callbacks=callbacks_list, batch_size = batch_size, validation_split=0.005) #iterates 50 times    
    print(regressor.summary())
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.savefig('BGRU_LOSS'+ str(nfr))
    plt.show()
    end=time.time()
    Computation_time=end-start
    return Computation_time, regressor

def BLSTM_model(x,y, input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps):
    start=time.time()
    regressor = Sequential()
    regressor.add(Bidirectional(LSTM(units =h2,input_shape = (x.shape[1], input_dim), activation="tanh", recurrent_activation="softmax", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_) , return_sequences = True)))
    regressor.add(Dropout(dropout))
    regressor.add(Bidirectional(LSTM(units = h2, activation="tanh", recurrent_activation="sigmoid", use_bias=True, kernel_initializer="glorot_uniform", recurrent_initializer="orthogonal", recurrent_dropout=dropout, kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_))))
    adamax=optimizers.Adamax(lr=learning_rate)#, beta_1=0.9, beta_2=0.99)     
    regressor.add(Dense(units = output_dim, activation='linear'))
    regressor.compile(optimizer = adamax, loss = 'mean_absolute_error')
    def exp_decay(epoch):
        lrate=learning_rate*np.exp(-decay_rate*epoch)
        return lrate
    lr_rt=LearningRateScheduler(exp_decay)
    loss_history=History()
    callbacks_list=[loss_history, lr_rt]    
    
    history = regressor.fit(x, y, epochs = num_epochs, callbacks=callbacks_list, batch_size = batch_size, validation_split=0.005) #iterates 50 times    
    print(regressor.summary())
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.savefig('BLSTM_LOSS'+ str(nfr))
    plt.show()
    end=time.time()
    Computation_time=end-start
    return Computation_time, regressor

def IDNN_model(x,y, input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps):
    start=time.time()
    regressor = Sequential()
    regressor.add(Dense(units =h2, kernel_initializer = 'glorot_uniform', activation = 'tanh', input_dim=seq_dim*input_dim))
#    regressor.add(Dropout(dropout))  
#    regressor.add(Dense(units = h2, activation = 'relu'))
#    regressor.add(Dense(units = h2, activation = 'linear'))
    adamax=optimizers.Adamax(lr=learning_rate)#, beta_1=0.9, beta_2=0.99)     
    regressor.add(Dense(units = output_dim, activation='linear'))
    regressor.compile(optimizer = adamax, loss = 'mean_absolute_error')
    def exp_decay(epoch):
        lrate=learning_rate*np.exp(-decay_rate*epoch)
        return lrate
    lr_rt=LearningRateScheduler(exp_decay)
    loss_history=History()
    callbacks_list=[loss_history, lr_rt]    
    print(regressor.summary())
    history = regressor.fit(x, y, epochs = num_epochs, callbacks=callbacks_list, batch_size = batch_size, validation_split=0.005) #iterates 50 times    
#    history = regressor.fit(x, y, epochs = num_epochs,callbacks=callbacks_list, batch_size = batch_size, validation_split=0.005) #iterates 50 times    
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.savefig('RNN_LOSS'+ str(nfr))
    plt.show()
    end=time.time()
    Computation_time=end-start
    return Computation_time, regressor

def MLNN_model(x,y, input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, NFR, decay_rate, momentum, decay_steps):
    start=time.time()
    regressor = Sequential()
    regressor.add(Dense(units =h2, kernel_initializer = 'glorot_uniform', activation = 'tanh', kernel_regularizer=regularizers.l1_l2(l1=l1_, l2=l2_), input_dim=input_dim))
    regressor.add(Dropout(dropout)) 
    regressor.add(Dense(units = h2, kernel_initializer = 'glorot_uniform', activation='tanh'))
#    regressor.add(Dropout(dropout)) 
#    regressor.add(Dense(units = h2, activation='tanh'))    
#    lr_rt=ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)
    adamax=optimizers.Adamax(lr=learning_rate, beta_1=0.9, beta_2=0.99) 
#    sgd=optimizers.SGD(lr=learning_rate)#, momentum=momentum)#, decay=decay_rate)      
    regressor.add(Dense(units = output_dim, activation='linear'))
    regressor.compile(optimizer = adamax, loss = 'mean_absolute_error')
    def exp_decay(epoch):
        lrate=learning_rate*np.exp(-decay_rate*epoch)
        return lrate
    lr_rt=LearningRateScheduler(exp_decay)
    loss_history=History()
    callbacks_list=[loss_history, lr_rt]
#    callbacks=callbacks_list
    print(regressor.summary())
    history = regressor.fit(x, y, epochs = num_epochs,callbacks=callbacks_list,  batch_size = batch_size, validation_split=0.005) #iterates 50 times    
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.show()
    end=time.time()
    Computation_time=end-start
    return Computation_time, regressor

def gps_clean(data, vmax):
    for i in range(len(data)-1):
        if (data[i]>vmax or data[i]<-vmax) and (data[i+1]>vmax or data[i+1]<-vmax) :
            data[i]=data[i-1]
        elif (data[i]>vmax or data[i]<-vmax) and (data[i+1]<vmax or data[i+1]>-vmax):
            data[i]=data[i+1]
    return data


def seq_data_man(data, batch_size, seq_dim, input_dim, output_dim):
    X,Y,Z,A=data
    X=np.array(X)
    Y=np.array(Y)
    Z=np.array(Z)
    A=np.array(A)
    lx=len(X)
    x = []
    y = []
    z=[]
    a=[]
    for i in range(seq_dim,lx):
        x.append(X[i-seq_dim:i, 0:(input_dim)])
        y.append(Y[i-1, 0:output_dim])
        z.append(Z[i-1, 0:output_dim])
        a.append(A[i-1, 0:output_dim])
    x, y, z, a = np.array(x), np.array(y), np.array(z), np.array(a)
    return (x, y, z, a)
#def seq_data_man1(data, batch_size, seq_dim, input_dim, output_dim):
#    X,Y,Z,A=data
#    X=np.array(X)
#    Y=np.array(Y)
#    Z=np.array(Z)
#    A=np.array(A)
#    lx=len(X)
#    x = []
#    y = []
#    z=[]
#    a=[]
#    for i in range(seq_dim,lx):
#        x.append(X[i-seq_dim:i, 0:(input_dim)])
#        y.append(Y[i-1, 0:output_dim])
#        z.append(Z[i-1, 0:output_dim])
#        a.append(A[i-1, 0:output_dim])
#    x, y, z, a = np.array(x), np.array(y), np.array(z), np.array(a)
#    return (x, y, z, a)

def sample_freq(data,sf):
    k=[]
    for i in range(0,len(data),sf):
        s=data[i]
        k.append(s)
    return np.array(k)

def sample_freq1(data,sf):
    k=[]
#    x=[]
#    print(len(data))
    for i in range(sf,len(data),sf):
#        print (i)
#        print(k.shape)
        k.append(data[i-sf:i])
    s=np.reshape(k,(len(k),sf))
#    print(s.shape)
    return s

def calib1(data1):
    locPred=np.array(data1)
    Acc1=locPred[:,16:17]
    Acc2=locPred[:,17:18]
    gyro1=locPred[:,14:15] 
    Brkpr=locPred[:,26:27] 
    Acc1_bias=np.mean(Acc1,axis=0)
    Acc2_bias=np.mean(Acc2,axis=0)
    gyro1_bias=np.mean(gyro1,axis=0)
    Brkpr_bias=np.mean(Brkpr,axis=0)
    return Acc1_bias, Acc2_bias, gyro1_bias, Brkpr_bias#, sa_bias, sa1_bias, sa2_bias

def normalise(T1, Tmx, Tmn,Z):
    return (Z*(T1-Tmn))/(Tmx-Tmn)

def inv_normalise(N, Tmx, Tmn, Z):
    return (N*(Tmx-Tmn)/Z)+Tmn

def absolute_disp(lat, long):
    k=[]
    for i in range(1, len(lat)):
        lat1=lat[i-1]
        lat2=lat[i]
        lon1=long[i-1]
        lon2=long[i]
        kk=vincenty((lat1,lon1), (lat2, lon2))
        k.append(kk)
    return np.reshape(k,(len(k),1))

def integrate(data, sf):
    dx=sf/10
    arr=[]
    for i in range(1,len(data)):
        y=data[i-2:i]
        intg=np.trapz(y, dx=dx, axis=0)
        arr.append(intg)
    return np.reshape(arr,(len(arr),1))

def head_clean(angle):
    for i in range(1,len(angle)):
        if (angle[i]/angle[i-1])>180 or (angle[i-1]/angle[i])>180:
            angle[i]=angle[i-1]
    return angle

def Get_Cummulative(num):
    l=[]

    l.append(num[0])
    for i in range(len(num)-1):
        g=l[i]+num[i+1]
        l.append(g)
    return (np.array(l))

def diff(data):
    x=[]
    for i in range(1,len(data)):
        a=data[i]-data[i-1]
        x.append(a)
    return np.reshape(x,(len(x),1))
def clean(data):
    value=40
    for i in range(1,len(data)):
        if data[i]>value:
            data[i]=value
        elif data[i]<-value:
            data[i]=-value
    return np.array(data)

            
def get_average(data,avg):
    x=[]
    data=data*1000
    for i in range(avg, len(data)+avg, avg):
        a=(np.sum(data[i-avg:i]))/avg
        x.append(a)
    return (np.reshape(x,(len(x),1)))/1000

def average_vel(init, vel):
    a=(vel[0]+init)/2
    k=np.zeros(len(vel))

    k[0]=a#.append(a)
    for i in range(1,len(vel)):
        a=(vel[i]+vel[i-1])/2
        k[1]=a
    return np.reshape(k,(len(k),1))


def maxmin17(dat_to_norm,sf, Acc1_bias,gyro1_bias):
    locPred=np.array(dat_to_norm)
    dist1=locPred[:,2:3]
    dist2=locPred[:,3:4]
    rl=locPred[:,12:13]
    rr=locPred[:,13:14]
    dist1=locPred[:,2:3]
    dist2=locPred[:,3:4]
    ak=locPred[:,28:29]
    rr=sample_freq(rr,sf)
    rl=sample_freq(rl,sf)    
    dist11=sample_freq(dist1,sf)
    dist21=sample_freq(dist2,sf)    
    dista=absolute_disp(dist11, dist21)
    dist=dista*1000  
    r1=np.mean((dist/rr[1:]), axis=0)
    r2=np.mean((dist/rl[1:]), axis=0)  
    rr_=rr*r1
    rl_=rl*r2   
    return max(rr), min(rr), r1, r2, max(dist), min(dist), max(ak), min(ak),

def error_control_mech(data1, data2):
    s=data1-data2
    # s=np.sqrt(s**2)
    return min(s), max(s), np.mean(s)
    

def data_process13t(dat, seq_di, input_di, output_di,sf, Acc1_bia, Acc2_bia, gyro1_bia, batch_siz, amx, amn, r1, r2, dgmx, dgmn, gymx, gymn, Z, mode):

    if mode=='IDNN':
    
        Xin=np.zeros((1,input_di*seq_di))
        Yin=np.zeros((1,output_di))
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
    elif mode=='MLNN':
        Xin=np.zeros((1,input_di))
        Yin=np.zeros((1,output_di)) 
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
    else:
        Xin=np.zeros((1,seq_di, input_di))
        Yin=np.zeros((1,output_di))
        Zin=np.zeros((1,output_di))
        Ain=np.zeros((1,output_di))
        
    for i in range(len(dat)):
        locPred=np.array(dat[i])
        dist1=locPred[:,2:3]
        dist2=locPred[:,3:4]

        rl=locPred[:,12:13]
        rr=locPred[:,13:14]
        fl=locPred[:,10:11]
        fr=locPred[:,11:12]
        ak=locPred[:,28:29]
        


        

        
        
        dist11=sample_freq(dist1,sf)
        dist21=sample_freq(dist2,sf)
        dista=absolute_disp(dist11, dist21)
        dist=dista*1000
#x=np.array([0,0,0,0,0,0,2,1,2])
#y=np.array([5,5,5,5,5,5,5,5,5])
#i=0
#z='inf'
#while z=='inf':
#    z=y[i]/x[i]
#    i=i+1
#    z=str(z)
#print(z)
#print (i)
#        r1_='nan'
#        r2_='nan'
#        iii=0
#        while (r1_=='inf' or r1=='nan') and (r2_=='inf' or r2=='nan'):
#            r1_=dist[iii]/rr[iii]
#            r2_=dist[iii]/rl[iii]
#            iii=iii+1
#            r1=r1_
#            r2=r2_
#            r1_=str(float(r1_))
#            r2_=str(float(r2_))
#        r1=dist[1]/rr[1]#np.mean((dist/rr[1]), axis=0)
#        r2=dist[1]/rr[1]#np.mean((dist/rl[1]), axis=0)  
        rr_=sample_freq(rr,sf)
        rl_=sample_freq(rl,sf)
        fr_=sample_freq(fr,sf)
        fl_=sample_freq(fl,sf)
#        ak_=sample_freq(ak,sf)
        rr_=rr_*r1
        rl_=rl_*r2         

#        print(r1)
#        rr_=rr*r1#*0.0910
#        rl_=rl*r2#*0.0900
        dispt1=(rl_+rr_)/2  

#        
#        rr=diff(rr)
#        rl=diff(rl)
#        fr=diff(fr)
#        fl=diff(fl)        
#        rr=normalise(rr, amx, amn, Z)
#        rl=normalise(rl, amx, amn, Z)
#        fr=normalise(fr, amx, amn, Z)
#        fl=normalise(fl, amx, amn, Z)
#        ddd=normalise(dispt1, dgmx, dgmn, Z)

#        
        rr=normalise(rr[1:], amx, amn, Z)
        rl=normalise(rl[1:], amx, amn, Z)
        fr=normalise(fr[1:], amx, amn, Z)
        fl=normalise(fl[1:], amx, amn, Z)
        ak=normalise(ak[1:], gymx, gymn, Z)
        
        rr=sample_freq1(rr,sf)
        rl=sample_freq1(rl,sf)
        fr=sample_freq1(fr,sf)
        fl=sample_freq1(fl,sf)
        ak=sample_freq1(ak,sf)
#        xcn=np.concatenate((rr, rl), axis=1)  
#        print(ak.shape)        
        xcn=np.concatenate((rr[:len(dist)], rl[:len(dist)], fr[:len(dist)], fl[:len(dist)]), axis=1)  
#        xcn=np.concatenate((rr, rl,fr,fl, ddd[1:]), axis=1) 
#        print(xcn.shape)
        dx3=xcn#dispt1
        # xox=Get_Cummulative(fr)
        # xoy=Get_Cummulative(dist)
        # xoz=Get_Cummulative(dispt1[1:])
        # asd1,asd2,asd3=error_control_mech(xoy, xox)
        # # print(max(diff(dispt1[1:])), max(diff(dist)))
        # print(asd1,asd2,asd3)
        # plt.plot(xox, label='wheel encoder')
        # plt.plot(xoz, label='wheel displacement')
        # plt.plot(xoy, label='GPS')
        # plt.legend()
        # plt.show()
        x, y, z, a_a=seq_data_man((xcn, dist-dispt1[1:], dispt1[1:], dist), batch_siz, seq_di, input_di, output_di)
       
        if mode=='MLNN':
            Xin=np.concatenate((Xin,xcn[seq_di+1:]), axis=0) 
            Yin=np.concatenate((Yin,dist[seq_di:]), axis=0)
            Zin=np.concatenate((Zin,dispt1[seq_di+1:]), axis=0) 
            Ain=np.concatenate((Ain,dist[seq_di:]), axis=0) 
        elif mode=='IDNN':
            x=np.reshape(x,(len(x),seq_di*input_di))
            Xin=np.concatenate((Xin,x), axis=0) 
            Yin=np.concatenate((Yin,y), axis=0)
            Zin=np.concatenate((Zin,z), axis=0) 
            Ain=np.concatenate((Ain,a_a), axis=0) 
        else:
            Xin=np.concatenate((Xin,x), axis=0) 
            Yin=np.concatenate((Yin,y), axis=0) 
            Zin=np.concatenate((Zin,z), axis=0)  
            Ain=np.concatenate((Ain,a_a), axis=0) 
        Input_data=Xin[1:] 
        Output_data=Yin[1:]
        INS=Zin[1:]
        GPS=Ain[1:]
    return  GPS, INS,  Input_data, Output_data
   
def get_graph(s,t, labels, labelt, labelx, labely, labeltitle,no, outage):#s, ins, t=pred
    plt.plot(np.array(np.arange(outage)),np.concatenate((np.zeros((1,1)),s)), label=labels)#.format(**)+'INS')
    plt.ylabel(labely)
    plt.xlabel(labelx)
    plt.plot(np.array(np.arange(outage)),np.concatenate((np.zeros((1,1)),t)), label=labelt)#np.array([1,2,3,4,5,6,7,8,9,10])
    plt.legend()
    plt.grid(b=True)
#    plt.ylim(0, )
    plt.xlim(0,len(s)+1)
    print(len(labeltitle))
    plt.title(labeltitle, fontdict={'fontsize': 15, 'fontweight': 500}, loc='center')
    if len(labeltitle)==88:
        labeltitle=('Displacement CRSE for the Sharp Cornering and Successive Left and Right Turns Scenario')
    elif len(labeltitle)==87:
        labeltitle=('Displacement CAE for the Sharp Cornering and Successive Left and Right Turns Scenario')
    if len(labeltitle)==66:
        labeltitle=('Displacement CRSE for the Quick Changes in Acceleration Scenario')
    elif len(labeltitle)==65:
        labeltitle=('Displacement CAE for the Quick Changes in Acceleration Scenario')    
    plt.savefig(labeltitle+ str(no))
    plt.show()  
    
def get_crse(x,y,z,t, label, mode,no):#x=gps, y=ins, z=pred

    eins=np.sqrt(np.power(y,2))
    epred=np.sqrt(np.power(x-z,2))
    crse_ins=Get_Cummulative(eins[:t])
    crse_pred=Get_Cummulative(epred[:t])
#    get_graph(crse_ins, crse_pred, 'INS DR', mode, 'Time (s)', 'CRSE (m)', 'Displacement CRSE for the ' +label,no, t)
    return crse_ins[-1], crse_pred[-1]

def get_cae(x,y,z,t, label, mode,no):
    eins=y
    epred=x-z
    caeins=Get_Cummulative(eins[:t])
    caepred=Get_Cummulative(epred[:t])   
#    get_graph(caeins, caepred, 'INS DR', mode, 'Time (s)', 'CAE (m)', 'Displacement CAE for the ' +label,no, t)
    return np.sqrt(np.power(caeins[-1],2)), np.sqrt(np.power(caepred[-1],2))         

def get_perfmetric(cet, cetp):
    mean=np.mean(cet, axis=0)
    mini=np.amin(cet, axis=0) 
    stdv=np.std(cet, axis=0)
    maxi=np.amax(cet, axis=0)
    meanp=np.mean(cetp, axis=0)
    minip=np.amin(cetp, axis=0) 
    stdvp=np.std(cetp, axis=0) 
    maxip=np.amax(cetp, axis=0)  
    perf_metr=np.concatenate((np.reshape(maxi,(1,1)),np.reshape(mini,(1,1)), np.reshape(mean,(1,1)), np.reshape(stdv,(1,1))), axis=1)#, np.reshape(np.sum(mabe[:(100/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(300/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(600/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(900/Ts)]),(1,1))), axis=1)#, np.reshape(a90,(1,1)), np.reshape(a60,(1,1)), np.reshape(a30,(1,1)), np.reshape(a10),(1,1)), axis=1)   
    perf_metrp=np.concatenate((np.reshape(maxip,(1,1)),np.reshape(minip,(1,1)), np.reshape(meanp,(1,1)), np.reshape(stdvp,(1,1))), axis=1)#, np.reshape(np.sum(mabe[:(100/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(300/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(600/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(900/Ts)]),(1,1))), axis=1)#, np.reshape(a90,(1,1)), np.reshape(a60,(1,1)), np.reshape(a30,(1,1)), np.reshape(a10),(1,1)), axis=1)   
    return perf_metr, perf_metrp
def get_dist_covrd(grth):
    mean=np.mean(grth, axis=0)
    mini=np.amin(grth, axis=0) 
    stdv=np.std(grth, axis=0)
    maxi=np.amax(grth, axis=0)
    perf_metr=np.concatenate((np.reshape(maxi,(1,1)),np.reshape(mini,(1,1)), np.reshape(mean,(1,1)), np.reshape(stdv,(1,1))), axis=1)#, np.reshape(np.sum(mabe[:(100/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(300/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(600/Ts)]),(1,1)), np.reshape(np.sum(mabe[:(900/Ts)]),(1,1))), axis=1)#, np.reshape(a90,(1,1)), np.reshape(a60,(1,1)), np.reshape(a30,(1,1)), np.reshape(a10),(1,1)), axis=1)   
    return perf_metr     
#get_image(x,y)
def get_image(x,y,no):
        plt.plot(x, label='pred')#.format(**)+'INS')
#        plt.ylabel(labely)
#        plt.xlabel(labelx)
        plt.plot(y, label='errorgpspred')#np.array([1,2,3,4,5,6,7,8,9,10])
        plt.legend()
        plt.grid(b=True) 
        plt.savefig('meetingimage'+ str(no))
        plt.show()    
        
def predictcs(xthr,ythr, ithr, gthr, regress,  seq_dim,input_di, mode, Ts, mx, mn, Z, label, outage):
    xthr=xthr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    ythr=ythr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    ithr=ithr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    gthr=gthr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    crset=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    crsetwe=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    caet=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    caetwe=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    aepst=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    aepstwe=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    dist_covrd=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    cccPred=np.zeros((int(outage/Ts),1)) 
    cccins=np.zeros((int(outage/Ts),1)) 
    cccgps=np.zeros((int(outage/Ts),1)) 
     
        
    for w in range (0,len(xthr),int(outage/Ts)):
        xtest=xthr[w:w+int(outage/Ts)]     
        ytest=ythr[w:w+int(outage/Ts)]
        ins=ithr[w:w+int(outage/Ts)]
        trav=gthr[w:w+int(outage/Ts)]
        yyTest1=np.array(ytest)
        newP=regress.predict(xtest)
        crse_ins, crse_pred=get_crse(yyTest1,trav-ins,newP,int(outage/Ts), label, mode,w)
        cae_ins, cae_pred=get_cae(yyTest1,trav-ins,newP,int(outage/Ts), label, mode,w)
        aeps_ins, aeps_pred=crse_ins/int(outage/Ts), crse_pred/int(outage/Ts) #get_aeps(yyTest1,ins,newP,int(outage/Ts), label, mode,w, int(outage/Ts))
        crset[int(w/int(outage/Ts)),0]=float(crse_pred)
        caet[int(w/int(outage/Ts)),0]=float(cae_pred)
        aepst[int(w/int(outage/Ts)),0]=float(aeps_pred)
        crsetwe[int(w/int(outage/Ts)),0]=float(crse_ins)
        caetwe[int(w/int(outage/Ts)),0]=float(cae_ins)
        aepstwe[int(w/int(outage/Ts)),0]=float(aeps_ins)
        dist_covrd[int(w/int(outage/Ts)),0]=float(sum(np.sqrt(np.power(trav,2))))
        newPreds=np.concatenate((cccPred, np.reshape(newP,(len(newP),1))),axis=1)
        cccPred=newPreds
        INS=np.concatenate((cccins, np.reshape(ins,(len(newP),1))),axis=1)
        cccins=INS  
        GPS=np.concatenate((cccgps, np.reshape(ytest,(len(newP),1))),axis=1)
        cccgps=GPS 
#        get_image(ytest,newP,)
#        print(len(trav))
#        print(len(ytest))
        

        
    perf_metr_crsep, perf_metr_crsewe=get_perfmetric(crset, crsetwe)
    perf_metr_caep, perf_metr_caewe=get_perfmetric(caet, caetwe)
    perf_metr_aepsp, perf_metr_aepswe=get_perfmetric(aepst, aepstwe)
    dist_travld=get_dist_covrd(dist_covrd)
    return dist_travld, perf_metr_crsep, perf_metr_crsewe, perf_metr_caep, perf_metr_caewe, perf_metr_aepsp, perf_metr_aepswe, newPreds[:,1:], INS[:,1:], GPS[:,1:]#, cet, cetp#opt_runs, cm_runs, cm_runsPred 

def predicttrlr(num_epochs, batch_size,xtrainhr,ytrainhr,xthr,ythr, ithr, gthr, regressorr,  seq_dim,input_di, mode, Ts, mx, mn, Z, label, outage):
    xthr=xthr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    ythr=ythr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    ithr=ithr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    gthr=gthr[:int((np.floor(len(xthr)/(int(outage/Ts)))*(int(outage/Ts))))]
    crset=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    crsetwe=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    caet=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    caetwe=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    aepst=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    aepstwe=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    dist_covrd=np.zeros((int(len(xthr)/int(outage/Ts)),1))
    cccPred=np.zeros((int(outage/Ts),1)) 
    cccins=np.zeros((int(outage/Ts),1)) 
    cccgps=np.zeros((int(outage/Ts),1)) 
    history = regressorr.fit(xtrainhr,ytrainhr, epochs = num_epochs, batch_size = batch_size) #iterates 50 times    
    plt.plot(history.history['loss'], label='train')
    plt.legend()
#    plt.savefig('GRU_LOSS'+ str(nfr))
    plt.show()
           
    for w in range (0,len(xthr),int(outage/Ts)):
        xtest=xthr[w:w+int(outage/Ts)]     
        ytest=ythr[w:w+int(outage/Ts)]
        ins=ithr[w:w+int(outage/Ts)]
        trav=gthr[w:w+int(outage/Ts)]
        yyTest1=np.array(ytest)
        newP=regressorr.predict(xtest)
        crse_ins, crse_pred=get_crse(yyTest1,trav-ins,newP,int(outage/Ts), label, mode,w)
        cae_ins, cae_pred=get_cae(yyTest1,trav-ins,newP,int(outage/Ts), label, mode,w)
        aeps_ins, aeps_pred=crse_ins/int(outage/Ts), crse_pred/int(outage/Ts) #get_aeps(yyTest1,ins,newP,int(outage/Ts), label, mode,w, int(outage/Ts))
        crset[int(w/int(outage/Ts)),0]=float(crse_pred)
        caet[int(w/int(outage/Ts)),0]=float(cae_pred)
        aepst[int(w/int(outage/Ts)),0]=float(aeps_pred)
        crsetwe[int(w/int(outage/Ts)),0]=float(crse_ins)
        caetwe[int(w/int(outage/Ts)),0]=float(cae_ins)
        aepstwe[int(w/int(outage/Ts)),0]=float(aeps_ins)
        dist_covrd[int(w/int(outage/Ts)),0]=float(sum(np.sqrt(np.power(trav,2))))
        newPreds=np.concatenate((cccPred, np.reshape(newP,(len(newP),1))),axis=1)
        cccPred=newPreds
        INS=np.concatenate((cccins, np.reshape(ins,(len(newP),1))),axis=1)
        cccins=INS  
        GPS=np.concatenate((cccgps, np.reshape(ytest,(len(newP),1))),axis=1)
        cccgps=GPS   
    perf_metr_crsep, perf_metr_crsewe=get_perfmetric(crset, crsetwe)
    perf_metr_caep, perf_metr_caewe=get_perfmetric(caet, caetwe)
    perf_metr_aepsp, perf_metr_aepswe=get_perfmetric(aepst, aepstwe)
    dist_travld=get_dist_covrd(dist_covrd)
    return dist_travld, perf_metr_crsep, perf_metr_crsewe, perf_metr_caep, perf_metr_caewe, perf_metr_aepsp, perf_metr_aepswe, newPreds[:,1:], INS[:,1:], GPS[:,1:]#, cet, cetp#opt_runs, cm_runs, cm_runsPred 

def get_zeros(no):
    zm=np.zeros((int(no),4))
    return zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm#zm, zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm, zm, zm 

#zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm, zm,zm, zm, zm, zm, zm, zm, zm, zm, zm
