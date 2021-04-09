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



def RNN_model(x,y, input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps):

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


def sample_freq(data,sf):
    k=[]
    for i in range(0,len(data),sf):
        s=data[i]
        k.append(s)
    return np.array(k)

def sample_freq1(data,sf):
    k=[]
    for i in range(sf,len(data),sf):
        k.append(data[i-sf:i])
    s=np.reshape(k,(len(k),sf))
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

def Get_Cummulative(num):
    l=[]
    l.append(num[0])
    for i in range(len(num)-1):
        g=l[i]+num[i+1]
        l.append(g)
    return (np.array(l))

def maxmin17(dat_to_norm,sf, Acc1_bias,gyro1_bias):
    locPred=np.array(dat_to_norm)
    dist1=locPred[:,2:3]
    dist2=locPred[:,3:4]
    rl=locPred[:,12:13]
    rr=locPred[:,13:14]
    rr=sample_freq(rr,sf)
    rl=sample_freq(rl,sf)    
    dist11=sample_freq(dist1,sf)
    dist21=sample_freq(dist2,sf)    
    dista=absolute_disp(dist11, dist21)
    dist=dista*1000  
    r1=np.mean((dist/rr[1:]), axis=0)
    r2=np.mean((dist/rl[1:]), axis=0)    
    return max(rr), min(rr), r1, r2

def data_process13t(dat, seq_di, input_di, output_di,sf, Acc1_bia, Acc2_bia, gyro1_bia, batch_siz, amx, amn, r1, r2,  Z, mode):
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

        rr_=sample_freq(rr,sf)
        rl_=sample_freq(rl,sf)

        rr_=rr_*r1
        rl_=rl_*r2         

        dispt1=(rl_+rr_)/2  
        rr=normalise(rr[1:], amx, amn, Z)
        rl=normalise(rl[1:], amx, amn, Z)
        fr=normalise(fr[1:], amx, amn, Z)
        fl=normalise(fl[1:], amx, amn, Z)
       
        rr=sample_freq1(rr,sf)
        rl=sample_freq1(rl,sf)
        fr=sample_freq1(fr,sf)
        fl=sample_freq1(fl,sf)
        ak=sample_freq1(ak,sf)
        xcn=np.concatenate((rr[:len(dist)], rl[:len(dist)], fr[:len(dist)], fl[:len(dist)]), axis=1)  

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
    print(labeltitle)
    if labeltitle=='CRSE evolution for the 180 s GNSS Outage (V_Vfb02g)':
        print('yay!')
        outage=outage+1
        plt.plot(np.array(np.arange(outage)),np.concatenate((np.zeros((1,1)),s),axis=0), label=labels)#.format(**)+'INS')
        plt.ylabel(labely)
        plt.xlabel(labelx)
        plt.plot(np.array(np.arange(outage)),np.concatenate((np.zeros((1,1)),t),axis=0), label=labelt)#np.array([1,2,3,4,5,6,7,8,9,10])
        plt.legend()
        plt.grid(b=True)
        plt.xlim(0,181)
        labeltitle='CRSE evolution for the 180 s GNSS Outage (V_Vfb02g)'
        plt.title(labeltitle, fontdict={'fontsize': 15, 'fontweight': 500}, loc='center')
        labeltitle1=('A Positional CRSE evolution for the 180 s GNSS Outage (V_Vfb02g)')
        plt.savefig(labeltitle1+ str(no))
        plt.show() 
    elif labeltitle=='CTE evolution for the 180 s GNSS Outage (V_Vfb02g)':
        print('nay!')
        outage=outage+1
        plt.plot(np.array(np.arange(outage)),np.concatenate((np.zeros((1,1)),s),axis=0), label=labels)#.format(**)+'INS')
        plt.ylabel(labely)
        plt.xlabel(labelx)
        plt.plot(np.array(np.arange(outage)),np.concatenate((np.zeros((1,1)),t),axis=0), label=labelt)#np.array([1,2,3,4,5,6,7,8,9,10])
        plt.legend()
        plt.grid(b=True)
        plt.xlim(0,181)
        labeltitle='CTE evolution for the 180 s GNSS Outage (V_Vfb02g)'
        plt.title(labeltitle, fontdict={'fontsize': 15, 'fontweight': 500}, loc='center')
        labeltitle1=('A Positional CTE evolution for the 180 s GNSS Outage (V_Vfb02g)')
        plt.savefig(labeltitle1+ str(no))
        plt.show() 
           
    
def get_crse(x,y,z,t, label, mode,no):#x=gps, y=ins, z=pred
#    print(t)
    eins=np.sqrt(np.power(y,2))
    epred=np.sqrt(np.power(x-z,2))
    crse_ins=Get_Cummulative(eins[:t])
    crse_pred=Get_Cummulative(epred[:t])
#    if t==180:
#        get_graph(np.reshape(crse_ins[:180],(180,1)), np.reshape(crse_pred[:180],(180,1)), 'Physical Model', 'WhONet', 'Time (s)', 'CRSE (m)', 'CRSE evolution for the ' +label,no, t)
    return crse_ins[-1], crse_pred[-1]

def get_cte(x,y,z,t, label, mode,no):
    eins=y
    epred=x-z
    caeins=Get_Cummulative(eins[:t])
    caepred=Get_Cummulative(epred[:t])   
#    if t==180:
#        get_graph(np.reshape(caeins[:180],(180,1)), np.reshape(caepred[:180],(180,1)), 'Physical Model', 'WhONet', 'Time (s)', 'CTE (m)', 'CTE evolution for the ' +label,no, t)
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
        cae_ins, cae_pred=get_cte(yyTest1,trav-ins,newP,int(outage/Ts), label, mode,w)
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

