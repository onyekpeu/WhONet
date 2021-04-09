# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:03:11 2019

@author: onyekpeu
"""
#Best
import numpy as np

from function_files import *
from IO_VNB_Dataset import *
from empty_arrays import *

'Parameters'



new_opt_runsmw=np.zeros((4,len(par)))
new_opt_runsra=np.zeros((8,len(par)))

new_opt_runscia=np.zeros((8,len(par)))
new_opt_runshb=np.zeros((8,len(par)))

new_opt_runsslr=np.zeros((12,len(par)))
new_opt_runswr=np.zeros((12,len(par)))

new_opt_runs030=np.zeros((36,len(par)))
new_opt_runs060=np.zeros((36,len(par)))
new_opt_runs120=np.zeros((36,len(par)))
new_opt_runs180=np.zeros((36,len(par)))



dropout=0.05

input_dim = 4*10
output_dim = 1
num_epochs = 80
layer_dim = 1
learning_rate = 0.0007
batch_size =128
test_split =0
decay_rate=0
decay_steps=10000
momentum=0.8
samplefreq=10
Ts=int(samplefreq*1*1)
seq_dim=int(1*(10/Ts))
seq_dim_=int(seq_dim*Ts)
avg=1
l1_=0
l2_=0
h2 = 72
Z=1
outage1=100
outage2=300
outage3=600
outage4=900
outage5=1200
outage6=1800
number_of_runs=1
splitvalue=0.30# from 5% to 30%
mode='RNN'

  
#############################################################################
#############################################################################
#############################################################################
'Dataloading and indexing'
#############################################################################
#############################################################################
#############################################################################

TrainDat=[V_S1[:1380], V_S1[1580:], V_S3a, V_S2[:19640], V_S2[21360:52960], V_S2[53580:], V_Y1, V_Y2[:481], V_Y2[508:1417],V_Y2[1438:1950],V_Y2[1989:2164],V_Y2[2198:2368],
          V_Y2[2468:2779],V_Y2[2790:], V_St1[:1885], V_St1[1989:2335], V_St1[2482:],
          V_M[:2491], V_M[2511:2734], V_M[2742:], V_S4[:4741], V_S4[4856:], V_S3c[:1260],
           V_S3c[1426:1623], V_S3c[1647:], V_St6, V_Vw16a, V_Vta2, V_Vta1a, V_Vw5, V_Vta8,
          V_Vta10, V_Vta9, V_Vta13, V_Vta16, V_Vta17, V_Vta20, V_Vta21, V_Vta22, V_Vta27, V_Vta28, V_Vta29[:800], V_Vta29[1080:6780],
          V_Vta29[7220:], V_Vta30[:12900], V_Vta30[13180:], V_Vtb1, V_Vtb2, V_Vtb3, V_Vtb5[:1255], V_Vtb5[1267:3720], V_Vtb5[4160:4380], V_Vtb5[4860:6760], 
          V_Vtb5[7220:], V_Vtb9, V_Vw4[:4900], V_Vw4[5760:6220], V_Vw4[7420:33340], V_Vw4[33460:80660], V_Vw4[81000:116180], V_Vw4[117160:], V_Vw14b, V_Vw14c[:14060], V_Vw14c[15600:],
          V_Vfa01, V_Vfa02[:59860], V_Vfa02[59860:], V_Vfb01a[:1520], V_Vfb01a[1980:5360], V_Vfb01a[5740:9360], V_Vfb01a[11660:], V_Vfb01b, V_Vfb02b]
'''Bias Estimation'''
Acc1_bias, Acc2_bias, gyro1_bias, Brkpr_bias=calib1(Bias)


'''Testset'''
#Challenging Scenarios
RAdat1=[V_Vta11]
RAdat2=[V_Vfb02d]

#Quick changes in acceleration scenario
CIAdat1=[V_Vfb02e]
CIAdat2=[V_Vta12]

#Hard brake scenario
HBdat1=[V_Vw16b]
HBdat2=[V_Vw17]

#Sharp cornering and successive left and right scenario
SLRdat1=[V_Vw6]
SLRdat2=[V_Vw8]
SLRdat3=[V_Vw7]  

#Motorway scenario
MWdat=[V_Vw12]

#wet toad scenario
WRdat1=[V_Vtb8]
WRdat2=[V_Vtb11]
WRdat3=[V_Vtb13]

#Longer term Outages
AGdat1=[V_Vtb3]
AGdat2=[V_Vfb01c]
AGdat3=[V_Vfb02a]
AGdat4=[V_Vta1a] 
AGdat5=[V_Vfb02b]
AGdat6=[V_Vfb02g]
DFdat1=[V_St6]
DFdat2=[V_St7[:17000]]
DFdat3=[V_S3a]

amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn= maxmin17(V_Vw12,Ts, Acc1_bias, gyro1_bias)

'''Data processing for training'''
gtr,itr,x, y=data_process13t(TrainDat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)

'''Data processing for evaluation'''
#Motorway Scenario
gtmw,itmw,xtmw, ytmw=data_process13t(MWdat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
#Roundabout Scenario    
gtra1,itra1,xtra1, ytra1=data_process13t(RAdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
gtra2,itra2,xtra2, ytra2=data_process13t(RAdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode,)
#Hard brake Scenario
gthb1,ithb1,xthb1, ythb1=data_process13t(HBdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn,Z, mode) 
gthb2,ithb2,xthb2, ythb2=data_process13t(HBdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn,Z, mode)  
#Quick changes in acceleration Scenario
gtcia1,itcia1,xtcia1, ytcia1=data_process13t(CIAdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode) 
gtcia2,itcia2,xtcia2, ytcia2=data_process13t(CIAdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)       
#Sharp cornering and successive left and right turns Scenario
gtslr1,itslr1,xtslr1, ytslr1=data_process13t(SLRdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)           
gtslr2,itslr2,xtslr2, ytslr2=data_process13t(SLRdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)  
gtslr3,itslr3,xtslr3, ytslr3=data_process13t(SLRdat3, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)       
#Wet Road Scenario
gtwr1,itwr1,xtwr1, ytwr1=data_process13t(WRdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)      
gtwr2,itwr2,xtwr2, ytwr2=data_process13t(WRdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode) 
gtwr3,itwr3,xtwr3, ytwr3=data_process13t(WRdat3, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode) 
 #Longer-term GNSS Outages Scenario   
gtag1,itag1,xtag1, ytag1=data_process13t(AGdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
gtag2,itag2,xtag2, ytag2=data_process13t(AGdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
gtag3,itag3,xtag3, ytag3=data_process13t(AGdat3, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
gtag4,itag4,xtag4, ytag4=data_process13t(AGdat4, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
gtag5,itag5,xtag5, ytag5=data_process13t(AGdat5, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
gtag6,itag6,xtag6, ytag6=data_process13t(AGdat6, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
gtdf1,itdf1,xtdf1, ytdf1=data_process13t(DFdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
gtdf2,itdf2,xtdf2, ytdf2=data_process13t(DFdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)
gtdf3,itdf3,xtdf3, ytdf3=data_process13t(DFdat3, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, dimx, dimn, dgmx, dgmn, gymx, gymn, Z, mode)



'array creation to store maximum NN model CTE for each scenario after each full training'   
cte_runsmwNN=np.zeros((int(number_of_runs),4))
cte_runsra_1NN=np.zeros((int(number_of_runs),4))
cte_runsra_2NN=np.zeros((int(number_of_runs),4))
cte_runscia_1NN=np.zeros((int(number_of_runs),4))
cte_runscia_2NN=np.zeros((int(number_of_runs),4))
cte_runshb_1NN=np.zeros((int(number_of_runs),4))
cte_runshb_2NN=np.zeros((int(number_of_runs),4))        
cte_runsslr_1NN=np.zeros((int(number_of_runs),4)) 
cte_runsslr_2NN=np.zeros((int(number_of_runs),4))     
cte_runsslr_3NN=np.zeros((int(number_of_runs),4)) 
cte_runswr_1NN=np.zeros((int(number_of_runs),4))
cte_runswr_2NN=np.zeros((int(number_of_runs),4))    
cte_runswr_3NN=np.zeros((int(number_of_runs),4))    
cte_runsag30_1NN=np.zeros((int(number_of_runs),4)) 
cte_runsag30_2NN=np.zeros((int(number_of_runs),4)) 
cte_runsag30_3NN=np.zeros((int(number_of_runs),4)) 
cte_runsag30_4NN=np.zeros((int(number_of_runs),4)) 
cte_runsag30_5NN=np.zeros((int(number_of_runs),4))     
cte_runsag30_6NN=np.zeros((int(number_of_runs),4)) 

cte_runsag60_1NN=np.zeros((int(number_of_runs),4)) 
cte_runsag60_2NN=np.zeros((int(number_of_runs),4))
cte_runsag60_3NN=np.zeros((int(number_of_runs),4))
cte_runsag60_4NN=np.zeros((int(number_of_runs),4))
cte_runsag60_5NN=np.zeros((int(number_of_runs),4))    
cte_runsag60_6NN=np.zeros((int(number_of_runs),4))  

cte_runsag120_1NN=np.zeros((int(number_of_runs),4)) 
cte_runsag120_2NN=np.zeros((int(number_of_runs),4)) 
cte_runsag120_3NN=np.zeros((int(number_of_runs),4)) 
cte_runsag120_4NN=np.zeros((int(number_of_runs),4)) 
cte_runsag120_5NN=np.zeros((int(number_of_runs),4))     
cte_runsag120_6NN=np.zeros((int(number_of_runs),4)) 

cte_runsag180_1NN=np.zeros((int(number_of_runs),4)) 
cte_runsag180_2NN=np.zeros((int(number_of_runs),4)) 
cte_runsag180_3NN=np.zeros((int(number_of_runs),4)) 
cte_runsag180_4NN=np.zeros((int(number_of_runs),4)) 
cte_runsag180_5NN=np.zeros((int(number_of_runs),4))     
cte_runsag180_6NN=np.zeros((int(number_of_runs),4))

    
cte_runsdf30_1NN=np.zeros((int(number_of_runs),4))
cte_runsdf30_2NN=np.zeros((int(number_of_runs),4))
cte_runsdf30_3NN=np.zeros((int(number_of_runs),4))

cte_runsdf60_1NN=np.zeros((int(number_of_runs),4))
cte_runsdf60_2NN=np.zeros((int(number_of_runs),4))
cte_runsdf60_3NN=np.zeros((int(number_of_runs),4))

cte_runsdf120_1NN=np.zeros((int(number_of_runs),4))
cte_runsdf120_2NN=np.zeros((int(number_of_runs),4))
cte_runsdf120_3NN=np.zeros((int(number_of_runs),4))

cte_runsdf180_1NN=np.zeros((int(number_of_runs),4))
cte_runsdf180_2NN=np.zeros((int(number_of_runs),4))
cte_runsdf180_3NN=np.zeros((int(number_of_runs),4))

    
'array creation to store maximum INS physical model CTE for each scenario after each full training'
cte_runsmwPHY_MOD=np.zeros((int(number_of_runs),4))

cte_runsra_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsra_2PHY_MOD=np.zeros((int(number_of_runs),4))

cte_runscia_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runscia_2PHY_MOD=np.zeros((int(number_of_runs),4))    

cte_runshb_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runshb_2PHY_MOD=np.zeros((int(number_of_runs),4))

cte_runsslr_1PHY_MOD=np.zeros((int(number_of_runs),4)) 
cte_runsslr_2PHY_MOD=np.zeros((int(number_of_runs),4)) 
cte_runsslr_3PHY_MOD=np.zeros((int(number_of_runs),4)) 

cte_runswr_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runswr_2PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runswr_3PHY_MOD=np.zeros((int(number_of_runs),4))
    
cte_runsag30_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag30_2PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag30_3PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag30_4PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag30_5PHY_MOD=np.zeros((int(number_of_runs),4))    
cte_runsag30_6PHY_MOD=np.zeros((int(number_of_runs),4))

cte_runsag60_1PHY_MOD=np.zeros((int(number_of_runs),4))  
cte_runsag60_2PHY_MOD=np.zeros((int(number_of_runs),4)) 
cte_runsag60_3PHY_MOD=np.zeros((int(number_of_runs),4)) 
cte_runsag60_4PHY_MOD=np.zeros((int(number_of_runs),4)) 
cte_runsag60_5PHY_MOD=np.zeros((int(number_of_runs),4)) 
cte_runsag60_6PHY_MOD=np.zeros((int(number_of_runs),4)) 

cte_runsag120_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag120_2PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag120_3PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag120_4PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag120_5PHY_MOD=np.zeros((int(number_of_runs),4))    
cte_runsag120_6PHY_MOD=np.zeros((int(number_of_runs),4))
   
cte_runsag180_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag180_2PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag180_3PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag180_4PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsag180_5PHY_MOD=np.zeros((int(number_of_runs),4))    
cte_runsag180_6PHY_MOD=np.zeros((int(number_of_runs),4))

cte_runsdf30_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsdf30_2PHY_MOD=np.zeros((int(number_of_runs),4))        
cte_runsdf30_3PHY_MOD=np.zeros((int(number_of_runs),4))  

cte_runsdf60_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsdf60_2PHY_MOD=np.zeros((int(number_of_runs),4))    
cte_runsdf60_3PHY_MOD=np.zeros((int(number_of_runs),4)) 

cte_runsdf120_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsdf120_2PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsdf120_3PHY_MOD=np.zeros((int(number_of_runs),4))

cte_runsdf180_1PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsdf180_2PHY_MOD=np.zeros((int(number_of_runs),4))
cte_runsdf180_3PHY_MOD=np.zeros((int(number_of_runs),4))

'array creation to store maximum NN model CRSE for each scenario after each full training'
           
crse_runsmwNN=np.zeros((int(number_of_runs),4))
crse_runsra_1NN=np.zeros((int(number_of_runs),4))
crse_runsra_2NN=np.zeros((int(number_of_runs),4))
crse_runscia_1NN=np.zeros((int(number_of_runs),4))
crse_runscia_2NN=np.zeros((int(number_of_runs),4))
crse_runshb_1NN=np.zeros((int(number_of_runs),4))
crse_runshb_2NN=np.zeros((int(number_of_runs),4))        
crse_runsslr_1NN=np.zeros((int(number_of_runs),4)) 
crse_runsslr_2NN=np.zeros((int(number_of_runs),4))     
crse_runsslr_3NN=np.zeros((int(number_of_runs),4)) 
crse_runswr_1NN=np.zeros((int(number_of_runs),4))
crse_runswr_2NN=np.zeros((int(number_of_runs),4))    
crse_runswr_3NN=np.zeros((int(number_of_runs),4))    
crse_runsag30_1NN=np.zeros((int(number_of_runs),4)) 
crse_runsag30_2NN=np.zeros((int(number_of_runs),4)) 
crse_runsag30_3NN=np.zeros((int(number_of_runs),4)) 
crse_runsag30_4NN=np.zeros((int(number_of_runs),4)) 
crse_runsag30_5NN=np.zeros((int(number_of_runs),4))     
crse_runsag30_6NN=np.zeros((int(number_of_runs),4))   
   
crse_runsag60_1NN=np.zeros((int(number_of_runs),4)) 
crse_runsag60_2NN=np.zeros((int(number_of_runs),4))
crse_runsag60_3NN=np.zeros((int(number_of_runs),4))
crse_runsag60_4NN=np.zeros((int(number_of_runs),4))
crse_runsag60_5NN=np.zeros((int(number_of_runs),4))    
crse_runsag60_6NN=np.zeros((int(number_of_runs),4))  

crse_runsag120_1NN=np.zeros((int(number_of_runs),4)) 
crse_runsag120_2NN=np.zeros((int(number_of_runs),4)) 
crse_runsag120_3NN=np.zeros((int(number_of_runs),4)) 
crse_runsag120_4NN=np.zeros((int(number_of_runs),4)) 
crse_runsag120_5NN=np.zeros((int(number_of_runs),4))     
crse_runsag120_6NN=np.zeros((int(number_of_runs),4)) 

crse_runsag180_1NN=np.zeros((int(number_of_runs),4)) 
crse_runsag180_2NN=np.zeros((int(number_of_runs),4)) 
crse_runsag180_3NN=np.zeros((int(number_of_runs),4)) 
crse_runsag180_4NN=np.zeros((int(number_of_runs),4)) 
crse_runsag180_5NN=np.zeros((int(number_of_runs),4))     
crse_runsag180_6NN=np.zeros((int(number_of_runs),4))

crse_runsdf30_1NN=np.zeros((int(number_of_runs),4))
crse_runsdf30_2NN=np.zeros((int(number_of_runs),4))
crse_runsdf30_3NN=np.zeros((int(number_of_runs),4))

crse_runsdf60_1NN=np.zeros((int(number_of_runs),4))
crse_runsdf60_2NN=np.zeros((int(number_of_runs),4))
crse_runsdf60_3NN=np.zeros((int(number_of_runs),4))

crse_runsdf120_1NN=np.zeros((int(number_of_runs),4))
crse_runsdf120_2NN=np.zeros((int(number_of_runs),4))
crse_runsdf120_3NN=np.zeros((int(number_of_runs),4))

crse_runsdf180_1NN=np.zeros((int(number_of_runs),4))
crse_runsdf180_2NN=np.zeros((int(number_of_runs),4))
crse_runsdf180_3NN=np.zeros((int(number_of_runs),4))   

'array creation to store  maximum INS physical model CRSE for each scenario after each full training. '
crse_runsmwPHY_MOD=np.zeros((int(number_of_runs),4))

crse_runsra_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsra_2PHY_MOD=np.zeros((int(number_of_runs),4))

crse_runscia_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runscia_2PHY_MOD=np.zeros((int(number_of_runs),4))    

crse_runshb_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runshb_2PHY_MOD=np.zeros((int(number_of_runs),4))

crse_runsslr_1PHY_MOD=np.zeros((int(number_of_runs),4)) 
crse_runsslr_2PHY_MOD=np.zeros((int(number_of_runs),4)) 
crse_runsslr_3PHY_MOD=np.zeros((int(number_of_runs),4)) 

crse_runswr_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runswr_2PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runswr_3PHY_MOD=np.zeros((int(number_of_runs),4))

crse_runsag30_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag30_2PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag30_3PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag30_4PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag30_5PHY_MOD=np.zeros((int(number_of_runs),4))    
crse_runsag30_6PHY_MOD=np.zeros((int(number_of_runs),4))

crse_runsag60_1PHY_MOD=np.zeros((int(number_of_runs),4))  
crse_runsag60_2PHY_MOD=np.zeros((int(number_of_runs),4)) 
crse_runsag60_3PHY_MOD=np.zeros((int(number_of_runs),4)) 
crse_runsag60_4PHY_MOD=np.zeros((int(number_of_runs),4)) 
crse_runsag60_5PHY_MOD=np.zeros((int(number_of_runs),4)) 
crse_runsag60_6PHY_MOD=np.zeros((int(number_of_runs),4)) 

crse_runsag90_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag90_2PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag90_3PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag90_4PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag90_5PHY_MOD=np.zeros((int(number_of_runs),4))    
crse_runsag90_6PHY_MOD=np.zeros((int(number_of_runs),4))

crse_runsag120_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag120_2PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag120_3PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag120_4PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag120_5PHY_MOD=np.zeros((int(number_of_runs),4))    
crse_runsag120_6PHY_MOD=np.zeros((int(number_of_runs),4))

crse_runsag180_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag180_2PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag180_3PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag180_4PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsag180_5PHY_MOD=np.zeros((int(number_of_runs),4))    
crse_runsag180_6PHY_MOD=np.zeros((int(number_of_runs),4))

crse_runsdf30_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsdf30_2PHY_MOD=np.zeros((int(number_of_runs),4))        
crse_runsdf30_3PHY_MOD=np.zeros((int(number_of_runs),4))  

crse_runsdf60_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsdf60_2PHY_MOD=np.zeros((int(number_of_runs),4))    
crse_runsdf60_3PHY_MOD=np.zeros((int(number_of_runs),4)) 

crse_runsdf90_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsdf90_2PHY_MOD=np.zeros((int(number_of_runs),4))    
crse_runsdf90_3PHY_MOD=np.zeros((int(number_of_runs),4)) 

crse_runsdf120_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsdf120_2PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsdf120_3PHY_MOD=np.zeros((int(number_of_runs),4))

crse_runsdf180_1PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsdf180_2PHY_MOD=np.zeros((int(number_of_runs),4))
crse_runsdf180_3PHY_MOD=np.zeros((int(number_of_runs),4))
for nfr in range(number_of_runs):
    print('full training run: '+ str(nfr))

    #############################################################################
    #############################################################################
    #############################################################################
    'RNN TRAINING'
    #############################################################################
    #############################################################################
    #############################################################################

    Run_time, regress=RNN_model(np.array(x),np.array(y), input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps)

    'challenging scenarios'   

    dist_travldmwcs, perf_metrmw_crsepcs, perf_metrmw_crsedrcs, perf_metrmw_ctepcs, perf_metrmw_ctedrcs, perf_metrmw_aepspcs, perf_metrmw_aepsdrcs,newPpredsmwcs, insmwcs, gpsmwcs=predictcs(xtmw,ytmw, itmw, gtmw, regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Motorway Scenario',outage1) 

    dist_travldra1cs, perf_metrra1_crsepcs, perf_metrra1_crsedrcs, perf_metrra1_ctepcs, perf_metrra1_ctedrcs,perf_metrra1_aepspcs, perf_metrra1_aepsdrcs,newPpredsra1cs, insra1cs, gpsra1cs=predictcs(xtra1,ytra1, itra1, gtra1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Roundabout Scenario V_Vta11',outage1)  
    dist_travldra2cs, perf_metrra2_crsepcs, perf_metrra2_crsedrcs, perf_metrra2_ctepcs, perf_metrra2_ctedrcs,perf_metrra2_aepspcs, perf_metrra2_aepsdrcs,newPpredsra2cs, insra2cs, gpsra2cs=predictcs(xtra2,ytra2, itra2, gtra2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Roundabout Scenario V_Vfb02d',outage1) 

    dist_travldcia1cs, perf_metrcia1_crsepcs, perf_metrcia1_crsedrcs,perf_metrcia1_ctepcs, perf_metrcia1_ctedrcs,perf_metrcia1_aepspcs, perf_metrcia1_aepsdrcs,newPpredscia1cs, insciac1s, gpsciac1s=predictcs(xtcia1,ytcia1, itcia1, gtcia1, regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Quick Changes in \n Acceleration Scenario V_Vfb02e',outage1) 
    dist_travldcia2cs, perf_metrcia2_crsepcs, perf_metrcia2_crsedrcs,perf_metrcia2_ctepcs, perf_metrcia2_ctedrcs,perf_metrcia2_aepspcs, perf_metrcia2_aepsdrcs,newPpredscia2cs, insciac2s, gpsciac2s=predictcs(xtcia2,ytcia2, itcia2, gtcia2, regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Quick Changes in \n Acceleration Scenario V_Vta12',outage1) 

    dist_travldhb1cs, perf_metrhb1_crsepcs, perf_metrhb1_crsedrcs,perf_metrhb1_ctepcs, perf_metrhb1_ctedrcs,perf_metrhb1_aepspcs, perf_metrhb1_aepsdrcs,newPpredshb1cs, inshb1cs, gpshb1cs=predictcs(xthb1,ythb1, ithb1, gthb1, regress, seq_dim, input_dim, mode,Ts, dgmx, dgmn, Z, 'Hard Brake Scenario V_Vw16b',outage1) 
    dist_travldhb2cs, perf_metrhb2_crsepcs, perf_metrhb2_crsedrcs,perf_metrhb2_ctepcs, perf_metrhb2_ctedrcs,perf_metrhb2_aepspcs, perf_metrhb2_aepsdrcs,newPpredshb2cs, inshb2cs, gpshb2cs=predictcs(xthb2,ythb2, ithb2, gthb2, regress, seq_dim, input_dim, mode,Ts, dgmx, dgmn, Z, 'Hard Brake Scenario V_Vw17',outage1) 

    dist_travldslr1cs, perf_metrslr1_crsepcs, perf_metrslr1_crsedrcs,perf_metrslr1_ctepcs, perf_metrslr1_ctedrcs,perf_metrslr1_aepspcs, perf_metrslr1_aepsdrcs,newPpredsslr1cs, insslr1cs, gpsslr1cs=predictcs(xtslr1,ytslr1, itslr1, gtslr1, regress, seq_dim, input_dim, mode,Ts, dgmx, dgmn, Z, 'Sharp Cornering and \n Successive Left and Right Turns Scenario V_Vw6',outage1) 
    dist_travldslr2cs, perf_metrslr2_crsepcs, perf_metrslr2_crsedrcs,perf_metrslr2_ctepcs, perf_metrslr2_ctedrcs,perf_metrslr2_aepspcs, perf_metrslr2_aepsdrcs,newPpredsslr2cs, insslr2cs, gpsslr2cs=predictcs(xtslr2,ytslr2, itslr2, gtslr2, regress, seq_dim, input_dim, mode,Ts, dgmx, dgmn, Z, 'Sharp Cornering and \n Successive Left and Right Turns Scenario V_Vw8',outage1) 
    dist_travldslr3cs, perf_metrslr3_crsepcs, perf_metrslr3_crsedrcs,perf_metrslr3_ctepcs, perf_metrslr3_ctedrcs,perf_metrslr3_aepspcs, perf_metrslr3_aepsdrcs,newPpredsslr3cs, insslr3cs, gpsslr3cs=predictcs(xtslr3,ytslr3, itslr3, gtslr3, regress, seq_dim, input_dim, mode,Ts, dgmx, dgmn, Z, 'Sharp Cornering and \n Successive Left and Right Turns Scenario V_Vw7',outage1) 

    dist_travldwr1cs, perf_metrwr1_crsepcs, perf_metrwr1_crsedrcs,perf_metrwr1_ctepcs, perf_metrwr1_ctedrcs,perf_metrwr1_aepspcs, perf_metrwr1_aepsdrcs,newPpredswr1cs, inswr1cs, gpswr1cs=predictcs(xtwr1,ytwr1, itwr1, gtwr1, regress, seq_dim, input_dim, mode,Ts, dgmx, dgmn, Z, 'Wet Road Scenario V_Vtb8',outage1) 
    dist_travldwr2cs, perf_metrwr2_crsepcs, perf_metrwr2_crsedrcs,perf_metrwr2_ctepcs, perf_metrwr2_ctedrcs,perf_metrwr2_aepspcs, perf_metrwr2_aepsdrcs,newPpredswr2cs, inswr2cs, gpswr2cs=predictcs(xtwr2,ytwr2, itwr2, gtwr2, regress, seq_dim, input_dim, mode,Ts, dgmx, dgmn, Z, 'Wet Road Scenario V_Vtb11',outage1) 
    dist_travldwr3cs, perf_metrwr3_crsepcs, perf_metrwr3_crsedrcs,perf_metrwr3_ctepcs, perf_metrwr3_ctedrcs,perf_metrwr3_aepspcs, perf_metrwr3_aepsdrcs,newPpredswr3cs, inswr3cs, gpswr3cs=predictcs(xtwr3,ytwr3, itwr3, gtwr3, regress, seq_dim, input_dim, mode,Ts, dgmx, dgmn, Z, 'Wet Road Scenario V_Vtb13',outage1) 

    '30s outage'  
    dist_travldag1_30s, perf_metrag1_crsep30s, perf_metrag1_crsedr30s, perf_metrag1_ctep30s, perf_metrag1_ctedr30s,perf_metrag1_aepsp30s, perf_metrag1_aepsdr30s, newPpredsag1_30s, insag1_30s, gpsag1_30s=predictcs(xtag1,ytag1, itag1, gtag1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vta1a',outage2)   
    dist_travldag2_30s, perf_metrag2_crsep30s, perf_metrag2_crsedr30s, perf_metrag2_ctep30s, perf_metrag2_ctedr30s,perf_metrag2_aepsp30s, perf_metrag2_aepsdr30s, newPpredsag2_30s, insag2_30s, gpsag2_30s=predictcs(xtag2,ytag2, itag2, gtag2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vw2',outage2)    
    dist_travldag3_30s, perf_metrag3_crsep30s, perf_metrag3_crsedr30s, perf_metrag3_ctep30s, perf_metrag3_ctedr30s,perf_metrag3_aepsp30s, perf_metrag3_aepsdr30s, newPpredsag3_30s, insag3_30s, gpsag3_30s=predictcs(xtag3,ytag3, itag3, gtag3,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vtb1',outage2)   
    dist_travldag4_30s, perf_metrag4_crsep30s, perf_metrag4_crsedr30s, perf_metrag4_ctep30s, perf_metrag4_ctedr30s,perf_metrag4_aepsp30s, perf_metrag4_aepsdr30s, newPpredsag4_30s, insag4_30s, gpsag4_30s=predictcs(xtag4,ytag4, itag4, gtag4,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb01d',outage2)   
    dist_travldag5_30s, perf_metrag5_crsep30s, perf_metrag5_crsedr30s, perf_metrag5_ctep30s, perf_metrag5_ctedr30s,perf_metrag5_aepsp30s, perf_metrag5_aepsdr30s, newPpredsag5_30s, insag5_30s, gpsag5_30s=predictcs(xtag5,ytag5, itag5, gtag5,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb02a',outage2)   
    dist_travldag6_30s, perf_metrag6_crsep30s, perf_metrag6_crsedr30s, perf_metrag6_ctep30s, perf_metrag6_ctedr30s,perf_metrag6_aepsp30s, perf_metrag6_aepsdr30s, newPpredsag6_30s, insag6_30s, gpsag6_30s=predictcs(xtag6,ytag6, itag6, gtag6,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb02g',outage2)   
    dist_travlddf1_30s, perf_metrdf1_crsep30s, perf_metrdf1_crsedr30s, perf_metrdf1_ctep30s, perf_metrdf1_ctedr30s,perf_metrdf1_aepsp30s, perf_metrdf1_aepsdr30s, newPpredsdf1_30s, insdf1_30s, gpsdf1_30s=predictcs(xtdf1, ytdf1, itdf1, gtdf1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_S3b',outage2)   
    dist_travlddf2_30s, perf_metrdf2_crsep30s, perf_metrdf2_crsedr30s, perf_metrdf2_ctep30s, perf_metrdf2_ctedr30s,perf_metrdf2_aepsp30s, perf_metrdf2_aepsdr30s, newPpredsdf2_30s, insdf2_30s, gpsdf2_30s=predictcs(xtdf2,ytdf2, itdf2, gtdf2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_St4',outage2)   
    dist_travlddf3_30s, perf_metrdf3_crsep30s, perf_metrdf3_crsedr30s, perf_metrdf3_ctep30s, perf_metrdf3_ctedr30s,perf_metrdf3_aepsp30s, perf_metrdf3_aepsdr30s, newPpredsdf3_30s, insdf3_30s, gpsdf3_30s=predictcs(xtdf3,ytdf3, itdf3, gtdf3,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_St4',outage2)   

    '60s outage' 
    dist_travldag1_60s, perf_metrag1_crsep60s, perf_metrag1_crsedr60s, perf_metrag1_ctep60s, perf_metrag1_ctedr60s,perf_metrag1_aepsp60s, perf_metrag1_aepsdr60s, newPpredsag1_60s, insag1_60s, gpsag1_60s=predictcs(xtag1,ytag1, itag1, gtag1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vta1a',outage3)   
    dist_travldag2_60s, perf_metrag2_crsep60s, perf_metrag2_crsedr60s, perf_metrag2_ctep60s, perf_metrag2_ctedr60s,perf_metrag2_aepsp60s, perf_metrag2_aepsdr60s, newPpredsag2_60s, insag2_60s, gpsag2_60s=predictcs(xtag2,ytag2, itag2, gtag2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vw2',outage3)   
    dist_travldag3_60s, perf_metrag3_crsep60s, perf_metrag3_crsedr60s, perf_metrag3_ctep60s, perf_metrag3_ctedr60s,perf_metrag3_aepsp60s, perf_metrag3_aepsdr60s, newPpredsag3_60s, insag3_60s, gpsag3_60s=predictcs(xtag3,ytag3, itag3, gtag3,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vtb1',outage3)   
    dist_travldag4_60s, perf_metrag4_crsep60s, perf_metrag4_crsedr60s, perf_metrag4_ctep60s, perf_metrag4_ctedr60s,perf_metrag4_aepsp60s, perf_metrag4_aepsdr60s, newPpredsag4_60s, insag4_60s, gpsag4_60s=predictcs(xtag4,ytag4, itag4, gtag4,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb01d',outage3)   
    dist_travldag5_60s, perf_metrag5_crsep60s, perf_metrag5_crsedr60s, perf_metrag5_ctep60s, perf_metrag5_ctedr60s,perf_metrag5_aepsp60s, perf_metrag5_aepsdr60s, newPpredsag5_60s, insag5_60s, gpsag5_60s=predictcs(xtag5,ytag5, itag5, gtag5,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb02a',outage3)   
    dist_travldag6_60s, perf_metrag6_crsep60s, perf_metrag6_crsedr60s, perf_metrag6_ctep60s, perf_metrag6_ctedr60s,perf_metrag6_aepsp60s, perf_metrag6_aepsdr60s, newPpredsag6_60s, insag6_60s, gpsag6_60s=predictcs(xtag6,ytag6, itag6, gtag6,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb02g',outage3)   
    dist_travlddf1_60s, perf_metrdf1_crsep60s, perf_metrdf1_crsedr60s, perf_metrdf1_ctep60s, perf_metrdf1_ctedr60s,perf_metrdf1_aepsp60s, perf_metrdf1_aepsdr60s, newPpredsdf1_60s, insdf1_60s, gpsdf1_60s=predictcs(xtdf1,ytdf1, itdf1, gtdf1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_S3b',outage3)   
    dist_travlddf2_60s, perf_metrdf2_crsep60s, perf_metrdf2_crsedr60s, perf_metrdf2_ctep60s, perf_metrdf2_ctedr60s,perf_metrdf2_aepsp60s, perf_metrdf2_aepsdr60s, newPpredsdf2_60s, insdf2_60s, gpsdf2_60s=predictcs(xtdf2,ytdf2, itdf2, gtdf2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_St4',outage3)   
    dist_travlddf3_60s, perf_metrdf3_crsep60s, perf_metrdf3_crsedr60s, perf_metrdf3_ctep60s, perf_metrdf3_ctedr60s,perf_metrdf3_aepsp60s, perf_metrdf3_aepsdr60s, newPpredsdf3_60s, insdf3_60s, gpsdf3_60s=predictcs(xtdf3,ytdf3, itdf3, gtdf3,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_St4',outage3)   

    '90s outage'  
    dist_travldag1_90s, perf_metrag1_crsep90s, perf_metrag1_crsedr90s, perf_metrag1_ctep90s, perf_metrag1_ctedr90s,perf_metrag1_aepsp90s, perf_metrag1_aepsdr90s, newPpredsag1_90s, insag1_90s, gpsag1_90s=predictcs(xtag1,ytag1, itag1, gtag1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vta1a',outage4)   
    dist_travldag2_90s, perf_metrag2_crsep90s, perf_metrag2_crsedr90s, perf_metrag2_ctep90s, perf_metrag2_ctedr90s,perf_metrag2_aepsp90s, perf_metrag2_aepsdr90s, newPpredsag2_90s, insag2_90s, gpsag2_90s=predictcs(xtag2,ytag2, itag2, gtag2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vw2',outage4)   
    dist_travldag3_90s, perf_metrag3_crsep90s, perf_metrag3_crsedr90s, perf_metrag3_ctep90s, perf_metrag3_ctedr90s,perf_metrag3_aepsp90s, perf_metrag3_aepsdr90s, newPpredsag3_90s, insag3_90s, gpsag3_90s=predictcs(xtag3,ytag3, itag3, gtag3,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vtb1',outage4)   
    dist_travldag4_90s, perf_metrag4_crsep90s, perf_metrag4_crsedr90s, perf_metrag4_ctep90s, perf_metrag4_ctedr90s,perf_metrag4_aepsp90s, perf_metrag4_aepsdr90s, newPpredsag4_90s, insag4_90s, gpsag4_90s=predictcs(xtag4,ytag4, itag4, gtag4,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb01d',outage4)   
    dist_travldag5_90s, perf_metrag5_crsep90s, perf_metrag5_crsedr90s, perf_metrag5_ctep90s, perf_metrag5_ctedr90s,perf_metrag5_aepsp90s, perf_metrag5_aepsdr90s, newPpredsag5_90s, insag5_90s, gpsag5_90s=predictcs(xtag5,ytag5, itag5, gtag5,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb02a',outage4)   
    dist_travldag6_90s, perf_metrag6_crsep90s, perf_metrag6_crsedr90s, perf_metrag6_ctep90s, perf_metrag6_ctedr90s,perf_metrag6_aepsp90s, perf_metrag6_aepsdr90s, newPpredsag6_90s, insag6_90s, gpsag6_90s=predictcs(xtag6,ytag6, itag6, gtag6,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb02g',outage4)   
    dist_travlddf1_90s, perf_metrdf1_crsep90s, perf_metrdf1_crsedr90s, perf_metrdf1_ctep90s, perf_metrdf1_ctedr90s,perf_metrdf1_aepsp90s, perf_metrdf1_aepsdr90s, newPpredsdf1_90s, insdf1_90s, gpsdf1_90s=predictcs(xtdf1,ytdf1, itdf1, gtdf1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_S3b',outage4)   
    dist_travlddf2_90s, perf_metrdf2_crsep90s, perf_metrdf2_crsedr90s, perf_metrdf2_ctep90s, perf_metrdf2_ctedr90s,perf_metrdf2_aepsp90s, perf_metrdf2_aepsdr90s, newPpredsdf2_90s, insdf2_90s, gpsdf2_90s=predictcs(xtdf2,ytdf2, itdf2, gtdf2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_St4',outage4)   
    dist_travlddf3_90s, perf_metrdf3_crsep90s, perf_metrdf3_crsedr90s, perf_metrdf3_ctep90s, perf_metrdf3_ctedr90s,perf_metrdf3_aepsp90s, perf_metrdf3_aepsdr90s, newPpredsdf3_90s, insdf3_90s, gpsdf3_90s=predictcs(xtdf3,ytdf3, itdf3, gtdf3,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_St4',outage4)   

    '120s outage'  
    dist_travldag1_120s, perf_metrag1_crsep120s, perf_metrag1_crsedr120s, perf_metrag1_ctep120s, perf_metrag1_ctedr120s,perf_metrag1_aepsp120s, perf_metrag1_aepsdr120s, newPpredsag1_120s, insag1_120s, gpsag1_120s=predictcs(xtag1,ytag1, itag1, gtag1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vta1a',outage5)   
    dist_travldag2_120s, perf_metrag2_crsep120s, perf_metrag2_crsedr120s, perf_metrag2_ctep120s, perf_metrag2_ctedr120s,perf_metrag2_aepsp120s, perf_metrag2_aepsdr120s, newPpredsag2_120s, insag2_120s, gpsag2_120s=predictcs(xtag2,ytag2, itag2, gtag2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vw2',outage5)   
    dist_travldag3_120s, perf_metrag3_crsep120s, perf_metrag3_crsedr120s, perf_metrag3_ctep120s, perf_metrag3_ctedr120s,perf_metrag3_aepsp120s, perf_metrag3_aepsdr120s, newPpredsag3_120s, insag3_120s, gpsag3_120s=predictcs(xtag3,ytag3, itag3, gtag3,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vtb1',outage5)   
    dist_travldag4_120s, perf_metrag4_crsep120s, perf_metrag4_crsedr120s, perf_metrag4_ctep120s, perf_metrag4_ctedr120s,perf_metrag4_aepsp120s, perf_metrag4_aepsdr120s, newPpredsag4_120s, insag4_120s, gpsag4_120s=predictcs(xtag4,ytag4, itag4, gtag4,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb01d',outage5)   
    dist_travldag5_120s, perf_metrag5_crsep120s, perf_metrag5_crsedr120s, perf_metrag5_ctep120s, perf_metrag5_ctedr120s,perf_metrag5_aepsp120s, perf_metrag5_aepsdr120s, newPpredsag5_120s, insag5_120s, gpsag5_120s=predictcs(xtag5,ytag5, itag5, gtag5,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb02a',outage5)   
    dist_travldag6_120s, perf_metrag6_crsep120s, perf_metrag6_crsedr120s, perf_metrag6_ctep120s, perf_metrag6_ctedr120s,perf_metrag6_aepsp120s, perf_metrag6_aepsdr120s, newPpredsag6_120s, insag6_120s, gpsag6_120s=predictcs(xtag6,ytag6, itag6, gtag6,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Agrressive driving 120s Outage V_Vfb02g',outage5)   
    dist_travlddf1_120s, perf_metrdf1_crsep120s, perf_metrdf1_crsedr120s, perf_metrdf1_ctep120s, perf_metrdf1_ctedr120s,perf_metrdf1_aepsp120s, perf_metrdf1_aepsdr120s, newPpredsdf1_120s, insdf1_120s, gpsdf1_120s=predictcs(xtdf1,ytdf1, itdf1, gtdf1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_S3b',outage5)   
    dist_travlddf2_120s, perf_metrdf2_crsep120s, perf_metrdf2_crsedr120s, perf_metrdf2_ctep120s, perf_metrdf2_ctedr120s,perf_metrdf2_aepsp120s, perf_metrdf2_aepsdr120s, newPpredsdf2_120s, insdf2_120s, gpsdf2_120s=predictcs(xtdf2,ytdf2, itdf2, gtdf2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_St4',outage5)   
    dist_travlddf3_120s, perf_metrdf3_crsep120s, perf_metrdf3_crsedr120s, perf_metrdf3_ctep120s, perf_metrdf3_ctedr120s,perf_metrdf3_aepsp120s, perf_metrdf3_aepsdr120s, newPpredsdf3_120s, insdf3_120s, gpsdf3_120s=predictcs(xtdf3,ytdf3, itdf3, gtdf3,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, 'Defensive driving 120s Outage V_St4',outage5)   

    '180s outage'  
    dist_travldag1_180s, perf_metrag1_crsep180s, perf_metrag1_crsedr180s, perf_metrag1_ctep180s, perf_metrag1_ctedr180s,perf_metrag1_aepsp180s, perf_metrag1_aepsdr180s, newPpredsag1_180s, insag1_180s, gpsag1_180s=predictcs(xtag1,ytag1, itag1, gtag1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, '180 s GNSS Outage (V_Vtb3)',outage6)   
    dist_travldag2_180s, perf_metrag2_crsep180s, perf_metrag2_crsedr180s, perf_metrag2_ctep180s, perf_metrag2_ctedr180s,perf_metrag2_aepsp180s, perf_metrag2_aepsdr180s, newPpredsag2_180s, insag2_180s, gpsag2_180s=predictcs(xtag2,ytag2, itag2, gtag2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, '180 s GNSS Outage (V_Vfb01c)',outage6)   
    dist_travldag3_180s, perf_metrag3_crsep180s, perf_metrag3_crsedr180s, perf_metrag3_ctep180s, perf_metrag3_ctedr180s,perf_metrag3_aepsp180s, perf_metrag3_aepsdr180s, newPpredsag3_180s, insag3_180s, gpsag3_180s=predictcs(xtag3,ytag3, itag3, gtag3,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, '180 s GNSS Outage (V_Vfb02a)',outage6)   
    dist_travldag4_180s, perf_metrag4_crsep180s, perf_metrag4_crsedr180s, perf_metrag4_ctep180s, perf_metrag4_ctedr180s,perf_metrag4_aepsp180s, perf_metrag4_aepsdr180s, newPpredsag4_180s, insag4_180s, gpsag4_180s=predictcs(xtag4,ytag4, itag4, gtag4,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, '180 s GNSS Outage (V_Vta1a)',outage6)   
    dist_travldag5_180s, perf_metrag5_crsep180s, perf_metrag5_crsedr180s, perf_metrag5_ctep180s, perf_metrag5_ctedr180s,perf_metrag5_aepsp180s, perf_metrag5_aepsdr180s, newPpredsag5_180s, insag5_180s, gpsag5_180s=predictcs(xtag5,ytag5, itag5, gtag5,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, '180 s GNSS Outage (V_Vfb02b)',outage6)   
    dist_travldag6_180s, perf_metrag6_crsep180s, perf_metrag6_crsedr180s, perf_metrag6_ctep180s, perf_metrag6_ctedr180s,perf_metrag6_aepsp180s, perf_metrag6_aepsdr180s, newPpredsag6_180s, insag6_180s, gpsag6_180s=predictcs(xtag6,ytag6, itag6, gtag6,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, '180 s GNSS Outage (V_Vfb02g)',outage6)   
    dist_travlddf1_180s, perf_metrdf1_crsep180s, perf_metrdf1_crsedr180s, perf_metrdf1_ctep180s, perf_metrdf1_ctedr180s,perf_metrdf1_aepsp180s, perf_metrdf1_aepsdr180s, newPpredsdf1_180s, insdf1_180s, gpsdf1_180s=predictcs(xtdf1,ytdf1, itdf1, gtdf1,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, '180 s GNSS Outage (V_St6)',outage6)   
    dist_travlddf2_180s, perf_metrdf2_crsep180s, perf_metrdf2_crsedr180s, perf_metrdf2_ctep180s, perf_metrdf2_ctedr180s,perf_metrdf2_aepsp180s, perf_metrdf2_aepsdr180s, newPpredsdf2_180s, insdf2_180s, gpsdf2_180s=predictcs(xtdf2,ytdf2, itdf2, gtdf2,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, '180 s GNSS Outage (V_St7)',outage6)   
    dist_travlddf3_180s, perf_metrdf3_crsep180s, perf_metrdf3_crsedr180s, perf_metrdf3_ctep180s, perf_metrdf3_ctedr180s,perf_metrdf3_aepsp180s, perf_metrdf3_aepsdr180s, newPpredsdf3_180s, insdf3_180s, gpsdf3_180s=predictcs(xtdf3,ytdf3, itdf3, gtdf3,regress, seq_dim, input_dim, mode, Ts, dgmx, dgmn, Z, '180 s GNSS Outage (V_S3a)',outage6)   


    'indexes the maximum prediction crse performance metrics output'
    crse_runsmwNN[nfr]=perf_metrmw_crsepcs[:]

    crse_runsra_1NN[nfr]=perf_metrra1_crsepcs[:]  
    crse_runsra_2NN[nfr]=perf_metrra2_crsepcs[:]         

    crse_runscia_1NN[nfr]=perf_metrcia1_crsepcs[:] 
    crse_runscia_2NN[nfr]=perf_metrcia2_crsepcs[:]         

    crse_runshb_1NN[nfr]=perf_metrhb1_crsepcs[:]
    crse_runshb_2NN[nfr]=perf_metrhb2_crsepcs[:]        

    crse_runsslr_1NN[nfr]=perf_metrslr1_crsepcs[:]  
    crse_runsslr_2NN[nfr]=perf_metrslr2_crsepcs[:]         
    crse_runsslr_3NN[nfr]=perf_metrslr3_crsepcs[:] 

    crse_runswr_1NN[nfr]=perf_metrwr1_crsepcs[:]
    crse_runswr_2NN[nfr]=perf_metrwr2_crsepcs[:]        
    crse_runswr_3NN[nfr]=perf_metrwr3_crsepcs[:]        

    crse_runsag30_1NN[nfr]=perf_metrag1_crsep30s[:]
    crse_runsag30_2NN[nfr]=perf_metrag2_crsep30s[:]        
    crse_runsag30_3NN[nfr]=perf_metrag3_crsep30s[:]        
    crse_runsag30_4NN[nfr]=perf_metrag4_crsep30s[:]        
    crse_runsag30_5NN[nfr]=perf_metrag5_crsep30s[:] 
    crse_runsag30_6NN[nfr]=perf_metrag6_crsep30s[:]        
  
    crse_runsag60_1NN[nfr]=perf_metrag1_crsep60s[:]
    crse_runsag60_2NN[nfr]=perf_metrag2_crsep60s[:]        
    crse_runsag60_3NN[nfr]=perf_metrag3_crsep60s[:]        
    crse_runsag60_4NN[nfr]=perf_metrag4_crsep60s[:]        
    crse_runsag60_5NN[nfr]=perf_metrag5_crsep60s[:] 
    crse_runsag60_6NN[nfr]=perf_metrag6_crsep60s[:]        

    crse_runsag120_1NN[nfr]=perf_metrag1_crsep120s[:]
    crse_runsag120_2NN[nfr]=perf_metrag2_crsep120s[:]        
    crse_runsag120_3NN[nfr]=perf_metrag3_crsep120s[:]        
    crse_runsag120_4NN[nfr]=perf_metrag4_crsep120s[:]        
    crse_runsag120_5NN[nfr]=perf_metrag5_crsep120s[:] 
    crse_runsag120_6NN[nfr]=perf_metrag6_crsep120s[:]        

    crse_runsag180_1NN[nfr]=perf_metrag1_crsep180s[:]
    crse_runsag180_2NN[nfr]=perf_metrag2_crsep180s[:]        
    crse_runsag180_3NN[nfr]=perf_metrag3_crsep180s[:]        
    crse_runsag180_4NN[nfr]=perf_metrag4_crsep180s[:]        
    crse_runsag180_5NN[nfr]=perf_metrag5_crsep180s[:] 
    crse_runsag180_6NN[nfr]=perf_metrag6_crsep180s[:]

    crse_runsdf30_1NN[nfr]=perf_metrdf1_crsep30s[:]
    crse_runsdf30_2NN[nfr]=perf_metrdf2_crsep30s[:]               
    crse_runsdf30_3NN[nfr]=perf_metrdf3_crsep30s[:] 

    crse_runsdf60_1NN[nfr]=perf_metrdf1_crsep60s[:]
    crse_runsdf60_2NN[nfr]=perf_metrdf2_crsep60s[:]               
    crse_runsdf60_3NN[nfr]=perf_metrdf3_crsep60s[:]   

    crse_runsdf120_1NN[nfr]=perf_metrdf1_crsep120s[:]   
    crse_runsdf120_2NN[nfr]=perf_metrdf2_crsep120s[:]                 
    crse_runsdf120_3NN[nfr]=perf_metrdf3_crsep120s[:] 

    crse_runsdf180_1NN[nfr]=perf_metrdf1_crsep180s[:]   
    crse_runsdf180_2NN[nfr]=perf_metrdf2_crsep180s[:]  
    crse_runsdf180_3NN[nfr]=perf_metrdf3_crsep180s[:]  

    'indexes the maximum prediction crse performance metrics output for the NN'
    cte_runsmwNN[nfr]=perf_metrmw_ctepcs[:]

    cte_runsra_1NN[nfr]=perf_metrra1_ctepcs[:]  
    cte_runsra_2NN[nfr]=perf_metrra2_ctepcs[:]         

    cte_runscia_1NN[nfr]=perf_metrcia1_ctepcs[:] 
    cte_runscia_2NN[nfr]=perf_metrcia2_ctepcs[:]         

    cte_runshb_1NN[nfr]=perf_metrhb1_ctepcs[:]
    cte_runshb_2NN[nfr]=perf_metrhb2_ctepcs[:]        

    cte_runsslr_1NN[nfr]=perf_metrslr1_ctepcs[:]  
    cte_runsslr_2NN[nfr]=perf_metrslr2_ctepcs[:]         
    cte_runsslr_3NN[nfr]=perf_metrslr3_ctepcs[:] 

    cte_runswr_1NN[nfr]=perf_metrwr1_ctepcs[:]
    cte_runswr_2NN[nfr]=perf_metrwr2_ctepcs[:]        
    cte_runswr_3NN[nfr]=perf_metrwr3_ctepcs[:]        

    cte_runsag30_1NN[nfr]=perf_metrag1_ctep30s[:]
    cte_runsag30_2NN[nfr]=perf_metrag2_ctep30s[:]        
    cte_runsag30_3NN[nfr]=perf_metrag3_ctep30s[:]        
    cte_runsag30_4NN[nfr]=perf_metrag4_ctep30s[:]        
    cte_runsag30_5NN[nfr]=perf_metrag5_ctep30s[:] 
    cte_runsag30_6NN[nfr]=perf_metrag6_ctep30s[:]        

    cte_runsag60_1NN[nfr]=perf_metrag1_ctep60s[:]
    cte_runsag60_2NN[nfr]=perf_metrag2_ctep60s[:]        
    cte_runsag60_3NN[nfr]=perf_metrag3_ctep60s[:]        
    cte_runsag60_4NN[nfr]=perf_metrag4_ctep60s[:]        
    cte_runsag60_5NN[nfr]=perf_metrag5_ctep60s[:] 
    cte_runsag60_6NN[nfr]=perf_metrag6_ctep60s[:]        

    cte_runsag120_1NN[nfr]=perf_metrag1_ctep120s[:]
    cte_runsag120_2NN[nfr]=perf_metrag2_ctep120s[:]        
    cte_runsag120_3NN[nfr]=perf_metrag3_ctep120s[:]        
    cte_runsag120_4NN[nfr]=perf_metrag4_ctep120s[:]        
    cte_runsag120_5NN[nfr]=perf_metrag5_ctep120s[:] 
    cte_runsag120_6NN[nfr]=perf_metrag6_ctep120s[:]        

    cte_runsag180_1NN[nfr]=perf_metrag1_ctep180s[:]
    cte_runsag180_2NN[nfr]=perf_metrag2_ctep180s[:]        
    cte_runsag180_3NN[nfr]=perf_metrag3_ctep180s[:]        
    cte_runsag180_4NN[nfr]=perf_metrag4_ctep180s[:]        
    cte_runsag180_5NN[nfr]=perf_metrag5_ctep180s[:] 
    cte_runsag180_6NN[nfr]=perf_metrag6_ctep180s[:]

    cte_runsdf30_1NN[nfr]=perf_metrdf1_ctep30s[:]
    cte_runsdf30_2NN[nfr]=perf_metrdf2_ctep30s[:]               
    cte_runsdf30_3NN[nfr]=perf_metrdf3_ctep30s[:]  

    cte_runsdf60_1NN[nfr]=perf_metrdf1_ctep60s[:]
    cte_runsdf60_2NN[nfr]=perf_metrdf2_ctep60s[:]               
    cte_runsdf60_3NN[nfr]=perf_metrdf3_ctep60s[:] 

    cte_runsdf120_1NN[nfr]=perf_metrdf1_ctep120s[:]   
    cte_runsdf120_2NN[nfr]=perf_metrdf2_ctep120s[:]
    cte_runsdf120_3NN[nfr]=perf_metrdf3_ctep120s[:]

    cte_runsdf180_1NN[nfr]=perf_metrdf1_ctep180s[:]   
    cte_runsdf180_2NN[nfr]=perf_metrdf2_ctep180s[:]
    cte_runsdf180_3NN[nfr]=perf_metrdf3_ctep180s[:]


        
      'indexes the crse performance metric output for the physical model'
 
    crse_runsmwPHY_MOD[nfr]=perf_metrmw_crsedrcs[:]

    crse_runsra_1PHY_MOD[nfr]=perf_metrra1_crsedrcs[:] 
    crse_runsra_2PHY_MOD[nfr]=perf_metrra2_crsedrcs[:]         

    crse_runscia_1PHY_MOD[nfr]=perf_metrcia1_crsedrcs[:]
    crse_runscia_2PHY_MOD[nfr]=perf_metrcia2_crsedrcs[:]        

    crse_runshb_1PHY_MOD[nfr]=perf_metrhb1_crsedrcs[:]
    crse_runshb_2PHY_MOD[nfr]=perf_metrhb2_crsedrcs[:]        

    crse_runsslr_1PHY_MOD[nfr]=perf_metrslr1_crsedrcs[:] 
    crse_runsslr_2PHY_MOD[nfr]=perf_metrslr2_crsedrcs[:]         
    crse_runsslr_3PHY_MOD[nfr]=perf_metrslr3_crsedrcs[:]         

    crse_runswr_1PHY_MOD[nfr]=perf_metrwr1_crsedrcs[:]
    crse_runswr_2PHY_MOD[nfr]=perf_metrwr2_crsedrcs[:]        
    crse_runswr_3PHY_MOD[nfr]=perf_metrwr3_crsedrcs[:]

    crse_runsag30_1PHY_MOD[nfr]=perf_metrag1_crsedr30s[:]
    crse_runsag30_2PHY_MOD[nfr]=perf_metrag2_crsedr30s[:]        
    crse_runsag30_3PHY_MOD[nfr]=perf_metrag3_crsedr30s[:]        
    crse_runsag30_4PHY_MOD[nfr]=perf_metrag4_crsedr30s[:]        
    crse_runsag30_5PHY_MOD[nfr]=perf_metrag5_crsedr30s[:]
    crse_runsag30_6PHY_MOD[nfr]=perf_metrag6_crsedr30s[:]        

    crse_runsag60_1PHY_MOD[nfr]=perf_metrag1_crsedr60s[:]
    crse_runsag60_2PHY_MOD[nfr]=perf_metrag2_crsedr60s[:]        
    crse_runsag60_3PHY_MOD[nfr]=perf_metrag3_crsedr60s[:]        
    crse_runsag60_4PHY_MOD[nfr]=perf_metrag4_crsedr60s[:]        
    crse_runsag60_5PHY_MOD[nfr]=perf_metrag5_crsedr60s[:]
    crse_runsag60_6PHY_MOD[nfr]=perf_metrag6_crsedr60s[:]        

    crse_runsag120_1PHY_MOD[nfr]=perf_metrag1_crsedr120s[:]
    crse_runsag120_2PHY_MOD[nfr]=perf_metrag2_crsedr120s[:]        
    crse_runsag120_3PHY_MOD[nfr]=perf_metrag3_crsedr120s[:]        
    crse_runsag120_4PHY_MOD[nfr]=perf_metrag4_crsedr120s[:]
    crse_runsag120_5PHY_MOD[nfr]=perf_metrag5_crsedr120s[:]
    crse_runsag120_6PHY_MOD[nfr]=perf_metrag6_crsedr120s[:]        

    crse_runsag180_1PHY_MOD[nfr]=perf_metrag1_crsedr180s[:]
    crse_runsag180_2PHY_MOD[nfr]=perf_metrag2_crsedr180s[:]        
    crse_runsag180_3PHY_MOD[nfr]=perf_metrag3_crsedr180s[:]        
    crse_runsag180_4PHY_MOD[nfr]=perf_metrag4_crsedr180s[:]
    crse_runsag180_5PHY_MOD[nfr]=perf_metrag5_crsedr180s[:]
    crse_runsag180_6PHY_MOD[nfr]=perf_metrag6_crsedr180s[:]        

    crse_runsdf30_1PHY_MOD[nfr]=perf_metrdf1_crsedr30s[:]
    crse_runsdf30_2PHY_MOD[nfr]=perf_metrdf2_crsedr30s[:]              
    crse_runsdf30_3PHY_MOD[nfr]=perf_metrdf3_crsedr30s[:]  

    crse_runsdf60_1PHY_MOD[nfr]=perf_metrdf1_crsedr60s[:]
    crse_runsdf60_2PHY_MOD[nfr]=perf_metrdf2_crsedr60s[:]               
    crse_runsdf60_3PHY_MOD[nfr]=perf_metrdf3_crsedr60s[:] 

    crse_runsdf120_1PHY_MOD[nfr]=perf_metrdf1_crsedr120s[:] 
    crse_runsdf120_2PHY_MOD[nfr]=perf_metrdf2_crsedr120s[:]                  
    crse_runsdf120_3PHY_MOD[nfr]=perf_metrdf3_crsedr120s[:]  

    crse_runsdf180_1PHY_MOD[nfr]=perf_metrdf1_crsedr180s[:] 
    crse_runsdf180_2PHY_MOD[nfr]=perf_metrdf2_crsedr180s[:]                  
    crse_runsdf180_3PHY_MOD[nfr]=perf_metrdf3_crsedr180s[:] 



    'indexes the cte performance metric output for the physical model'
    cte_runsmwPHY_MOD[nfr]=perf_metrmw_ctedrcs[:]

    cte_runsra_1PHY_MOD[nfr]=perf_metrra1_ctedrcs[:] 
    cte_runsra_2PHY_MOD[nfr]=perf_metrra2_ctedrcs[:]         

    cte_runscia_1PHY_MOD[nfr]=perf_metrcia1_ctedrcs[:]
    cte_runscia_2PHY_MOD[nfr]=perf_metrcia2_ctedrcs[:]        

    cte_runshb_1PHY_MOD[nfr]=perf_metrhb1_ctedrcs[:]
    cte_runshb_2PHY_MOD[nfr]=perf_metrhb2_ctedrcs[:]        

    cte_runsslr_1PHY_MOD[nfr]=perf_metrslr1_ctedrcs[:] 
    cte_runsslr_2PHY_MOD[nfr]=perf_metrslr2_ctedrcs[:]         
    cte_runsslr_3PHY_MOD[nfr]=perf_metrslr3_ctedrcs[:]         

    cte_runswr_1PHY_MOD[nfr]=perf_metrwr1_ctedrcs[:]
    cte_runswr_2PHY_MOD[nfr]=perf_metrwr2_ctedrcs[:]        
    cte_runswr_3PHY_MOD[nfr]=perf_metrwr3_ctedrcs[:]

    cte_runsag30_1PHY_MOD[nfr]=perf_metrag1_ctedr30s[:]
    cte_runsag30_2PHY_MOD[nfr]=perf_metrag2_ctedr30s[:]        
    cte_runsag30_3PHY_MOD[nfr]=perf_metrag3_ctedr30s[:]        
    cte_runsag30_4PHY_MOD[nfr]=perf_metrag4_ctedr30s[:]        
    cte_runsag30_5PHY_MOD[nfr]=perf_metrag5_ctedr30s[:]
    cte_runsag30_6PHY_MOD[nfr]=perf_metrag6_ctedr30s[:]        

    cte_runsag60_1PHY_MOD[nfr]=perf_metrag1_ctedr60s[:]
    cte_runsag60_2PHY_MOD[nfr]=perf_metrag2_ctedr60s[:]        
    cte_runsag60_3PHY_MOD[nfr]=perf_metrag3_ctedr60s[:]        
    cte_runsag60_4PHY_MOD[nfr]=perf_metrag4_ctedr60s[:]        
    cte_runsag60_5PHY_MOD[nfr]=perf_metrag5_ctedr60s[:]
    cte_runsag60_6PHY_MOD[nfr]=perf_metrag6_ctedr60s[:]        

    cte_runsag120_1PHY_MOD[nfr]=perf_metrag1_ctedr120s[:]
    cte_runsag120_2PHY_MOD[nfr]=perf_metrag2_ctedr120s[:]        
    cte_runsag120_3PHY_MOD[nfr]=perf_metrag3_ctedr120s[:]        
    cte_runsag120_4PHY_MOD[nfr]=perf_metrag4_ctedr120s[:]
    cte_runsag120_5PHY_MOD[nfr]=perf_metrag5_ctedr120s[:]
    cte_runsag120_6PHY_MOD[nfr]=perf_metrag6_ctedr120s[:]        

    cte_runsag180_1PHY_MOD[nfr]=perf_metrag1_ctedr180s[:]
    cte_runsag180_2PHY_MOD[nfr]=perf_metrag2_ctedr180s[:]        
    cte_runsag180_3PHY_MOD[nfr]=perf_metrag3_ctedr180s[:]        
    cte_runsag180_4PHY_MOD[nfr]=perf_metrag4_ctedr180s[:]
    cte_runsag180_5PHY_MOD[nfr]=perf_metrag5_ctedr180s[:]
    cte_runsag180_6PHY_MOD[nfr]=perf_metrag6_ctedr180s[:]        

    cte_runsdf30_1PHY_MOD[nfr]=perf_metrdf1_ctedr30s[:]
    cte_runsdf30_2PHY_MOD[nfr]=perf_metrdf2_ctedr30s[:]              
    cte_runsdf30_3PHY_MOD[nfr]=perf_metrdf3_ctedr30s[:] 

    cte_runsdf60_1PHY_MOD[nfr]=perf_metrdf1_ctedr60s[:]
    cte_runsdf60_2PHY_MOD[nfr]=perf_metrdf2_ctedr60s[:]               
    cte_runsdf60_3PHY_MOD[nfr]=perf_metrdf3_ctedr60s[:]  

    cte_runsdf120_1PHY_MOD[nfr]=perf_metrdf1_ctedr120s[:] 
    cte_runsdf120_2PHY_MOD[nfr]=perf_metrdf2_ctedr120s[:]        
    cte_runsdf120_3PHY_MOD[nfr]=perf_metrdf3_ctedr120s[:]  

    cte_runsdf180_1PHY_MOD[nfr]=perf_metrdf1_ctedr180s[:] 
    cte_runsdf180_2PHY_MOD[nfr]=perf_metrdf2_ctedr180s[:]        
    cte_runsdf180_3PHY_MOD[nfr]=perf_metrdf3_ctedr180s[:]


     
'indexes the best results across all training runs'       
amwNN=np.amin(crse_runsmwNN,axis=0)

ara_1NN=np.amin(crse_runsra_1NN,axis=0)
ara_2NN=np.amin(crse_runsra_2NN,axis=0)

acia_1NN=np.amin(crse_runscia_1NN,axis=0)
acia_2NN=np.amin(crse_runscia_2NN,axis=0)

ahb_1NN=np.amin(crse_runshb_1NN,axis=0)
ahb_2NN=np.amin(crse_runshb_2NN,axis=0)

aslr_1NN=np.amin(crse_runsslr_1NN,axis=0)
aslr_2NN=np.amin(crse_runsslr_2NN,axis=0)
aslr_3NN=np.amin(crse_runsslr_3NN,axis=0)

awr_1NN=np.amin(crse_runswr_1NN,axis=0)
awr_2NN=np.amin(crse_runswr_2NN,axis=0)
awr_3NN=np.amin(crse_runswr_3NN,axis=0)

aag030_1NN=np.amin(crse_runsag30_1NN,axis=0)
aag030_2NN=np.amin(crse_runsag30_2NN,axis=0)
aag030_3NN=np.amin(crse_runsag30_3NN,axis=0)
aag030_4NN=np.amin(crse_runsag30_4NN,axis=0)
aag030_5NN=np.amin(crse_runsag30_5NN,axis=0)
aag030_6NN=np.amin(crse_runsag30_6NN,axis=0)

aag060_1NN=np.amin(crse_runsag60_1NN,axis=0)
aag060_2NN=np.amin(crse_runsag60_2NN,axis=0)
aag060_3NN=np.amin(crse_runsag60_3NN,axis=0)
aag060_4NN=np.amin(crse_runsag60_4NN,axis=0)
aag060_5NN=np.amin(crse_runsag60_5NN,axis=0)
aag060_6NN=np.amin(crse_runsag60_6NN,axis=0)

aag120_1NN=np.amin(crse_runsag120_1NN,axis=0)
aag120_2NN=np.amin(crse_runsag120_2NN,axis=0)
aag120_3NN=np.amin(crse_runsag120_3NN,axis=0)
aag120_4NN=np.amin(crse_runsag120_4NN,axis=0)
aag120_5NN=np.amin(crse_runsag120_5NN,axis=0)
aag120_6NN=np.amin(crse_runsag120_6NN,axis=0)

aag180_1NN=np.amin(crse_runsag180_1NN,axis=0)
aag180_2NN=np.amin(crse_runsag180_2NN,axis=0)
aag180_3NN=np.amin(crse_runsag180_3NN,axis=0)
aag180_4NN=np.amin(crse_runsag180_4NN,axis=0)
aag180_5NN=np.amin(crse_runsag180_5NN,axis=0)
aag180_6NN=np.amin(crse_runsag180_6NN,axis=0)

adf030_1NN=np.amin(crse_runsdf30_1NN,axis=0)
adf030_2NN=np.amin(crse_runsdf30_2NN,axis=0)
adf030_3NN=np.amin(crse_runsdf30_3NN,axis=0)

adf060_1NN=np.amin(crse_runsdf60_1NN,axis=0)
adf060_2NN=np.amin(crse_runsdf60_2NN,axis=0)
adf060_3NN=np.amin(crse_runsdf60_3NN,axis=0)

adf120_1NN=np.amin(crse_runsdf120_1NN,axis=0)
adf120_2NN=np.amin(crse_runsdf120_2NN,axis=0)
adf120_3NN=np.amin(crse_runsdf120_3NN,axis=0)
    
adf180_1NN=np.amin(crse_runsdf180_1NN,axis=0)
adf180_2NN=np.amin(crse_runsdf180_2NN,axis=0)
adf180_3NN=np.amin(crse_runsdf180_3NN,axis=0)

amwPHY_MOD=np.amin(crse_runsmwPHY_MOD,axis=0)

ara_1PHY_MOD=np.amin(crse_runsra_1PHY_MOD,axis=0)
ara_2PHY_MOD=np.amin(crse_runsra_2PHY_MOD,axis=0)

acia_1PHY_MOD=np.amin(crse_runscia_1PHY_MOD,axis=0)
acia_2PHY_MOD=np.amin(crse_runscia_2PHY_MOD,axis=0)


ahb_1PHY_MOD=np.amin(crse_runshb_1PHY_MOD,axis=0)
ahb_2PHY_MOD=np.amin(crse_runshb_2PHY_MOD,axis=0)

aslr_1PHY_MOD=np.amin(crse_runsslr_1PHY_MOD,axis=0)
aslr_2PHY_MOD=np.amin(crse_runsslr_2PHY_MOD,axis=0)
aslr_3PHY_MOD=np.amin(crse_runsslr_3PHY_MOD,axis=0)

awr_1PHY_MOD=np.amin(crse_runswr_1PHY_MOD,axis=0)
awr_2PHY_MOD=np.amin(crse_runswr_2PHY_MOD,axis=0)
awr_3PHY_MOD=np.amin(crse_runswr_3PHY_MOD,axis=0)

aag030_1PHY_MOD=np.amin(crse_runsag30_1PHY_MOD,axis=0)
aag030_2PHY_MOD=np.amin(crse_runsag30_2PHY_MOD,axis=0)
aag030_3PHY_MOD=np.amin(crse_runsag30_3PHY_MOD,axis=0)
aag030_4PHY_MOD=np.amin(crse_runsag30_4PHY_MOD,axis=0)
aag030_5PHY_MOD=np.amin(crse_runsag30_5PHY_MOD,axis=0)    
aag030_6PHY_MOD=np.amin(crse_runsag30_6PHY_MOD,axis=0)

aag060_1PHY_MOD=np.amin(crse_runsag60_1PHY_MOD,axis=0)
aag060_2PHY_MOD=np.amin(crse_runsag60_2PHY_MOD,axis=0)    
aag060_3PHY_MOD=np.amin(crse_runsag60_3PHY_MOD,axis=0)    
aag060_4PHY_MOD=np.amin(crse_runsag60_4PHY_MOD,axis=0)
aag060_5PHY_MOD=np.amin(crse_runsag60_5PHY_MOD,axis=0)    
aag060_6PHY_MOD=np.amin(crse_runsag60_6PHY_MOD,axis=0)    

aag120_1PHY_MOD=np.amin(crse_runsag120_1PHY_MOD,axis=0)
aag120_2PHY_MOD=np.amin(crse_runsag120_2PHY_MOD,axis=0)
aag120_3PHY_MOD=np.amin(crse_runsag120_3PHY_MOD,axis=0)
aag120_4PHY_MOD=np.amin(crse_runsag120_4PHY_MOD,axis=0)
aag120_5PHY_MOD=np.amin(crse_runsag120_5PHY_MOD,axis=0)
aag120_6PHY_MOD=np.amin(crse_runsag120_6PHY_MOD,axis=0)

aag180_1PHY_MOD=np.amin(crse_runsag180_1PHY_MOD,axis=0)
aag180_2PHY_MOD=np.amin(crse_runsag180_2PHY_MOD,axis=0)
aag180_3PHY_MOD=np.amin(crse_runsag180_3PHY_MOD,axis=0)
aag180_4PHY_MOD=np.amin(crse_runsag180_4PHY_MOD,axis=0)
aag180_5PHY_MOD=np.amin(crse_runsag180_5PHY_MOD,axis=0)
aag180_6PHY_MOD=np.amin(crse_runsag180_6PHY_MOD,axis=0)
    
adf030_1PHY_MOD=np.amin(crse_runsdf30_1PHY_MOD,axis=0)
adf030_2PHY_MOD=np.amin(crse_runsdf30_2PHY_MOD,axis=0)    
adf030_3PHY_MOD=np.amin(crse_runsdf30_3PHY_MOD,axis=0) 

adf060_1PHY_MOD=np.amin(crse_runsdf60_1PHY_MOD,axis=0)
adf060_2PHY_MOD=np.amin(crse_runsdf60_2PHY_MOD,axis=0)   
adf060_3PHY_MOD=np.amin(crse_runsdf60_3PHY_MOD,axis=0) 

adf120_1PHY_MOD=np.amin(crse_runsdf120_1PHY_MOD,axis=0)
adf120_2PHY_MOD=np.amin(crse_runsdf120_2PHY_MOD,axis=0)   
adf120_3PHY_MOD=np.amin(crse_runsdf120_3PHY_MOD,axis=0)

adf180_1PHY_MOD=np.amin(crse_runsdf180_1PHY_MOD,axis=0)
adf180_2PHY_MOD=np.amin(crse_runsdf180_2PHY_MOD,axis=0)   
adf180_3PHY_MOD=np.amin(crse_runsdf180_3PHY_MOD,axis=0) 

dmwNN=np.amin(cte_runsmwNN,axis=0)

dra_1NN=np.amin(cte_runsra_1NN,axis=0)
dra_2NN=np.amin(cte_runsra_2NN,axis=0)

dcia_1NN=np.amin(cte_runscia_1NN,axis=0)
dcia_2NN=np.amin(cte_runscia_2NN,axis=0)
    
dhb_1NN=np.amin(cte_runshb_1NN,axis=0)
dhb_2NN=np.amin(cte_runshb_2NN,axis=0)

dslr_1NN=np.amin(cte_runsslr_1NN,axis=0)
dslr_2NN=np.amin(cte_runsslr_2NN,axis=0)
dslr_3NN=np.amin(cte_runsslr_3NN,axis=0)

dwr_1NN=np.amin(cte_runswr_1NN,axis=0)
dwr_2NN=np.amin(cte_runswr_2NN,axis=0)
dwr_3NN=np.amin(cte_runswr_3NN,axis=0)

dag030_1NN=np.amin(cte_runsag30_1NN,axis=0)
dag030_2NN=np.amin(cte_runsag30_2NN,axis=0)
dag030_3NN=np.amin(cte_runsag30_3NN,axis=0)
dag030_4NN=np.amin(cte_runsag30_4NN,axis=0)
dag030_5NN=np.amin(cte_runsag30_5NN,axis=0)
dag030_6NN=np.amin(cte_runsag30_6NN,axis=0)

dag060_1NN=np.amin(cte_runsag60_1NN,axis=0)
dag060_2NN=np.amin(cte_runsag60_2NN,axis=0)
dag060_3NN=np.amin(cte_runsag60_3NN,axis=0)
dag060_4NN=np.amin(cte_runsag60_4NN,axis=0)
dag060_5NN=np.amin(cte_runsag60_5NN,axis=0)
dag060_6NN=np.amin(cte_runsag60_6NN,axis=0)

dag120_1NN=np.amin(cte_runsag120_1NN,axis=0)
dag120_2NN=np.amin(cte_runsag120_2NN,axis=0)
dag120_3NN=np.amin(cte_runsag120_3NN,axis=0)
dag120_4NN=np.amin(cte_runsag120_4NN,axis=0)
dag120_5NN=np.amin(cte_runsag120_5NN,axis=0)
dag120_6NN=np.amin(cte_runsag120_6NN,axis=0)

dag180_1NN=np.amin(cte_runsag180_1NN,axis=0)
dag180_2NN=np.amin(cte_runsag180_2NN,axis=0)
dag180_3NN=np.amin(cte_runsag180_3NN,axis=0)
dag180_4NN=np.amin(cte_runsag180_4NN,axis=0)
dag180_5NN=np.amin(cte_runsag180_5NN,axis=0)
dag180_6NN=np.amin(cte_runsag180_6NN,axis=0)

ddf030_1NN=np.amin(cte_runsdf30_1NN,axis=0)
ddf030_2NN=np.amin(cte_runsdf30_2NN,axis=0)
ddf030_3NN=np.amin(cte_runsdf30_3NN,axis=0)

ddf060_1NN=np.amin(cte_runsdf60_1NN,axis=0)
ddf060_2NN=np.amin(cte_runsdf60_2NN,axis=0)
ddf060_3NN=np.amin(cte_runsdf60_3NN,axis=0)

ddf120_1NN=np.amin(cte_runsdf120_1NN,axis=0)
ddf120_2NN=np.amin(cte_runsdf120_2NN,axis=0)
ddf120_3NN=np.amin(cte_runsdf120_3NN,axis=0)

ddf180_1NN=np.amin(cte_runsdf180_1NN,axis=0)
ddf180_2NN=np.amin(cte_runsdf180_2NN,axis=0)
ddf180_3NN=np.amin(cte_runsdf180_3NN,axis=0)

dmwPHY_MOD=np.amin(cte_runsmwPHY_MOD,axis=0)

dra_1PHY_MOD=np.amin(cte_runsra_1PHY_MOD,axis=0)
dra_2PHY_MOD=np.amin(cte_runsra_2PHY_MOD,axis=0)

dcia_1PHY_MOD=np.amin(cte_runscia_1PHY_MOD,axis=0)
dcia_2PHY_MOD=np.amin(cte_runscia_2PHY_MOD,axis=0)



dmwNN=np.amin(cte_runsmwNN,axis=0)

dhb_1PHY_MOD=np.amin(cte_runshb_1PHY_MOD,axis=0)
dhb_2PHY_MOD=np.amin(cte_runshb_2PHY_MOD,axis=0)

dslr_1PHY_MOD=np.amin(cte_runsslr_1PHY_MOD,axis=0)
dslr_2PHY_MOD=np.amin(cte_runsslr_2PHY_MOD,axis=0)
dslr_3PHY_MOD=np.amin(cte_runsslr_3PHY_MOD,axis=0)

dwr_1PHY_MOD=np.amin(cte_runswr_1PHY_MOD,axis=0)
dwr_2PHY_MOD=np.amin(cte_runswr_2PHY_MOD,axis=0)
dwr_3PHY_MOD=np.amin(cte_runswr_3PHY_MOD,axis=0)

dag030_1PHY_MOD=np.amin(cte_runsag30_1PHY_MOD,axis=0)
dag030_2PHY_MOD=np.amin(cte_runsag30_2PHY_MOD,axis=0)
dag030_3PHY_MOD=np.amin(cte_runsag30_3PHY_MOD,axis=0)
dag030_4PHY_MOD=np.amin(cte_runsag30_4PHY_MOD,axis=0)
dag030_5PHY_MOD=np.amin(cte_runsag30_5PHY_MOD,axis=0)    
dag030_6PHY_MOD=np.amin(cte_runsag30_6PHY_MOD,axis=0)


dag060_1PHY_MOD=np.amin(cte_runsag60_1PHY_MOD,axis=0)
dag060_2PHY_MOD=np.amin(cte_runsag60_2PHY_MOD,axis=0)    
dag060_3PHY_MOD=np.amin(cte_runsag60_3PHY_MOD,axis=0)    
dag060_4PHY_MOD=np.amin(cte_runsag60_4PHY_MOD,axis=0)
dag060_5PHY_MOD=np.amin(cte_runsag60_5PHY_MOD,axis=0)    
dag060_6PHY_MOD=np.amin(cte_runsag60_6PHY_MOD,axis=0)    

dag120_1PHY_MOD=np.amin(cte_runsag120_1PHY_MOD,axis=0)
dag120_2PHY_MOD=np.amin(cte_runsag120_2PHY_MOD,axis=0)
dag120_3PHY_MOD=np.amin(cte_runsag120_3PHY_MOD,axis=0)
dag120_4PHY_MOD=np.amin(cte_runsag120_4PHY_MOD,axis=0)
dag120_5PHY_MOD=np.amin(cte_runsag120_5PHY_MOD,axis=0)
dag120_6PHY_MOD=np.amin(cte_runsag120_6PHY_MOD,axis=0)


dag180_1PHY_MOD=np.amin(cte_runsag180_1PHY_MOD,axis=0)
dag180_2PHY_MOD=np.amin(cte_runsag180_2PHY_MOD,axis=0)
dag180_3PHY_MOD=np.amin(cte_runsag180_3PHY_MOD,axis=0)
dag180_4PHY_MOD=np.amin(cte_runsag180_4PHY_MOD,axis=0)
dag180_5PHY_MOD=np.amin(cte_runsag180_5PHY_MOD,axis=0)
dag180_6PHY_MOD=np.amin(cte_runsag180_6PHY_MOD,axis=0)


ddf030_1PHY_MOD=np.amin(cte_runsdf30_1PHY_MOD,axis=0)
ddf030_2PHY_MOD=np.amin(cte_runsdf30_2PHY_MOD,axis=0)    
ddf030_3PHY_MOD=np.amin(cte_runsdf30_3PHY_MOD,axis=0) 

ddf060_1PHY_MOD=np.amin(cte_runsdf60_1PHY_MOD,axis=0)
ddf060_2PHY_MOD=np.amin(cte_runsdf60_2PHY_MOD,axis=0)   
ddf060_3PHY_MOD=np.amin(cte_runsdf60_3PHY_MOD,axis=0)  

ddf120_1PHY_MOD=np.amin(cte_runsdf120_1PHY_MOD,axis=0)
ddf120_2PHY_MOD=np.amin(cte_runsdf120_2PHY_MOD,axis=0)   
ddf120_3PHY_MOD=np.amin(cte_runsdf120_3PHY_MOD,axis=0)  

ddf180_1PHY_MOD=np.amin(cte_runsdf180_1PHY_MOD,axis=0)
ddf180_2PHY_MOD=np.amin(cte_runsdf180_2PHY_MOD,axis=0)   
ddf180_3PHY_MOD=np.amin(cte_runsdf180_3PHY_MOD,axis=0) 

label=np.array([Max,Min, Avg, Std])

label1=np.concatenate((np.reshape(ara_1NN,(4,1)), np.reshape(ara_1NN,(4,1))))
label2=np.concatenate((np.reshape(ara_1NN,(4,1)), np.reshape(ara_1NN,(4,1)),  np.reshape(ara_1NN,(4,1))))
label3=np.concatenate((np.reshape(ara_1NN,(4,1)), np.reshape(ara_1NN,(4,1)),  np.reshape(ara_1NN,(4,1)),np.reshape(ara_1NN,(4,1)), np.reshape(ara_1NN,(4,1)),  np.reshape(ara_1NN,(4,1)),np.reshape(ara_1NN,(4,1)),  np.reshape(ara_1NN,(4,1))))
bmwNN=np.reshape(amwNN,(4,1))  
braNN=np.concatenate((np.reshape(ara_1NN,(4,1)),np.reshape(ara_2NN,(4,1))),axis=0)  
bciaNN=np.concatenate((np.reshape(acia_1NN,(4,1)),np.reshape(acia_2NN,(4,1))),axis=0)  
bhbNN=np.concatenate((np.reshape(ahb_1NN,(4,1)),np.reshape(ahb_2NN,(4,1))),axis=0)  
bslrNN=np.concatenate((np.reshape(aslr_1NN,(4,1)),np.reshape(aslr_2NN,(4,1)),np.reshape(aslr_3NN,(4,1))),axis=0)  
bwrNN=np.concatenate((np.reshape(awr_1NN,(4,1)),np.reshape(awr_2NN,(4,1)),np.reshape(awr_3NN,(4,1))),axis=0)  
b030NN=np.concatenate((np.reshape(aag030_1NN,(4,1)),np.reshape(aag030_2NN,(4,1)),np.reshape(aag030_3NN,(4,1)),np.reshape(aag030_4NN,(4,1)),np.reshape(aag030_5NN,(4,1)),np.reshape(aag030_6NN,(4,1)),np.reshape(adf030_1NN,(4,1)),np.reshape(adf030_2NN,(4,1)),np.reshape(adf030_3NN,(4,1))),axis=0)  
b060NN=np.concatenate((np.reshape(aag060_1NN,(4,1)),np.reshape(aag060_2NN,(4,1)),np.reshape(aag060_3NN,(4,1)),np.reshape(aag060_4NN,(4,1)),np.reshape(aag060_5NN,(4,1)),np.reshape(aag060_6NN,(4,1)),np.reshape(adf060_1NN,(4,1)),np.reshape(adf060_2NN,(4,1)),np.reshape(adf060_3NN,(4,1))),axis=0) 
b120NN=np.concatenate((np.reshape(aag120_1NN,(4,1)),np.reshape(aag120_2NN,(4,1)),np.reshape(aag120_3NN,(4,1)),np.reshape(aag120_4NN,(4,1)),np.reshape(aag120_5NN,(4,1)),np.reshape(aag120_6NN,(4,1)),np.reshape(adf120_1NN,(4,1)),np.reshape(adf120_2NN,(4,1)),np.reshape(adf120_3NN,(4,1))),axis=0) 
b180NN=np.concatenate((np.reshape(aag180_1NN,(4,1)),np.reshape(aag180_2NN,(4,1)),np.reshape(aag180_3NN,(4,1)),np.reshape(aag180_4NN,(4,1)),np.reshape(aag180_5NN,(4,1)),np.reshape(aag180_6NN,(4,1)),np.reshape(adf180_1NN,(4,1)),np.reshape(adf180_2NN,(4,1)),np.reshape(adf180_3NN,(4,1))),axis=0) 


bhbPHY_MOD=np.concatenate((np.reshape(ahb_1PHY_MOD,(4,1)),np.reshape(ahb_2PHY_MOD,(4,1))),axis=0)
bmwPHY_MOD=np.reshape(amwPHY_MOD,(4,1))  
braPHY_MOD=np.concatenate((np.reshape(ara_1PHY_MOD,(4,1)),np.reshape(ara_2PHY_MOD,(4,1))),axis=0)  
bciaPHY_MOD=np.concatenate((np.reshape(acia_1PHY_MOD,(4,1)),np.reshape(acia_2PHY_MOD,(4,1))),axis=0)  
bslrPHY_MOD=np.concatenate((np.reshape(aslr_1PHY_MOD,(4,1)),np.reshape(aslr_2PHY_MOD,(4,1)),np.reshape(aslr_3PHY_MOD,(4,1))),axis=0)  
bwrPHY_MOD=np.concatenate((np.reshape(awr_1PHY_MOD,(4,1)),np.reshape(awr_2PHY_MOD,(4,1)),np.reshape(awr_3PHY_MOD,(4,1))),axis=0)  
b030PHY_MOD=np.concatenate((np.reshape(aag030_1PHY_MOD,(4,1)),np.reshape(aag030_2PHY_MOD,(4,1)),np.reshape(aag030_3PHY_MOD,(4,1)),np.reshape(aag030_4PHY_MOD,(4,1)),np.reshape(aag030_5PHY_MOD,(4,1)),np.reshape(aag030_6PHY_MOD,(4,1)),np.reshape(adf030_1PHY_MOD,(4,1)),np.reshape(adf030_2PHY_MOD,(4,1)),np.reshape(adf030_3PHY_MOD,(4,1))),axis=0)  
b060PHY_MOD=np.concatenate((np.reshape(aag060_1PHY_MOD,(4,1)),np.reshape(aag060_2PHY_MOD,(4,1)),np.reshape(aag060_3PHY_MOD,(4,1)),np.reshape(aag060_4PHY_MOD,(4,1)),np.reshape(aag060_5PHY_MOD,(4,1)),np.reshape(aag060_6PHY_MOD,(4,1)),np.reshape(adf060_1PHY_MOD,(4,1)),np.reshape(adf060_2PHY_MOD,(4,1)),np.reshape(adf060_3PHY_MOD,(4,1))),axis=0) 
b120PHY_MOD=np.concatenate((np.reshape(aag120_1PHY_MOD,(4,1)),np.reshape(aag120_2PHY_MOD,(4,1)),np.reshape(aag120_3PHY_MOD,(4,1)),np.reshape(aag120_4PHY_MOD,(4,1)),np.reshape(aag120_5PHY_MOD,(4,1)),np.reshape(aag120_6PHY_MOD,(4,1)),np.reshape(adf120_1PHY_MOD,(4,1)),np.reshape(adf120_2PHY_MOD,(4,1)),np.reshape(adf120_3PHY_MOD,(4,1))),axis=0)    
b180PHY_MOD=np.concatenate((np.reshape(aag180_1PHY_MOD,(4,1)),np.reshape(aag180_2PHY_MOD,(4,1)),np.reshape(aag180_3PHY_MOD,(4,1)),np.reshape(aag180_4PHY_MOD,(4,1)),np.reshape(aag180_5PHY_MOD,(4,1)),np.reshape(aag180_6PHY_MOD,(4,1)),np.reshape(adf180_1PHY_MOD,(4,1)),np.reshape(adf180_2PHY_MOD,(4,1)),np.reshape(adf180_3PHY_MOD,(4,1))),axis=0)    

bmw=np.concatenate((label,bmw),axis=1)
emwNN=np.reshape(dmwNN,(4,1))  
eraNN=np.concatenate((np.reshape(dra_1NN,(4,1)),np.reshape(dra_2NN,(4,1))),axis=0)  
eciaNN=np.concatenate((np.reshape(dcia_1NN,(4,1)),np.reshape(dcia_2NN,(4,1))),axis=0)  
ehbNN=np.concatenate((np.reshape(dhb_1NN,(4,1)),np.reshape(dhb_2NN,(4,1))),axis=0)  
eslrNN=np.concatenate((np.reshape(dslr_1NN,(4,1)),np.reshape(dslr_2NN,(4,1)),np.reshape(dslr_3NN,(4,1))),axis=0)  
ewrNN=np.concatenate((np.reshape(dwr_1NN,(4,1)),np.reshape(dwr_2NN,(4,1)),np.reshape(dwr_3NN,(4,1))),axis=0)  
e030NN=np.concatenate((np.reshape(dag030_1NN,(4,1)),np.reshape(dag030_2NN,(4,1)),np.reshape(dag030_3NN,(4,1)),np.reshape(dag030_4NN,(4,1)),np.reshape(dag030_5NN,(4,1)),np.reshape(dag030_6NN,(4,1)),np.reshape(ddf030_1NN,(4,1)),np.reshape(ddf030_2NN,(4,1)),np.reshape(ddf030_3NN,(4,1))),axis=0)  
e060NN=np.concatenate((np.reshape(dag060_1NN,(4,1)),np.reshape(dag060_2NN,(4,1)),np.reshape(dag060_3NN,(4,1)),np.reshape(dag060_4NN,(4,1)),np.reshape(dag060_5NN,(4,1)),np.reshape(dag060_6NN,(4,1)),np.reshape(ddf060_1NN,(4,1)),np.reshape(ddf060_2NN,(4,1)),np.reshape(ddf060_3NN,(4,1))),axis=0) 
e120NN=np.concatenate((np.reshape(dag120_1NN,(4,1)),np.reshape(dag120_2NN,(4,1)),np.reshape(dag120_3NN,(4,1)),np.reshape(dag120_4NN,(4,1)),np.reshape(dag120_5NN,(4,1)),np.reshape(dag120_6NN,(4,1)),np.reshape(ddf120_1NN,(4,1)),np.reshape(ddf120_2NN,(4,1)),np.reshape(ddf120_3NN,(4,1))),axis=0) 
e180NN=np.concatenate((np.reshape(dag180_1NN,(4,1)),np.reshape(dag180_2NN,(4,1)),np.reshape(dag180_3NN,(4,1)),np.reshape(dag180_4NN,(4,1)),np.reshape(dag180_5NN,(4,1)),np.reshape(dag180_6NN,(4,1)),np.reshape(ddf180_1NN,(4,1)),np.reshape(ddf180_2NN,(4,1)),np.reshape(ddf180_3NN,(4,1))),axis=0) 

ehbPHY_MOD=np.concatenate((np.reshape(dhb_1PHY_MOD,(4,1)),np.reshape(dhb_2PHY_MOD,(4,1))),axis=0)      
emwPHY_MOD=np.reshape(dmwPHY_MOD,(4,1))
eraPHY_MOD=np.concatenate((np.reshape(dra_1PHY_MOD,(4,1)),np.reshape(dra_2PHY_MOD,(4,1))),axis=0)  
eciaPHY_MOD=np.concatenate((np.reshape(dcia_1PHY_MOD,(4,1)),np.reshape(dcia_2PHY_MOD,(4,1))),axis=0)  
eslrPHY_MOD=np.concatenate((np.reshape(dslr_1PHY_MOD,(4,1)),np.reshape(dslr_2PHY_MOD,(4,1)),np.reshape(dslr_3PHY_MOD,(4,1))),axis=0)  
ewrPHY_MOD=np.concatenate((np.reshape(dwr_1PHY_MOD,(4,1)),np.reshape(dwr_2PHY_MOD,(4,1)),np.reshape(dwr_3PHY_MOD,(4,1))),axis=0)  
e030PHY_MOD=np.concatenate((np.reshape(dag030_1PHY_MOD,(4,1)),np.reshape(dag030_2PHY_MOD,(4,1)),np.reshape(dag030_3PHY_MOD,(4,1)),np.reshape(dag030_4PHY_MOD,(4,1)),np.reshape(dag030_5PHY_MOD,(4,1)),np.reshape(dag030_6PHY_MOD,(4,1)),np.reshape(ddf030_1PHY_MOD,(4,1)),np.reshape(ddf030_2PHY_MOD,(4,1)),np.reshape(ddf030_3PHY_MOD,(4,1))),axis=0)  
e060PHY_MOD=np.concatenate((np.reshape(dag060_1PHY_MOD,(4,1)),np.reshape(dag060_2PHY_MOD,(4,1)),np.reshape(dag060_3PHY_MOD,(4,1)),np.reshape(dag060_4PHY_MOD,(4,1)),np.reshape(dag060_5PHY_MOD,(4,1)),np.reshape(dag060_6PHY_MOD,(4,1)),np.reshape(ddf060_1PHY_MOD,(4,1)),np.reshape(ddf060_2PHY_MOD,(4,1)),np.reshape(ddf060_3PHY_MOD,(4,1))),axis=0) 
e120PHY_MOD=np.concatenate((np.reshape(dag120_1PHY_MOD,(4,1)),np.reshape(dag120_2PHY_MOD,(4,1)),np.reshape(dag120_3PHY_MOD,(4,1)),np.reshape(dag120_4PHY_MOD,(4,1)),np.reshape(dag120_5PHY_MOD,(4,1)),np.reshape(dag120_6PHY_MOD,(4,1)),np.reshape(ddf120_1PHY_MOD,(4,1)),np.reshape(ddf120_2PHY_MOD,(4,1)),np.reshape(ddf120_3PHY_MOD,(4,1))),axis=0)     
e180PHY_MOD=np.concatenate((np.reshape(dag180_1PHY_MOD,(4,1)),np.reshape(dag180_2PHY_MOD,(4,1)),np.reshape(dag180_3PHY_MOD,(4,1)),np.reshape(dag180_4PHY_MOD,(4,1)),np.reshape(dag180_5PHY_MOD,(4,1)),np.reshape(dag180_6PHY_MOD,(4,1)),np.reshape(ddf180_1PHY_MOD,(4,1)),np.reshape(ddf180_2PHY_MOD,(4,1)),np.reshape(ddf180_3PHY_MOD,(4,1))),axis=0)     

'''FInal results from analysis (WhONet and the Physical Model)'''
bmw=np.concatenate((label, bmwNN,bmwPHY_MOD),axis=1)#Motorway Scenario
bmw=np.concatenate((['Performance Metric','WhONet', 'Physical Model'],bmw),axis=0)
bra=np.concatenate((label1,braNN,braPHY_MOD),axis=1)#Roundabout Scenario
bcia=np.concatenate((label1,bciaNN,bciaPHY_MOD),axis=1)#Quick changes in acceleration Scenario
bhb=np.concatenate((label1,bhbNN,bhbPHY_MOD),axis=1)#hard brake Scenario
bslr=np.concatenate((label2,bslrNN,bslrPHY_MOD),axis=1)#successive left right turn Scenario
bwr=np.concatenate((label2,bwrNN,bwrPHY_MOD),axis=1)#wet road Scenario
b30=np.concatenate((label3,b30NN,b30PHY_MOD),axis=1)#30 seconds Scenario
b60=np.concatenate((label3,b60NN,b60PHY_MOD),axis=1)#60 seconds Scenario
b120=np.concatenate((label3,b120NN,b120PHY_MOD),axis=1)#120 seconds Scenario
b180=np.concatenate((label3,b180NN,b180PHY_MOD),axis=1)#180 seconds Scenario
