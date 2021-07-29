# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 22:03:11 2019

@author: onyekpeu
"""

import numpy as np

from  function_fileswsdisperrcsv2 import *
from IO_VNB_Dataset import *


'Parameters'
dropout=0.05
input_dim = 4*10
output_dim = 1
num_epochs = 80
layer_dim = 1
learning_rate = 0.0007
batch_size =128
decay_rate=0
decay_steps=10000
momentum=0.8
samplefreq=10
Ts=int(samplefreq*1*1)
seq_dim=int(1*(10/Ts))
l1_=0
l2_=0
h2 = 72#neurons in hidden layers
Z=1
outage1=100 #challenging scenario
outage2=300 # 30s outage scenario
outage3=600 #60s outage scenario
outage4=900
outage5=1200 # 120s outage scenario
outage6=1800 #180s outage scenario
number_of_runs=1

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
SLRdat2=[V_Vw7]
SLRdat3=[V_Vw8]  

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

amx, amn, rad1, rad2= maxmin17(V_Vw12,Ts, Acc1_bias, gyro1_bias)

'''Data processing for training'''
gtr,itr,x, y=data_process13t(TrainDat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)

'''Data processing for evaluation'''
#Motorway Scenario
gtmw,itmw,xtmw, ytmw=data_process13t(MWdat, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2,  Z, mode)
#Roundabout Scenario    
gtra1,itra1,xtra1, ytra1=data_process13t(RAdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2,   Z, mode)
gtra2,itra2,xtra2, ytra2=data_process13t(RAdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2,  Z, mode,)
#Hard brake Scenario
gthb1,ithb1,xthb1, ythb1=data_process13t(HBdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode) 
gthb2,ithb2,xthb2, ythb2=data_process13t(HBdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)  
#Quick changes in acceleration Scenario
gtcia1,itcia1,xtcia1, ytcia1=data_process13t(CIAdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode) 
gtcia2,itcia2,xtcia2, ytcia2=data_process13t(CIAdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2,Z, mode)       
#Sharp cornering and successive left and right turns Scenario
gtslr1,itslr1,xtslr1, ytslr1=data_process13t(SLRdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)           
gtslr2,itslr2,xtslr2, ytslr2=data_process13t(SLRdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)  
gtslr3,itslr3,xtslr3, ytslr3=data_process13t(SLRdat3, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)       
#Wet Road Scenario
gtwr1,itwr1,xtwr1, ytwr1=data_process13t(WRdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)      
gtwr2,itwr2,xtwr2, ytwr2=data_process13t(WRdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode) 
gtwr3,itwr3,xtwr3, ytwr3=data_process13t(WRdat3, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode) 
 #Longer-term GNSS Outages Scenario   
gtag1,itag1,xtag1, ytag1=data_process13t(AGdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)
gtag2,itag2,xtag2, ytag2=data_process13t(AGdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)
gtag3,itag3,xtag3, ytag3=data_process13t(AGdat3, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)
gtag4,itag4,xtag4, ytag4=data_process13t(AGdat4, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)
gtag5,itag5,xtag5, ytag5=data_process13t(AGdat5, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)
gtag6,itag6,xtag6, ytag6=data_process13t(AGdat6, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)
gtdf1,itdf1,xtdf1, ytdf1=data_process13t(DFdat1, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)
gtdf2,itdf2,xtdf2, ytdf2=data_process13t(DFdat2, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)
gtdf3,itdf3,xtdf3, ytdf3=data_process13t(DFdat3, seq_dim, input_dim, output_dim, Ts, Acc1_bias, Acc2_bias, gyro1_bias, batch_size, amx, amn, rad1, rad2, Z, mode)


#############################################################################
#############################################################################
#############################################################################
'RNN TRAINING'
#############################################################################
#############################################################################
#############################################################################

Run_time, regress=RNN_model(np.array(x),np.array(y), input_dim,output_dim, seq_dim, batch_size, num_epochs, dropout, h2, learning_rate, l1_, l2_, nfr, decay_rate, momentum, decay_steps)

'Evaluation'
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

print('-----------------Results---------------')
print('Motorway V_Vw12:',np.round(perf_metr_motorway_crse_NN,2))

print('Roundabout V_Vta11:',np.round(perf_metr_roundabout1_crse_NN,2))
print('Roundabout V_Vfb02d:',np.round(perf_metr_roundabout2_crse_NN,2))


print('Quick Change in Acceleration V_Vfb02e:',np.round(perf_metr_change_in_acc1_crse_NN,2))
print('Quick Change in Acceleration V_Vta12:',np.round(perf_metr_change_in_acc2_crse_NN,2))


print('Hard Brake V_Vw16b:',np.round(perf_metr_hard_brake1_crse_NN,2))
print('Hard Brake V_Vw17:',np.round(perf_metr_hard_brake2_crse_NN,2))

print('Wet Road V_Vtb8:',np.round(perf_metr_wet_road1_crse_NN,2))
print('Wet Road V_Vtb11:',np.round(perf_metr_wet_road2_crse_NN,2))
print('Wet Road V_Vtb13:',np.round(perf_metr_wet_road3_crse_NN,2))

print('Successive Left and Right Turns V_Vw6:',np.round(perf_metr_suc_lft_rht_turn1_crse_NN,2))
print('Successive Left and Right Turns V_Vw7:',np.round(perf_metr_suc_lft_rht_turn1_crse_NN,2))
print('Successive Left and Right Turns V_Vw8:',np.round(perf_metr_suc_lft_rht_turn1_crse_NN,2))


print('180 s GNSS Outage (V_Vtb3)', np.round(perf_metr_V_Vtb3_crse_NN_180s,2))
print('180 s GNSS Outage (V_Vfb01c)' , np.round(perf_metr_V_Vfb01c_crse_NN_180s,2))
print('180 s GNSS Outage (V_Vfb02a)' , np.round(perf_metr_V_Vfb02a_crse_NN_180s,2))
print('180 s GNSS Outage (V_Vta1a)', np.round(perf_metr_V_Vta1a_crse_NN_180s,2))  
print('180 s GNSS Outage (V_Vfb02b)', np.round(perf_metr_V_Vfb02b_crse_NN_180s,2))
print('180 s GNSS Outage (V_Vfb02g)' , np.round(perf_metr_V_Vfb02g_crse_NN_180s,2))
print('180 s GNSS Outage (V_St6)' , np.round(perf_metr_V_St6_crse_NN_180s,2))
print('180 s GNSS Outage (V_St7)', np.round(perf_metr_V_St7_crse_NN_180s,2))    
print('180 s GNSS Outage (V_S3a)', np.round(perf_metr_V_S3a_crse_NN_180s,2))    


print('180 s GNSS Outage (Alpha1)', np.round(perf_metr_alpha1_crse_NN_180s,2))
print('180 s GNSS Outage (Alpha2)', np.round(perf_metr_alpha2_crse_NN_180s,2))
print('180 s GNSS Outage (Alpha3)', np.round(perf_metr_alpha3_crse_NN_180s,2))
print('180 s GNSS Outage (Alpha4)', np.round(perf_metr_alpha4_crse_NN_180s ,2))   

print('180 s GNSS Outage (Bravo1)', np.round(perf_metr_bravo1_crse_NN_180s,2))
print('180 s GNSS Outage (Bravo2)', np.round(perf_metr_bravo2_crse_NN_180s,2))
print('180 s GNSS Outage (Bravo3)', np.round(perf_metr_bravo3_crse_NN_180s,2))

print('180 s GNSS Outage (Charlie1)', np.round(perf_metr_charlie1_crse_NN_180s,2))
print('180 s GNSS Outage (Charlie2)', np.round(perf_metr_charlie2_crse_NN_180s,2))
print('180 s GNSS Outage (Charlie3)', np.round(perf_metr_charlie3_crse_NN_180s,2))
print('180 s GNSS Outage (Charlie4)', np.round(perf_metr_charlie4_crse_NN_180s,2))     

print('180 s GNSS Outage (Delta1)', np.round(perf_metr_delta1_crse_NN_180s,2))
print('180 s GNSS Outage (Delta2)', np.round(perf_metr_delta2_crse_NN_180s,2))
print('180 s GNSS Outage (Delta3)', np.round(perf_metr_delta3_crse_NN_180s,2))

print('*******************************************')
