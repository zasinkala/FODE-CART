# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 09:39:32 2020

@author: Zachariah Sinkala

Schematic description of the model structure. CAR T cells proliferate, have a 
cytotoxic effect on tumor cells, differentiate into memory cell, and die 
naturally or due to immunosuppressive mechanisms. Memory T cells are readily 
responsive to the tumor associated antigen so that, when they interact with 
tumor cells, they differentiate into effector T cells, producing a rapid 
response of the immune system.
"""

###############################################################################
# import libraries

import numpy as np
from scipy import interpolate
from scipy.optimize import minimize
from fodeint import  fodeint
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd



###############################################################################
#Set the size of a figures

from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 6)
rcParams['legend.fontsize'] = 16
rcParams['axes.labelsize'] = 16
plt.style.context("seaborn-paper")


##############################################################################
# optimizers

optimizers=['leastsq','Nelder-Mead','SLSQP','emcee','ampgo']


###########################################################################
#######  Load data  #######################################################
#Load the CSV into a DataFrame

df=pd.read_csv("GBMEGFP.csv") 

Td1=df.values[:,0]
Td=df['time'].values
Cd1=df.values[:,1]/1e6
Cd=df['TEGFP'].values/1e6

##############################################################################
## Model steps
##############################################################################
start_time=Td[0]
end_time=Td[-1]
intervals=1000
mt=np.linspace(start_time,end_time,intervals) 

##############################################################################
## D describes the tumor-cell lysis rate (rate at which tumor cells 
## are killed by effector cells
##############################################################################

##############################################################################
## mCART Lysis of glioma U87 cells
##############################################################################
def D(C,G,w):
    d= 2.0**?? #2.23079264**??
    s= 0.7170**??#0.05686372**??
    l= 1.3973**??#2.23079264**??
    return d*C**l/(w**??+s*G**l+C**l)
    

#w describe the search time which is incorporated to represents tumor cell lysis by CAR T cells
      #When the search time is positive, the effectiveness of tumour cell lysis decreases



## System
###############################################################################
def eq(par,initial_cond,start_time,end_time,incr):
    t=np.linspace(start_time,end_time,incr)
    def funct(y,t):
        for i in range(len(y)):
            if y[i]<0:
                y[i]=0
        G,C_T,C_M,F,T_??,M ,X= y
        #??_2,e_1,??_T,b_1,??_M=par
        #??_D,K,??_M,??_2,e_1,??_T,b_1,??_M=par
        ??_M,??,??_2,e_1,??_T,??_M=par
        DD=D(C_T,G,w= 0.01939177)#0.0001#.3251;0.24
        #dos=dosage
                
        dG = r**??*G*(1-(G/K_G**??))\
            -( 1/(( e_1**??/290**??*initial_cond[1]**??)+T_?? ) )*DD*G\
            -( (??_11)**??/((e_1**??/290**??*initial_cond[1]**??)+T_??) )*(G/(G+k_1**??))*M       #  G Glioma cells
        dC_T=??_D**??*(( DD**2*G**2)/(K**??+DD**2*G**2))*C_T-(??_M)**??*C_T+??_M**??*G*C_M-??**??*G*C_T-??_T**??*C_T 
        dC_M =(??*??_M)**??*C_T-??_M**??*G*C_M-??_M**??*C_M
        dF   = (??_0)**??+??_1**??*C_T/(C_T+K_F**??) -  ??_F**??*F 
        dT_??= b_1**??*G - ??_2**??*T_??
        dM =r_M**??*M*(1-(M/K_M**??))+ a_1*F/(k_4 **??+ F)*(1/(e_2**??+T_??))-??_3**??*(G/(G+k_2**??))*M
        dX=(C_T-f(t))**2
        return [dG,dC_T,dC_M, dF,dT_??,dM,dX]
    #integrate 
    ds=fodeint(??,funct,initial_cond,t)
    return (ds[:,0],ds[:,1],ds[:,2],ds[:,3],ds[:,4],ds[:,5],ds[:,6],t)
###############################################################################

def eq2(par,initial_cond,start_time,end_time,incr):
    t=np.linspace(start_time,end_time,incr)
    def funct(y,t):
        for i in range(len(y)):
            if y[i]<0:
                y[i]=0
        G,C_T,C_M,F,T_??,M = y
        ??_M,??,??_2,e_1,??_T,??_M=par
        DD=D(C_T,G,w=0.01939177)#0.0001 w=.3251;0.24
        #dos=dosage
        ######################################################################        
        
                          
        dG = r**??*G*(1-(G/K_G**??))\
            -(1/((e_1/290**??*initial_cond[1]**??)+T_??))*DD*G\
                -((??_11)**??/((e_1/290**??*initial_cond[1]**??)+T_??))*(G/(G+k_1**??))*M  
        dC_T=??_D**??*(( DD**2*G**2)/(K**??+DD**2*G**2))*C_T-??_M**??*C_T\
            +??_M**??*G*C_M-??**??*G*C_T-??_T**??*C_T 
        dC_M =(??*??_M)**??*C_T-??_M**??*G*C_M-??_M**??*C_M
        dF   = (??_0)**??+??_1**??*C_T/(C_T+K_F**??) -  ??_F**??*F 
        dT_??= b_1**??*G - ??_2**??*T_??
        dM =r_M**??*M*(1-(M/K_M**??))+ a_1*(F/(k_4 **??+ F))*(1/(e_2**??+T_??))\
            -??_3**??*(G/(G+k_2**??))*M
        return [dG,dC_T,dC_M, dF,dT_??,dM]#,dX]
    
    
          
    #integrate 
    ds=fodeint(??,funct,initial_cond,t)
    
    return (ds[:,0],ds[:,1],ds[:,2],ds[:,3],ds[:,4],ds[:,5],t)


# ############################################################################
# # Represent data in dataframe into array
# ############################################################################
vdata=df.values[:,1]
teval=df.values[:,0]
time=np.linspace(start_time,end_time,intervals)
# Final conditions
pp = np.zeros(intervals); pp[-1] = 1.0
final=pp
##############################################################################
# W need time steps equally space
##############################################################################



data=np.column_stack((vdata))
##############################################################################
# fractional parameter
##############################################################################


r=0.53948089 #Fixed
r_M=0.3307*24 #Fixed
K_G=0.64625879 #Fixed
K_M=1.0        #Fixed
a_1=0.1163*24*1e-9 #Fixed
k_4=1.05e-2   #Fixed
e_2=1e-2     #Fixed
k_1=2.7e-3 #Fixed 2.7e-2
k_2=2.7e-2 #Fixed
??_11=1.5*24*1e-6 #Fixed
??_3=0.0194*24 #Fixed
??_0=0.2 #Fixed
??_1=0.1 #Fixed
??_M=4.5e-2 #Fixed
??=3.0 #fixed
??_D=0.265 #Fixed 0.265
K=0.05 #Fixed
??_M=6.0 #Tuning
??=4.5e-2 # Tuning
??_M=0.05 # Tuning
K_F=0.095665 #Fixed
??_F= 2.448   # Fixed
??_2=0.409    #Tuning
e_1=2.05#0.266/(0.1*.20) #Tuning
??_T=0.215      #Tuning
b_1=8.4962 #Fixed
dosage=10

###############################################################################
rates=(??_M,??,??_2,e_1,??_T,??_M)


##############################################################################
f = interpolate.interp1d(Td,Cd )
tnew = np.arange(Td[0], Td[-1], 1)
ynew = f(tnew)   # use interpolation function returned by `interp1d`
plt.plot(Td, Cd, 'o', tnew, ynew, '-')
plt.show()

Td=tnew
Cd=ynew

obs=100
for ?? in [0.7715522030079129]:#??=0.7715522030079129

    ##########################################################
    ## Model index to compare data
    ###########################################################
    findindex=lambda x:np.where(mt>=x)[0][0]
    mindex=list(map(findindex,Td))
    
    IC=[0.5,Cd[0],0.00,0,0,0,0]
    F0,F1,F2,F3,F4,F5,F6,T=eq(rates,IC,Td[0],Td[-1],1000)
    
    
    plt.figure(figsize=(8,5))
    plt.plot(T,F1,'-b')
    plt.plot(Td,Cd,'go')
    plt.show()
    
    ##########################################################################
    ## Score fit of the system
    ##########################################################################
    
    def score(parms):
        # Get Solution to system
        IC=[0.5,Cd[0],0.00,0,0,0,0]
        
        F0,F1,F2,F3,F4,F5,F6,T=eq(parms,IC,start_time,end_time,intervals)
        # Pick model points to compare
        #Cm=F1[mindex]
        # Score the difference between model and data points
        #ss=lambda data, model:((data-model)**2).sum()
        return np.sum(final*F6)#ss(Cd,Cm)
    
    
   
    fit_score=score(rates)
    opt=minimize(score,rates,method='Nelder-Mead')#,bounds=bounds)
    bestrates=opt.x
    bestscore=opt.fun
    ??_M,??,??_2,e_1,??_T,??_M=opt.x
    newrates=(??_M,??,??_2,e_1,??_T,??_M)
    
    NIC=[0.46828865,Cd[0],0.00,0,0,0,0]
    ##########################################################################
    ## Generate solution to system
    ###########################################################################
    F0,F1,F2,F3,F4,F5,F6,T=eq(newrates,IC,start_time,end_time,intervals)
    Cm=F1[mindex]
    Tm=T[mindex]
    
    
    plt.figure()
    plt.plot(T,F1,'b-',Td1,Cd1,'go')
    plt.xlabel('Days')
    plt.ylabel('$10^6$ cells')
    plt.title('CAR T Cells simulation and data')
    plt.show()
    plt.figure(figsize=(9,6))
    plt.plot(T,F1,'b-',Tm,Cm,'ro',Td,Cd,'go')
    plt.xlabel('Days')
    plt.ylabel('$10^6$ cells')
    plt.title('CAR T Cells simulation and data')
    plt.show()
    
    ##########################################################################
    NIC10=[0.4682885,Cd[0],0.00,0,0,0,0]
    F0,F1,F2,F3,F4,F5,F6,T=eq(newrates,NIC10,start_time,end_time,intervals)

    #plot data and fitted curves
    fig = plt.figure()
    plt.plot(Td, Cd, 'go',label='data')
    plt.plot(T, F1, 'g-', linewidth=2,label='Sim')
    plt.xlabel('Days')
    plt.ylabel('$10^6$ Cells')
    plt.legend()
    plt.grid
    plt.title('CAR T Cells')
    plt.show()
    
    ##########################################################################
    #Effector  cART cells
    
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.01,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY01,start_time,end_time,intervals)        
    ax.plot(T, F1, 'g-', linewidth=2,label='Small dose')
    ax.plot(T, G1, 'r-', linewidth=2,label='Medium dose')
    ax.plot(T, H1, 'b-', linewidth=2,label='Small dose')
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time=T[-1]+20
    YY01=[F0[-1],0.10,F2[-1],F3[-1],F4[-1],F5[-1]]
    YY02=[G0[-1],1,G2[-1],G3[-1],G4[-1],G5[-1]]
    YY03=[H0[-1],10,H2[-1],H3[-1],H4[-1],H5[-1]]
    
    
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    
    ax.plot(T, F1, 'g-')
    ax.plot(T, G1, 'r-')
    ax.plot(T, H1, 'b-')
    
    ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    ax.axvline(x=10,linestyle="dashed",label='t=10')
    ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$10^6$ Cells')
    
    ax.set_title('Effector CAR T Cells')
    ax.legend()#["Small dose","Medium dose","Maximum dose"])
    plt.show()
    
    ##########################################################################
    #Memory CAR T cells
    
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.05,0,0.0000,0,1e-7,0.000]
    YY02=[0.05,0,0.0000,0,1e-7,0.000]
    YY03=[0.05,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    

    ax.plot(T, F2, 'g-', linewidth=2,label='Small dose')
    ax.plot(T, G2, 'r-', linewidth=2,label='Medium dose')
    ax.plot(T, H2, 'b-', linewidth=2,label='Maximum dose')
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time =T[-1] +20
    YY01=[F0[-1],.10,F2[-1],F3[-1],F4[-1],F5[-1]]
    YY02=[G0[-1],1.0,G2[-1],G3[-1],G4[-1],G5[-1]]
    YY03=[H0[-1],10.0,H2[-1],H3[-1],H4[-1],H5[-1]]

    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    
    
    
    
    ax.plot(T, F1, 'g-')
    ax.plot(T, G1, 'r-')
    ax.plot(T,H1, 'b-')
    
    ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    ax.axvline(x=10,linestyle="dashed",label='t=10')
    ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$10^6$ Cells')
    
    ax.set_title('Memory CAR T Cells')
    ax.legend()
    plt.show()
    #fig.savefig('KRCMCT42.png')
    
    ##########################################################################
    # Glioma Cells
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.01,0,0.0000,0,1e-7,0.000]
    YY02=[0.01,0,0.0000,0,1e-7,0.000]
    YY03=[0.01,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    
    ax.plot(T, F0, 'g-', linewidth=2,label='Small dose')
    ax.plot(T, G0, 'r-', linewidth=2,label='Medium dose')
    ax.plot(T, H0, 'b-', linewidth=2,label='Maximum dose')

    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time=T[-1]+20
    YY01=[F0[-1],0.10,F2[-1],F3[-1],F4[-1],F5[-1]]
    YY02=[G0[-1],1.0,G2[-1],G3[-1],G4[-1],G5[-1]]
    YY03=[H0[-1],10.0,H2[-1],H3[-1],H4[-1],H5[-1]]
    
        
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    
    
    ax.plot(T, F0, 'g-')
    ax.plot(T, G0, 'r-')
    ax.plot(T, H0, 'b-')
    
    ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    ax.axvline(x=10,linestyle="dashed",label='t=10')
    ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$10^6$ Cells')
    
    ax.set_title('Glioma Cells')
    ax.legend()#["Small dose","Medium dose","Maximum dose"])
    plt.show()
    #fig.savefig('Glioma42.png')

    #######################################################################
    # Macrophages
    
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.05,0,0.0000,0,1e-7,0.000]
    YY02=[0.05,0,0.0000,0,1e-7,0.000]
    YY03=[0.05,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    


    ax.plot(T, F5, 'g-', linewidth=2,label='Small dose')
    ax.plot(T, G5, 'r-', linewidth=2,label='Medium dose')
    ax.plot(T, H5, 'b-', linewidth=2,label='Maximum dose')
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time=T[-1]+20

    YY01=[F0[-1],0.10,F2[-1],F3[-1],F4[-1],F5[-1]]
    YY02=[G0[-1],1,G2[-1],G3[-1],G4[-1],G5[-1]]
    YY03=[H0[-1],10,H2[-1],H3[-1],H4[-1],H5[-1]]
    
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    

    ax.plot(T, F5, 'g-')
    ax.plot(T, G5, 'r-')
    ax.plot(T, H5, 'b-')

    ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    ax.axvline(x=10,linestyle="dashed",label='t=10')
    ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$10^6$ Cells')
    
    ax.set_title('Macrophages')
    ax.legend()
    plt.show()
    #fig.savefig('KRCMCT42.png')
    
    #####################################################################
    # Immune Inihibitor factor
    
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.01,0,0.0000,0,1e-7,0.000]
    YY02=[0.01,0,0.0000,0,1e-7,0.000]
    YY03=[0.01,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    


    ax.plot(T, F4, 'g-', linewidth=2,label='Small dose')
    ax.plot(T, G4, 'r-', linewidth=2,label='Medium dose')
    ax.plot(T, H4, 'b-', linewidth=2,label='Maximum dose')
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time=T[-1]+20

    YY01=[F0[-1],0.10,F2[-1],F3[-1],F4[-1],F5[-1]]
    YY02=[G0[-1],1,G2[-1],G3[-1],G4[-1],G5[-1]]
    YY03=[H0[-1],10,H2[-1],H3[-1],H4[-1],H5[-1]]
    
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    

    ax.plot(T, F4, 'g-')
    ax.plot(T, G4, 'r-')
    ax.plot(T, H4, 'b-')

    ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    ax.axvline(x=10,linestyle="dashed",label='t=10')
    ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$\mu$ Grams')
    
    ax.set_title('Immune Inihibitor factor')
    ax.legend()
    plt.show()
    #fig.savefig('ImmuneInhibitor42.png')


    
    ###################################################################
    #Immune stimulator factor
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.01,0,0.0000,0,1e-7,0.000]
    YY02=[0.01,0,0.0000,0,1e-7,0.000]
    YY03=[0.01,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    


    ax.plot(T, F3, 'g-', linewidth=2,label='Small dose')
    ax.plot(T, G3, 'r-', linewidth=2,label='Medium dose')
    ax.plot(T, H3, 'b-', linewidth=2,label='Maximum dose')
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time=T[-1]+20

    YY01=[F0[-1],0.10,F2[-1],F3[-1],F4[-1],F5[-1]]
    YY02=[G0[-1],1,G2[-1],G3[-1],G4[-1],G5[-1]]
    YY03=[H0[-1],10,H2[-1],H3[-1],H4[-1],H5[-1]]
    
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    

    ax.plot(T, F3, 'g-')
    ax.plot(T, G3, 'r-')
    ax.plot(T, H3, 'b-')

    ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    ax.axvline(x=10,linestyle="dashed",label='t=10')
    ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$\mu$ Grams')
    
    ax.set_title('Immune stimulator factor')
    ax.legend()
    plt.show()
    #fig.savefig('Immunestimulator42.png')
    

   
    

    ###########################################################################################
    ##Dynamics of Glioma, Effector CAR T, and Memory CAR T cells with small dose treatment
    ###########################################################################################
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY0=[0.01,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    
    


    ax.plot(T, F0, 'g-', linewidth=2,label='Glioma cells')
    ax.plot(T, F1, 'r-', linewidth=2,label='Effector CAR T cells')
    ax.plot(T, F2, 'b-', linewidth=2,label='Memory CAR T Cells')
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time=T[-1]+obs

    YY0=[F0[-1],.10,F2[-1],F3[-1],F4[-1],F5[-1]]
    
    
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    
    

    ax.plot(T, F0, 'g-')
    ax.plot(T, F1, 'r-')
    ax.plot(T, F2, 'b-')

    # ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    # ax.axvline(x=10,linestyle="dashed",label='t=10')
    # ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$10^6$ Cells')
    
    ax.set_title('Dynamics of Glioma, Effector CAR T, and Memory CAR T cells with small dose treatment')
    ax.legend()
    plt.show()
    #########################################################################################
    #Dynamics of Glioma, Effector CAR T, and Memory CAR T cells with medimum dose treatment
    ###########################################################################################
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY0=[0.05,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    
    


    ax.plot(T, F0, 'g-', linewidth=2,label='Glioma cells')
    ax.plot(T, F1, 'r-', linewidth=2,label='Effector CAR T cells')
    ax.plot(T, F2, 'b-', linewidth=2,label='Memory CAR T Cells')
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time=T[-1]+obs

    YY0=[F0[-1],1,F2[-1],F3[-1],F4[-1],F5[-1]]
    
    
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    
    

    ax.plot(T, F0, 'g-')
    ax.plot(T, F1, 'r-')
    ax.plot(T, F2, 'b-')

    # ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    # ax.axvline(x=10,linestyle="dashed",label='t=10')
    # ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$10^6$ Cells')
    
    ax.set_title('Dynamics of Glima, Effector CAR T, and Memory CAR T cells with medium dose treatment')
    ax.legend()
    plt.show()
    #fig.savefig('GEMmedium.png')
    
    #######################################################################################
    #Dynamics of Glioma, Effector CAR T, and Memory CAR T cells with maximum dose treatment
    #######################################################################################
    
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY0=[0.01,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    
    


    ax.plot(T, F0, 'g-', linewidth=2,label='Glioma cells')
    ax.plot(T, F1, 'r-', linewidth=2,label='Effector CAR T cells')
    ax.plot(T, F2, 'b-', linewidth=2,label='Memory CAR T Cells')
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time=T[-1]+obs

    YY0=[F0[-1],10,F2[-1],F3[-1],F4[-1],F5[-1]]
    
    
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    
    

    ax.plot(T, F0, 'g-')
    ax.plot(T, F1, 'r-')
    ax.plot(T, F2, 'b-')

    # ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    # ax.axvline(x=10,linestyle="dashed",label='t=10')
    # ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$10^6$ Cells')
    
    ax.set_title('Dynamics of Glioma, Effector CAR T, and Memory CAR T cells with maximum dose treatment')
    ax.legend()
    plt.show()
    #fig.savefig('GEMmaximum.png')
    
   
    





   

   