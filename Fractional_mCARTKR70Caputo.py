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




##############################################################################
#Set the size of a figures
from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 6)
rcParams['legend.fontsize'] = 16
rcParams['axes.labelsize'] = 16
plt.style.context("seaborn-paper")


###################################################################
#optimizers
optimizers=['leastsq','Nelder-Mead','SLSQP','emcee','ampgo']


###########################################################################
#######  Load data  #######################################################
## Data used for inferences
## We used data from [8] and [5] for parameter inference. We extracted data
## from these sources using the free software G3Data Graph Analyzer [3] and
## organized them in the tables shown below
###########################################################################

#Load the CSV into a DataFrame
df=pd.read_csv("GBMEGFP.csv") 

Td1=df.values[:,0]
Td=df['time'].values
Cd1=df.values[:,1]/1e6
Cd=df['TEGFP'].values/1e6

#################################################
## Model steps
#####################################################
start_time=Td[0]
end_time=Td[-1]
intervals=1000
mt=np.linspace(start_time,end_time,intervals) 

############################################################################
## D describes the tumor-cell lysis rate (rate at which tumor cells 
## are killed by effector cells
############################################################################

################################################################
## mCART Lysis of glioma KR70 cells
###################################################################
def D(C,G,w):
    d= 1.02157418**θ#1.9667**θ#
    s= 0.00469810**θ#0.529**θ#
    l= 1.34552076**θ#0.5**θ#
    return d*C**l/(w**θ+s*G**l+C**l)
    
#w describe the search time which is incorporated to represents tumor cell lysis by CAR T cells
      #When the search time is positive, the effectiveness of tumour cell lysis decrease

##############################################################################


## System
#######################################################
def eq(par,initial_cond,start_time,end_time,incr):
    t=np.linspace(start_time,end_time,incr)
    def funct(y,t):
        for i in range(len(y)):
            if y[i]<0:
                y[i]=0
        G,C_T,C_M,F,T_β,M ,X= y
       
        θ_M,α,μ_2,e_1,μ_T,α_M=par
        DD=D(C_T,G,w=.3251)#0.0001#.3251;0.24 0.02603970
        #dos=dosage
                
        dG = r**θ*G*(1-(G/K_G**θ))\
            -( 1/(( e_1**θ/290**θ*initial_cond[1]**θ)+T_β ) )*DD*G\
            -( (α_11)**θ/((e_1**θ/290**θ*initial_cond[1]**θ)+T_β) )*(G/(G+k_1**θ))*M       #  G Glioma cells
        dC_T=α_D**θ*(( DD**2*G**2)/(K**θ+DD**2*G**2))*C_T-(α_M)**θ*C_T+θ_M**θ*G*C_M-α**θ*G*C_T-μ_T**θ*C_T 
        dC_M =(ϵ*α_M)**θ*C_T-θ_M**θ*G*C_M-μ_M**θ*C_M
        dF   = (α_0)**θ+α_1**θ*C_T/(C_T+K_F**θ) -  μ_F**θ*F 
        dT_β= b_1**θ*G - μ_2**θ*T_β
        dM =r_M**θ*M*(1-(M/K_M**θ))+ a_1*F/(k_4 **θ+ F)*(1/(e_2**θ+T_β))-α_3**θ*(G/(G+k_2**θ))*M
        dX=(C_T-f(t))**2
        return [dG,dC_T,dC_M, dF,dT_β,dM,dX]
    #integrate 
    ds=fodeint(θ,funct,initial_cond,t)
   
    return (ds[:,0],ds[:,1],ds[:,2],ds[:,3],ds[:,4],ds[:,5],ds[:,6],t)
######################################################################################

def eq2(par,initial_cond,start_time,end_time,incr):
    t=np.linspace(start_time,end_time,incr)
    def funct(y,t):
        for i in range(len(y)):
            if y[i]<0:
                y[i]=0
        G,C_T,C_M,F,T_β,M = y
        θ_M,α,μ_2,e_1,μ_T,α_M=par
        DD=D(C_T,G,w= .3251)#0.0001 w=.3251;0.24
             
        ####################################################
                          
        dG = r**θ*G*(1-(G/K_G**θ))\
            -(1/((e_1/290**θ*initial_cond[1]**θ)+T_β))*DD*G\
                -((α_11)**θ/((e_1/290**θ*initial_cond[1]**θ)+T_β))*(G/(G+k_1**θ))*M  
        dC_T=α_D**θ*(( DD**2*G**2)/(K**θ+DD**2*G**2))*C_T-α_M**θ*C_T\
            +θ_M**θ*G*C_M-α**θ*G*C_T-μ_T**θ*C_T 
        dC_M =(ϵ*α_M)**θ*C_T-θ_M**θ*G*C_M-μ_M**θ*C_M
        dF   = (α_0)**θ+α_1**θ*C_T/(C_T+K_F**θ) -  μ_F**θ*F 
        dT_β= b_1**θ*G - μ_2**θ*T_β
        dM =r_M**θ*M*(1-(M/K_M**θ))+ a_1*(F/(k_4 **θ+ F))*(1/(e_2**θ+T_β))\
            -α_3**θ*(G/(G+k_2**θ))*M
        return [dG,dC_T,dC_M, dF,dT_β,dM]#,dX]
    
    
          
    #integrate 
    ds=fodeint(θ,funct,initial_cond,t)
   
    return (ds[:,0],ds[:,1],ds[:,2],ds[:,3],ds[:,4],ds[:,5],t)

#########################################################################################
# ###################################################
# # Represent data in dataframe into array
# ######################################################
vdata=df.values[:,1]
teval=df.values[:,0]
time=np.linspace(start_time,end_time,intervals)
# Final conditions
pp = np.zeros(intervals); pp[-1] = 1.0
final=pp
######################################
# W need time steps equally space
########################################


#Arrays to stack
data=np.column_stack((vdata))

##############################################################################
# fractional parameter
##############################################################################

r=0.53948089 #Fixed 0.18879808191577216
r_M=0.3307*24 #Fixed
K_G=0.64625879 #Fixed 5.99999999995355
K_M=1.0        #Fixed
a_1=0.1163*24*1e-9 #Fixed
k_4=1.05e-2   #Fixed
e_2=1e-2     #Fixed
k_1=2.7e-3 #Fixed 2.7e-2
k_2=2.7e-2 #Fixed
α_11=1.5*24*1e-6 #Fixed
α_3=0.0194*24 #Fixed
α_0=0.2 #Fixed
α_1=0.1 #Fixed
μ_M=4.5e-2 #Fixed
ϵ=3.0 #fixed
α_D=0.265 #Fixed 0.265
K=0.05 #Fixed
θ_M=6.0 #Tuning
α=4.5e-2 # Tuning
α_M=0.05 # Tuning
K_F=0.095665 #Fixed
μ_F= 2.448   # Fixed
μ_2=0.409    #Tuning
e_1=2.05#0.266/(0.1*.20) #Tuning
μ_T=0.215      #Tuning
b_1=8.4962 #Fixed
s=290 # Fixed scale factor associated with  glioma cell cd70 expression
dosage=10

##################################################################
rates=(θ_M,α,μ_2,e_1,μ_T,α_M)


##################################################################
f = interpolate.interp1d(Td,Cd )
tnew = np.arange(Td[0], Td[-1], 1)
ynew = f(tnew)   # use interpolation function returned by `interp1d`
plt.plot(Td, Cd, 'o', tnew, ynew, '-')
plt.show()

Td=tnew
Cd=ynew

obs=100
for θ in [0.7715522030079129]:

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
    
    #################################################################
    ## Score fit of the system
    ####################################################################
    
    def score(parms):
        # Get Solution to system
        IC=[0.5,Cd[0],0.00,0,0,0,0]
        #IC=[0.46828865,Cd[0],0.00,0,0,0]
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
    θ_M,α,μ_2,e_1,μ_T,α_M=opt.x
    newrates=(θ_M,α,μ_2,e_1,μ_T,α_M)
    NIC=[0.46828865,Cd[0],0.00,0,0,0,0]
    ########################################################################
    ## Generate solution to system
    ########################################################################
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
    
    ####################################################################
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
    fig.savefig('Gliomahglioma.png')
   
   
    
    ###################################################################
    #For effector CAR T cells
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.1,0,0.0000,0,1e-7,0.000]
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
    
    #######################################################################
    # Memory CAR T cells dynamics for 25 days
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.1,0,0.0000,0,1e-7,0.000]
    YY02=[0.1,0,0.0000,0,1e-7,0.000]
    YY03=[0.1,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    

    ax.plot(T, F1, 'g-', linewidth=2,label='Small dose')
    ax.plot(T, G1, 'r-', linewidth=2,label='Medium dose')
    ax.plot(T, H1, 'b-', linewidth=2,label='Maximum dose')
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time =T[-1] +20
    YY01=[F0[-1],.10,F2[-1],F3[-1],F4[-1],F5[-1]]
    YY02=[G0[-1],1.0,G2[-1],G3[-1],G4[-1],G5[-1]]
    YY03=[H0[-1],10.0,H2[-1],H3[-1],H4[-1],H5[-1]]

    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals)
    G0,G1,G2,G3,G4,G5,T=eq2(newrates,YY02,start_time,end_time,intervals)
    H0,H1,H2,H3,H4,H5,T=eq2(newrates,YY03,start_time,end_time,intervals)
    
    
    
    
    ax.plot(T, F2, 'g-')
    ax.plot(T, G2, 'r-')
    ax.plot(T,H2, 'b-')
    
    ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    ax.axvline(x=10,linestyle="dashed",label='t=10')
    ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$10^6$ Cells')
    
    ax.set_title('Memory CAR T Cells')
    ax.legend()
    plt.show()
    
    
    
    #######################################################################
    #Memory cells dynamics for 100 days
    fig = plt.figure()
    start_time=0
    end_time=7
    YY0=[0.1,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    plt.plot(T, F2, 'g-', linewidth=2)
    
    start_time=T[-1]
    end_time=T[-1]+100
    YY0=[F0[-1],.10,F2[-1],F3[-1],F4[-1],F5[-1]]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    plt.plot(T, F2, 'g-', linewidth=2,label='Small dose')
    
    start_time=0
    end_time=7
    YY0=[0.1,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    plt.plot(T, F2, 'r-', linewidth=2)
    
    start_time=T[-1]
    end_time=T[-1]+100
    YY0=[F0[-1],1,F2[-1],F3[-1],F4[-1],F5[-1]]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    plt.plot(T, F2, 'r-', linewidth=2,label='Medium dose')
    
    start_time=0
    end_time=7
    YY0=[0.1,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    plt.plot(T, F2, 'b-', linewidth=2)
    
    start_time=T[-1]
    end_time=T[-1]+100
    YY0=[F0[-1],10,F2[-1],F3[-1],F4[-1],F5[-1]]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    plt.plot(T, F2, 'b-', linewidth=2,label='Large dose')


    plt.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    plt.axvline(x=10,linestyle="dashed",label='t=10')
    plt.axvline(x=17,linestyle="dashed",label='t=17')
    plt.xlabel('time[day]')
    plt.ylabel('$10^6$ Cells')
    plt.legend()
    plt.title('Memory cells')
    
    
   
    
    ######################################################################
    #Glioma cells
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.1,0,0.0000,0,1e-7,0.000]
    YY02=[0.1,0,0.0000,0,1e-7,0.000]
    YY03=[0.1,0,0.0000,0,1e-7,0.000]
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
    fig.savefig('Glioma42.png')

    #######################################################################

    #fig.savefig('CART42.png')
    
    #####################################################################
    #Macrophages
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.1,0,0.0000,0,1e-7,0.000]
    YY02=[0.1,0,0.0000,0,1e-7,0.000]
    YY03=[0.1,0,0.0000,0,1e-7,0.000]
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
   
    #####################################################################
    #Immune Inihibitor factor
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.1,0,0.0000,0,1e-7,0.000]
    YY02=[0.1,0,0.0000,0,1e-7,0.000]
    YY03=[0.1,0,0.0000,0,1e-7,0.000]
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
    


    
    #####################################################################
    #Immune stimulator factor
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY01=[0.1,0,0.0000,0,1e-7,0.000]
    YY02=[0.1,0,0.0000,0,1e-7,0.000]
    YY03=[0.1,0,0.0000,0,1e-7,0.000]
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
    fig.savefig('Immunestimulator42.png')
    
    
   

    ######################################################################
    ## Glioma cells,  Effector and Memory CAR  T cells dynamics for small dosage
    #####################################################################
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY0=[0.1,0,0.0000,0,1e-7,0.000]
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY01,start_time,end_time,intervals) 
    


    ax.plot(T, F0, 'g-', linewidth=2,label='Glioma cells')
    ax.plot(T, F1, 'r-', linewidth=2,label='Effector CAR T cells')
    ax.plot(T, F2, 'b-', linewidth=2,label='Memory CAR T Cells')
    
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:.2f}"))
    start_time=T[-1]
    end_time=T[-1]+obs

    YY0=[F0[-1],0.10,F2[-1],F3[-1],F4[-1],F5[-1]]
    
    
    F0,F1,F2,F3,F4,F5,T=eq2(newrates,YY0,start_time,end_time,intervals)
    
    

    ax.plot(T, F0, 'g-')
    ax.plot(T, F1, 'r-')
    ax.plot(T, F2, 'b-')

    # ax.axvline(x=7,linestyle="dashed",label='t=7',color='y')
    # ax.axvline(x=10,linestyle="dashed",label='t=10')
    # ax.axvline(x=17,linestyle="dashed",label='t=17')
    ax.set_xlabel('Days')
    ax.set_ylabel('$10^6$ Cells')
    
    ax.set_title('Dynamics of Glima, Effector CAR T, and Memory CAR T cells with small dose treatment')
    ax.legend()
    plt.show()
   
    ######################################################################
    ## Glioma cells,  Effector and Memory CAR  T cells dynamics for medium dosage
    #####################################################################
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
    
    ######################################################################
    ## Glioma cells,  Effector and Memory CAR  T cells dynamics for maximum dosage treatment
    #####################################################################
    fig, ax=plt.subplots()
    start_time=0
    end_time=7
    YY0=[0.1,0,0.0000,0,1e-7,0.000]
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
    
    ######################################################################
    #####################################################################
    