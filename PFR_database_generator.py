from scipy.integrate import quad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, log, pi
import random
import time
import pyarrow.parquet as pq
import pyarrow as pa

#### REACTIONS
## T + H -> B + M  
## B -> 0.5D + 0.5H

# T: Toluene
# H: Hydrogen (H2)
# B: Benzene
# D: Biphenyl
# M: Methane

start_time = time.time()

def constants(T, R):
    
    # Reaction 1
    k1 = 6.3*10**10*exp(-26170/T) # (l/mol)^0.5 s^-1
    k1 = k1/(1000**-(0.5)) # (m3/mol)^0.5 s^-1

    # Reaction 2
    Ak2 = 3.34*10**7 # mol/m3*s*atm2
    Ak2 = Ak2/(101325**2) # mol/m3*s*Pa2
    E = 30190*4.184 # J/mol
    k2 = Ak2*exp(-E/(R*T)) # mol/m3*s*Pa2
    k2 = k2*(R*T)**2 #m3/mol*s

    # Reaction 2 equilibrum 
    Keq = exp(-19.76 - 1692/T + 3.31*log(T) - 0.00163*T + (1.96*10**-7)*T**2)
    
    return k1, k2, Keq
    
def reactions_rate(k1, k2, ct0, fT, fH, fB, fD, Keq):
    ftotal = fT + fH + fB + fM + fD
    Ct = ct0*fT/ftotal
    Ch = ct0*fH/ftotal
    Cb = ct0*fB/ftotal
    Cd = ct0*fD/ftotal
    
    r1 = k1*Ct*(Ch**0.5)
    r2 = k2*(Cb**2-Cd*Ch/Keq)
    
    return r1, r2

def dElementdV(r1,r2):
    dTdV = -r1
    dHdV = -r1 + 0.5*r2
    dBdV = r1 - r2
    dDdV = 0.5*r2
    dMdV = r1
    
    return dTdV, dHdV, dBdV, dDdV, dMdV


def cp(T, a, b, c, d, e):
    return 8.314*(a + b*(10**-3)*T + c*(10**-5)*(T**2) + d*(10**-8)*(T**3) + e*(10**-11)*(T**4))


def dTempdV(r1, r2, T, fT, fH, fB, fM, fD):
    cp_T = cp(T, 3.866, 3.558, 13.356, -18.659, 7.69) # Reference Cp Parameters from Prausnitz
    cp_H = cp(T, 2.883, 3.681, -0.772, 0.692, -0.213)
    cp_B = cp(T, 3.551, -6.184, 14.365, -19.807, 8.234)
    cp_D = cp(T, -0.843, 61.392, 6.352, -13.754, 6.169)
    cp_M = cp(T, 4.568, -8.975, 3.631, -3.407, 1.091)
    
    Tref = 298.15
    
    cpdT_T, _ = quad(cp, Tref, T, args=(3.866, 3.558, 13.356, -18.659, 7.69))
    cpdT_H, _ = quad(cp, Tref, T, args=(2.883, 3.681, -0.772, 0.692, -0.213))
    cpdT_B, _ = quad(cp, Tref, T, args=(3.551, -6.184, 14.365, -19.807, 8.234))
    cpdT_D, _ = quad(cp, Tref, T, args=(-0.843, 61.392, 6.352, -13.754, 6.169))
    cpdT_M, _ = quad(cp, Tref, T, args=(4.568, -8.975, 3.631, -3.407, 1.091))
    
    delta_H1ref = (82.8 + (-74.5) - 50 - 0)*1000 # J/mol, Reference: NIST
    delta_H2ref = (0.5*182.2 + 0 - 50)*1000 # J/mol
    
    delta_H1 = delta_H1ref + (cpdT_B + cpdT_M - cpdT_T - cpdT_H)
    delta_H2 = delta_H2ref + (0.5*cpdT_H + 0.5*cpdT_D - cpdT_B)
    
    return (r1*-delta_H1 + r2*-delta_H2)/(fT*cp_T + fH*cp_H + fB*cp_B + fM*cp_M + fD*cp_D)


# Reactor Project parameters
V_reactor = 30 # m3
X_lim = 0.75
h = 0.1

# number of data points
n = int(1e6)

# inputs and outputs
ftotal0_list, yt0_list, yh0_list, yb0_list, yd0_list, ym0_list = [],[],[],[],[],[]
T0_list, P_list = [],[]
T_list, X_list = [],[]
fT_list, fH_list, fB_list, fD_list, fM_list = [],[],[],[],[]

for j in range(n):

    # random value for each variable
    ftotal0 = round(random.normalvariate(100, 14), 3) # mol/s
    yt0 = max(0.05,round(random.normalvariate(0.1, 0.1),3))
    yh0_ratio = max(1,round(random.normalvariate(6, 3),3))
    yb0 = max(0,round(random.normalvariate(0.03, 0.05),3))
    yd0 = max(0,round(random.normalvariate(0.03, 0.05),3))
    ym0 = max(0,round(random.normalvariate(0.05, 0.1),3))
    T0 = round(random.normalvariate(500, 20),2) #°C
    Pressure = round(random.normalvariate(35, 3),2) #bar
    
    # Convert measures and normalize y
    T0 = 273.15 + T0 # K
    yh0 = yt0*yh0_ratio
    total_sum = yt0 + yh0 + yb0 + yd0 + ym0
    yt0, yh0, yb0, yd0, ym0 = yt0/total_sum, yh0/total_sum, yb0/total_sum, yd0/total_sum, ym0/total_sum
    
    # molar flow
    fT0 = ftotal0*yt0 
    fH0 = ftotal0*yh0
    fB0 = ftotal0*yb0
    fD0 = ftotal0*yd0
    fM0 = ftotal0*ym0
    ftotal0 = fT0 + fH0 + fB0 + fD0 + fM0
    
    # concentration
    R = 8.314 #m3.Pa/mol.K
    ct0 = yt0*Pressure*10**5/(R*T0) # mol/m3
    
    # ODE initial state
    fT, fH, fB, fD, fM = fT0, fH0, fB0, fD0, fM0
    T = T0
    V = 0 
    X = 0
    
    start_time_while = time.time()
    
    while V < V_reactor and X < X_lim:
        k1, k2, Keq = constants(T, R)
        r1, r2 = reactions_rate(k1, k2, ct0, fT, fH, fB, fD, Keq)
        dTdV, dHdV, dBdV, dDdV, dMdV = dElementdV(r1,r2)
        
        fT += h*dTdV
        fH += h*dHdV
        fB += h*dBdV
        fD += h*dDdV
        fM += h*dMdV
        T  += h*dTempdV(r1, r2, T, fT, fH, fB, fM, fD)
        V  += h
        
        X = (fT0 - fT)/fT0
        deltaT = T - T0
        
        # makes sure calculation doesn't take too long
        delta_time = time.time() - start_time_while
        if delta_time > 1:
            X = 0
            break
        
    if X > 0.75:
        X = 0.75
        
    if deltaT <= 120 and V > 8 and X > 0.4:
        ftotal0_list.append(ftotal0)
        yt0_list.append(yt0)
        yh0_list.append(yh0)
        yb0_list.append(yb0)
        yd0_list.append(yd0)
        ym0_list.append(ym0)
        T0_list.append(T0)
        P_list.append(Pressure)
        T_list.append(T)
        X_list.append(X)
        fT_list.append(fT)
        fH_list.append(fH)
        fB_list.append(fB)
        fD_list.append(fD)
        fM_list.append(fM)
        
    if j%100000 == 0:
        delta_time = time.time() - start_time
        minutes = int(delta_time // 60)
        seconds = int(delta_time % 60)
        print(f"{j} rows created. {minutes}min and {seconds}s running")
        
    
    

df = pd.DataFrame({
                    'TOTAL_MOLAR_FLOW': ftotal0_list,
                    'T_MOLAR_FRACTION': yt0_list,
                    'H_MOLAR_FRACTION': yh0_list,
                    'B_MOLAR_FRACTION': yb0_list,
                    'D_MOLAR_FRACTION': yd0_list,
                    'M_MOLAR_FRACTION': ym0_list,
                    'TEMPERATURE_0': T0_list,
                    'PRESSURE': P_list,
                    'TEMPERATURE': T_list,
                    'T_MOLAR_FLOW': fT_list,
                    'H_MOLAR_FLOW': fH_list,
                    'B_MOLAR_FLOW': fB_list,
                    'D_MOLAR_FLOW': fD_list,
                    'M_MOLAR_FLOW': fM_list,
                    'YIELD': X_list
                   })

# data from previous runs and append new data
data_load = pd.read_parquet('PFR_reactor.parquet')
df = pd.concat([data_load, df], ignore_index=True)

df.to_parquet('PFR_reactor.parquet', engine='pyarrow')




# Ajuste a carga térmica a fim de que a temperatura na saída seja no máximo 83 K superior a de alimentação
# talvez eu possa usar da temperatura para simular o controle



    




    
    