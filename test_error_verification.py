# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 10:17:07 2018

@author: henri
"""

import numpy as np
from outflow import outflow
import matplotlib.pyplot as plt
from upw import upw

# Flux function
def f(u):
    return np.divide(np.power(u,2),np.power(u,2) + np.power(1-u,2))
    
s = np.linspace(0,1,501)
dfv = max(np.diff(f(s))/np.diff(s))
df = lambda u: np.zeros(len(u)) + dfv

dx = 1/1000
xr = np.arange(-0.5*dx,1+1.5*dx,dx)

n = 10
dX = 1/n
x  = np.arange(-0.5*dX,1+1.5*dX,dX)
x_e = [False]*1001
for i in range(len(x[1:-1])):
    x_e[int(dX*1000*(i+0.5))] = True


u0 = np.zeros(len(xr))
u0[xr<0.1]=1.0
ur = upw(u0, .995, dx, .65, f, df, outflow)
u0 = np.zeros(len(x))
u0[x<0.1]=1.0
uu = upw(u0, .995, dX, .65, f, df, outflow)
ur_e = 0.5*ur[1:] + 0.5*ur[:-1]
ur_comp = np.zeros(len(x[1:-1]))
k = 0
for i in range(len(x_e)):
    if x_e[i]:
        ur_comp[k] = ur_e[i]
        k += 1
uu = uu[1:-1]