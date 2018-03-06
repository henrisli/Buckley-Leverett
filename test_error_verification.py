# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 18:31:12 2018

@author: henri
"""
import numpy as np
import matplotlib.pyplot as plt
from upw import upw
from lxf import lxf
from lxw import lxw
from Nessyahu_Tadmor import nt
from cuw import cuw
from outflow import outflow
from periodic import periodic

# Flux function
@np.vectorize
def f(u):
    return u**2/(u**2 + (1-u)**2)
    
s = np.linspace(0,1,501)
dfv = max(np.diff(f(s))/np.diff(s))
df = lambda u: 0*u + dfv

dx = 1/1000
xr = np.arange(-0.5*dx,1+1.5*dx,dx)
u0 = 0*xr
u0[xr<0.1]=1.0
ur = upw(u0, .995, dx, .65, f, df, outflow)
    
# Solutions on coarser grids
N  = np.arange(10,110,10)
error_upw = np.zeros(len(N))
error_lxf = np.zeros(len(N))
error_lxw = np.zeros(len(N))
i = 0
n = 10
dx = 1/n
x  = np.arange(-0.5*dx,1+1.5*dx,dx)
u0 = 0*x
u0[x<0.1]=1.0
    
uu = upw(u0, .995, dx, .65, f, df, outflow)
uf = lxf(u0, .995, dx, .65, f, df, outflow)
uw = lxw(u0, .995, dx, .65, f, df, outflow)
a = np.zeros((len(xr)-2,3))
for i in range(len(x)-2):
    a[(xr<x[i+1]) & (xr>x[i]),] = [uu[i+1],uf[i+1],uw[i+1]]