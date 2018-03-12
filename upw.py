# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 13:43:41 2018

@author: henri
"""

import numpy as np

def upw(u0, cfl, dx, T, flux, df, boundary):
    u = np.copy(u0)
    t = 0.0
    dt = cfl*dx/max(abs(df(u0)))
    i = np.arange(1,len(u0)-1,1)
    while t<T:
        if t+dt > T:
            dt = T-t
        t += dt
        u = boundary(u)
        f = flux(u)
        u[i] = u[i] - dt/dx*(f[i]-f[i-1])
        dt = cfl*dx/max(abs(df(u)))
    return u