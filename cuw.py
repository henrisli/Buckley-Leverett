# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:54:42 2018

@author: henri
"""
import numpy as np

def cuw(u0, cfl, dx, T, flux, dflux, boundary, lim = None):
    if lim == None:
        def lim(r):
            return np.maximum(0, np.minimum(1.3*r, np.minimum(0.5+0.5*r,1.3)))
    
    def limiter(a,b):
        return lim(np.divide(b, a + 1e-6))*a
    
    dt = cfl*dx/max(abs(dflux(u0)))
    u = np.copy(u0)
    U = np.copy(u0)
    f = 0*u
    t = 0.0
    n = len(u0)
    i = np.arange(2,n-2)
    j = np.arange(n-1)
    phi = np.zeros(n)
    while t<T:
        if (t+dt > T):
            dt = T-t
        
        t += dt
        u = boundary(u, 2)
        du = u[1:] - u[:-1]
        phi[1:-1] = limiter(du[:-1], du[1:])
        # Only when we have periodic boundary conditions: 
        phi[0] = phi[-2]
        phi[-1] = phi[1]  
        ur = u + 0.5*phi
        fr = flux(ur)
        dfr = dflux(ur)
        ul = u - 0.5*phi
        fl = flux(ul)
        dfl = dflux(ul)
        ap = np.maximum(np.maximum(dfr,dfl),0)
        am = np.minimum(np.minimum(dfr,dfl),0)
        mdf = np.maximum(ap,-am)
        mdf = max(mdf)
        f[j] = np.divide(ap[j]*fr[j] - am[j+1]*fl[j+1] + ap[j]*am[j+1]*(ul[j+1] - ur[j]), (ap[j]-am[j+1]+1e-6))
        U[i] = u[i] - dt/dx*(f[i]-f[i-1])
          
        U = boundary(U,2)
        du = U[1:] - U[:-1]
        phi[1:-1] = limiter(du[:-1], du[1:])
        # Only when we have periodic boundary conditions: 
        phi[0] = phi[-2]
        phi[-1] = phi[1]
        
        ur = U + 0.5*phi
        fr = flux(ur)
        dfr = dflux(ur)
        ul = U - 0.5*phi
        fl = flux(ul)
        dfl = dflux(ul)
        ap = np.maximum(np.maximum(dfr, dfl), 0)
        am = np.minimum(np.minimum(dfr, dfl), 0)
        mdf = np.maximum(np.maximum(ap, -am), mdf)
        mdf = max(mdf)
        #f[j] = (ap[j]*fr[j] - am[j+1]*fl[j+1] + ap[j]*am[j+1]*(ul[j+1] - ur[j]))/ (ap[j]-am[j+1]+1e-6)
        f[j] = (ap[j]*fr[j] - am[j+1]*fl[j+1] + ap[j]*am[j+1]*(ul[j+1] - ur[j]))/ (ap[j]-am[j+1]+1e-6);
    
        u[i] = 0.5*u[i] + 0.5*( U[i]-dt/dx*(f[i]-f[i-1]) )
        dt = cfl*dx/mdf
    return u