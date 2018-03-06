# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:27:36 2018

@author: henri
"""
import numpy as np

def nt(u0, cfl, dx, T, flux, dflux, boundary, lim = None):
    if lim == None:
        @np.vectorize
        def lim(r):
            return max(0, min(1.3*r, min(0.5+0.5*r, 1.3)))
    @np.vectorize
    def limiter(a,b):
        return lim(np.divide(b,(a+1e-6)))*a
    
    dt = cfl*dx/max(abs(dflux(u0)))
    
    u=np.copy(u0)
    t = 0.0
    while t<T:
        print(t,T)
        if t+dt > T:
            dt = 0.5*(T-t)
        
        u = boundary(u,2)
        du = u[1:]-u[:-1]
        s = limiter(du[:-1], du[1:])
        s = np.insert(s, 0, s[-1])
        s = np.append(s, s[1])
        f = flux(u)
        df = f[1:]-f[:-1]
        sigma = limiter(df[:-1], df[1:])
        # Only when we have periodic boundary conditions: 
        sigma = np.insert(sigma,0,sigma[-1])
        sigma = np.append(sigma,sigma[1])    
        v = u - 0.5*dt/dx*sigma
        g = flux(v) + 0.125*dx/dt*s;
        u[1:] = 0.5*(u[:-1] + u[1:]) - dt/dx*(g[1:]-g[:-1])
        
        u = boundary(u,2)
        du = u[1:]-u[:-1]
        s = limiter(du[:-1], du[1:])
        s = np.insert(s, 0, s[-1])
        s = np.append(s, s[1])
        f = flux(u)
        df = f[1:]-f[:-1]
        sigma = limiter(df[:-1], df[1:])
        # Only when we have periodic boundary conditions: 
        sigma = np.insert(sigma, 0, sigma[-1])
        sigma = np.append(sigma, sigma[1])  
        v = u - 0.5*dt/dx*sigma
        g = flux(v) + 0.125*dx/dt*s
        u[:-1] = 0.5*(u[:-1]+u[1:]) - dt/dx*(g[1:]-g[:-1])
        
        t += 2*dt
        dt = cfl*dx/max(abs(dflux(u)))
    return u
  