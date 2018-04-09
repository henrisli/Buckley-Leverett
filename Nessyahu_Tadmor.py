# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 15:27:36 2018

@author: henri
"""
import numpy as np

def nt(u0, cfl, dx, T, flux, dflux, boundary, lim = None):
    if lim == None:
        def lim(r):
            return np.maximum(0, np.minimum(1.3*r, np.minimum(0.5+0.5*r, 1.3)))

    def limiter(a,b):
        return lim(np.divide(b,(a+1e-6)))*a
    
    dt = cfl*dx/max(abs(dflux(u0)))
    
    u=np.copy(u0)
    t = 0.0
    s = np.zeros(len(u))
    sigma = np.zeros(len(u))
    while t<T:
        if t+dt > T:
            dt = 0.5*(T-t)
        
        u = boundary(u,2)
        du = np.diff(u)
        s[1:-1] = limiter(du[:-1], du[1:])
        s[0] = s[-2]
        s[-1] = s[1]
        f = flux(u)
        df = np.diff(f)
        sigma[1:-1] = limiter(df[:-1], df[1:])
        sigma[0] = sigma[-2]
        sigma[-1] = sigma[1]    
        v = u - 0.5*dt/dx*sigma
        g = flux(v) + 0.125*dx/dt*s
        u[1:] = 0.5*(u[:-1] + u[1:]) - dt/dx*(g[1:]-g[:-1])
        
        u = boundary(u,2)
        du = np.diff(u)
        s[1:-1] = limiter(du[:-1], du[1:])
        s[0] = s[-2]
        s[-1] = s[1]
        f = flux(u)
        df = np.diff(f)
        sigma[1:-1] = limiter(df[:-1], df[1:]) 
        sigma[0] = sigma[-2]
        sigma[-1] = sigma[1] 
        v = u - 0.5*dt/dx*sigma
        g = flux(v) + 0.125*dx/dt*s
        u[:-1] = 0.5*(u[:-1]+u[1:]) - dt/dx*(g[1:]-g[:-1])
        
        t += 2*dt
        dt = cfl*dx/max(abs(dflux(u)))
    u = boundary(u, 2)
    du = np.diff(u)
    s[1:-1] = limiter(du[:-1], du[1:])
    s[0] = s[-2]
    s[-1] = s[1]
    return u, s
  