# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:39:17 2018

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
from inflow import inflow
from periodic import periodic

import cProfile # To run profiling: type cProfile.run('function_name')


# Advection: classic schemes 
# Initial data: set up initial data and add a fictitious node at each end
# of the interval. These cells will be used to impose boundary conditions.
def Advection_classic_schemes():
    dx = 1/100
    x  = np.arange(-0.5*dx, 1+1.5*dx, dx)
    u0 = np.sin((x-0.1)*np.pi/0.3)**2
    u0[(x<0.1) | (x>0.4)] = 0
    u0[(x<0.9) & (x>0.6)] = 1.0
    
    
    # Flux function
    @np.vectorize
    def f(u):
        return u
    
    df = lambda u: 0*u + 1
    
    # Run simulation
    uu = upw(u0, .995, dx, 20, f, df, periodic)
    uf = lxf(u0, .995, dx, 20, f, df, periodic)
    uw = lxw(u0, .995, dx, 20, f, df, periodic)
    
    # Plot results
    plt.figure()
    plt.plot([1,2,3])
    plt.subplot(131)
    plt.plot(x[1:-1], u0[1:-1])
    plt.plot(x[1:-1], uu[1:-1], '.', markersize = 3)
    plt.title("Upwind")
    plt.xticks(fontsize = 7)
    plt.yticks(fontsize = 7)
    plt.subplot(132)
    plt.plot(x[1:-1],u0[1:-1])
    plt.plot(x[1:-1],uf[1:-1],'.', markersize = 3)
    plt.title("Lax-Friedrichs")
    plt.xticks(fontsize = 7)
    plt.yticks(fontsize = 7)
    plt.subplot(133)
    plt.plot(x[1:-1],u0[1:-1])
    plt.plot(x[1:-1],uw[1:-1],'.', markersize = 3)
    plt.title("Lax-Wendroff")
    plt.xticks(fontsize = 7)
    plt.yticks(fontsize = 7)
    plt.savefig("Advection_classic_schemes.pdf")
    
    
# Advection: high-resolution schemes    
def Advection_high_resolution_schemes():
    def limiter_function(r):
        return np.maximum(0, np.minimum(2*r, np.minimum((r+1)/2, 2)))
    
    dx = 1/100
    xh = np.arange(-1.5*dx, 1 + 2.5*dx, dx)
    u0 = np.sin((xh-0.1)*np.pi/0.3)**2
    u0[(xh<0.1) | (xh>0.4)] = 0
    #u0 = np.zeros(len(xh))
    u0[(xh<0.9) & (xh>0.6)] = 1.0
    
    # Flux function
    def f(u):
        return u
    
    df = lambda u: np.ones(len(u))
    
    un = nt(u0, 0.495, dx, 200, f, df, periodic, limiter_function)
    uc = cuw(u0, 0.495, dx, 15, f, df, periodic)
    
    plt.figure()
    plt.plot(xh[2:-2], u0[2:-2])
    plt.plot(xh[2:-2], un[2:-2], 'o', mfc = 'none', markersize = 8)
    plt.plot(xh[2:-2], uc[2:-2], 'x', markersize = 3)
    plt.legend(["Initial", "Nessyahu-Tadmor", "Central upwind"], loc = 1, fontsize = 5)
    plt.savefig("Advection_high_resolution_schemes.pdf")

# Solution of Buckley-Leverett equation, both with classical
# and high-resolution schemes.
def BL_solution(method):
    # Flux function
    def f(u):
        return np.divide(np.power(u,2),np.power(u,2) + np.power(1-u,2))
    
    def limiter_function(r):
        return np.maximum(0, np.minimum(2*r, np.minimum((r+1)/2, 2)))
    
    s = np.linspace(0,1,501)
    dfv = max(np.diff(f(s))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    # Reference solution
    dx = 1/1000
    xr = np.arange(-0.5*dx,1+1.5*dx,dx)
    u0 = 0*xr
    u0[xr<0.1]=1.0
    ur = upw(u0, .995, dx, .65, f, df, outflow)
    
    if method == 'classical':
        # Solutions on coarser grids
        N  = 50
        dx = 1/N
        x  = np.arange(-0.5*dx,1+1.5*dx,dx)
        u0 = 0*x
        u0[x<0.1]=1.0
        
        uu = upw(u0, .995, dx, 0.65, f, df, outflow)
        uf = lxf(u0, .995, dx, 0.65, f, df, outflow)
        uw = lxw(u0, .995, dx, 0.65, f, df, outflow)
    
        # Plot results
        plt.figure()
        plt.plot([1,2,3])
        plt.subplot(131)
        plt.plot(xr[1:-1], ur[1:-1])
        plt.plot(x[1:-1], uu[1:-1], '.', markersize = 3)
        plt.title("Upwind")
        plt.subplot(132)
        plt.plot(xr[1:-1],ur[1:-1])
        plt.plot(x[1:-1],uf[1:-1],'.', markersize = 3)
        plt.title("Lax-Friedrichs")
        plt.subplot(133)
        plt.plot(xr[1:-1],ur[1:-1])
        plt.plot(x[1:-1],uw[1:-1],'.', markersize = 3)
        plt.title("Lax-Wendroff")
        plt.savefig("solution_classical.pdf")
    
    elif method == 'high':
        N  = 50
        dx = 1/N
        xh = np.arange(-1.5*dx, 1 + 2.5*dx, dx)
        u0 = 0*xh
        u0[xh<0.1] = 1
        
        
        uc = cuw(u0, 0.495, dx, 0.65, f, df, inflow)
        un = nt(u0, 0.495, dx, 0.65, f, df, inflow)
        
        #Plot results
        plt.figure()
        plt.plot([1,2])
        plt.subplot(121)
        plt.plot(xr[1:-1],ur[1:-1])
        plt.plot(xh[2:-2],uc[2:-2], '.', markersize = 3)
        plt.title("Central upwind")
        plt.subplot(122)
        plt.plot(xr[1:-1],ur[1:-1])
        plt.plot(xh[2:-2],un[2:-2], '.', markersize = 3)
        plt.title("Nessyahu-Tadmor")
        plt.savefig("solution_high.pdf")


def Error_verification():
    # Flux function
    def f(u):
        return np.divide(np.power(u,2),np.power(u,2) + np.power(1-u,2))
    
    s = np.linspace(0,1,501)
    dfv = max(np.diff(f(s))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    # Reference solution
    dx = 1/1000
    xr = np.arange(-0.5*dx,1+1.5*dx,dx)
    u0 = 0*xr
    u0[xr<0.1]=1.0
    ur = upw(u0, .995, dx, .65, f, df, outflow)
    
    # Solutions on coarser grids
    N  = np.arange(10,20,10)
    error_upw = np.zeros(len(N))
    error_lxf = np.zeros(len(N))
    error_lxw = np.zeros(len(N))
    i = 0
    for n in N:
        dX = 1/n
        x  = np.arange(-0.5*dX,1+1.5*dX,dX)
        print(x)
        u0 = 0*x
        u0[x<0.1]=1.0
        #print(N[i])
        uu = upw(u0, .995, dx, .65, f, df, outflow)
        uf = lxf(u0, .995, dx, .65, f, df, outflow)
        uw = lxw(u0, .995, dx, .65, f, df, outflow)
        a = np.zeros((len(xr)-2,3))
        print(len(x)-2)
        for j in range(len(x)-2):
            j1 = int(np.floor(j*dX/dx))
            j2 = int(np.floor((j+1)*dX/dx))
            for k in range(j1,j2-1):
                a[k,:] = [uu[j+1],uf[j+1],uw[j+1]]
        error_upw[i] = np.linalg.norm(a[:,0] - ur[1:-1], 2)
        error_lxf[i] = np.linalg.norm(a[:,1] - ur[1:-1], 2)
        error_lxw[i] = np.linalg.norm(a[:,2] - ur[1:-1], 2)
        i += 1
    
    plt.figure()
    plt.loglog(N,error_upw)
    plt.loglog(N,error_lxf)
    plt.loglog(N,error_lxw)
    plt.legend(["UW","LF","LW"])
    plt.ylabel("Error")
    plt.xlabel("N")
    plt.savefig("Error.pdf")
    
    
    
BL_solution('high')