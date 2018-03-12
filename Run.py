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
    # Flux function
    def f(u):
        return u
    
    df = lambda u: np.zeros(len(u)) + 1
    
    N = 50
    dx = 1/N
    x  = np.arange(-0.5*dx, 1+1.5*dx, dx)
    u0 = np.sin((x-0.1)*np.pi/0.3)**2
    u0[(x<0.1) | (x>0.4)] = 0
    u0[(x<0.9) & (x>0.6)] = 1.0
    
    
    # Run simulation
    uu = upw(u0, 0.995, dx, 20, f, df, periodic)
    uf = lxf(u0, 0.995, dx, 20, f, df, periodic)
    uw = lxw(u0, 0.995, dx, 20, f, df, periodic)
    
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
    # Limiter function
    def limiter_function(r):
        return np.maximum(0, np.minimum(2*r, np.minimum((r+1)/2, 2)))
    
    #Set up initial values
    dx = 1/100
    xh = np.arange(-1.5*dx, 1 + 2.5*dx, dx)
    u0 = np.sin((xh-0.1)*np.pi/0.3)**2
    u0[(xh<0.1) | (xh>0.4)] = 0
    u0[(xh<0.9) & (xh>0.6)] = 1.0
    
    # Flux function
    def f(u):
        return u
    
    df = lambda u: np.ones(len(u))
    
    un = nt(u0, 0.495, dx, 1, f, df, periodic, limiter_function)
    uc = cuw(u0, 0.495, dx, 1, f, df, periodic)
    
    plt.figure()
    plt.plot(xh[2:-2], u0[2:-2])
    plt.plot(xh[2:-2], un[2:-2], 'o', mfc = 'none', markersize = 8)
    plt.plot(xh[2:-2], uc[2:-2], 'x', markersize = 3)
    plt.legend(["Initial", "Nessyahu-Tadmor", "Central upwind"], loc = 1, fontsize = 5)
    plt.savefig("Advection_high_resolution_schemes.pdf")

# Solution of Buckley-Leverett equation, both with classical
# and high-resolution schemes.
def BL_solution(method, initial, M = 1):
    # Flux function
    def f(u):
        return np.divide(np.power(u,2),np.power(u,2) + M*np.power(1-u,2))
    
    def limiter_function(r):
        return np.maximum(0, np.minimum(2*r, np.minimum((r+1)/2, 2)))
    
    s = np.linspace(0,1,1001)
    dfv = max(np.diff(f(s))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    # Reference solution
    dx = 1/1000
    xr = np.arange(-0.5*dx,1+1.5*dx,dx)
    # Discontinuous initial:
    if initial == 'dis':
        u0 = np.zeros(len(xr))
        u0[xr<0.1]=1.0
    # Continuous initial:
    elif initial == 'cont':
        u0 = np.sin((xr-0.1)*np.pi/0.3)**2
        u0[(xr<0.1) | (xr>0.4)] = 0
        
    ur = upw(u0, 0.995, dx, 0.65, f, df, outflow)
    
     # Solutions on coarser grids
    N  = 50
    dx = 1/N
    
    if method == 'classical':
       
        x  = np.arange(-0.5*dx,1+1.5*dx,dx)
        # Discontinuous initial:
        if initial == 'dis':
            u0 = np.zeros(len(x))
            u0[x<0.1]=1.0
        # Continuous initial:
        elif initial == 'cont':
            u0 = np.sin((x-0.1)*np.pi/0.3)**2
            u0[(x<0.1) | (x>0.4)] = 0
        
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
        if initial == 'cont':
            plt.savefig("solution_classical_cont.pdf")
        elif initial == 'dis':
            plt.savefig("solution_classical_discont.pdf")
    
    elif method == 'high':
        
        xh = np.arange(-1.5*dx, 1 + 2.5*dx, dx)
        
        # Discontinuous initial:
        if initial == 'dis':
            u0 = np.zeros(len(xh))
            u0[xh<0.1]=1.0
        # Continuous initial:
        if initial == 'cont':
            u0 = np.sin((xh-0.1)*np.pi/0.3)**2
            u0[(xh<0.1) | (xh>0.4)] = 0
            
            
        uc = cuw(u0, 0.495, dx, 0.5, f, df, inflow)
        un = nt(u0, 0.495, dx, 0.5, f, df, inflow)
        
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
        if initial == 'cont':
            plt.savefig("solution_high_cont.pdf")
        elif initial == 'dis':
            plt.savefig("solution_high_discont.pdf")


def Error_verification_space(method, initial):
    # BL flux function:
    def f(u):
        return np.divide(np.power(u,2),np.power(u,2) + np.power(1-u,2))
    
    # Advection flux:
    #def f(u):
    #    return u
    
    # Derivative of flux for BL:
    s = np.linspace(0,1,501)
    dfv = max(np.diff(f(s))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    # Derivative of flux for advection:
    #df = lambda u: np.zeros(len(u)) + 1
        
    # Reference solution
    dx = 1/1000
    xr = np.arange(-0.5*dx,1+1.5*dx,dx)
    print(xr)
    # Discontinuous initial:
    if initial == 'dis':
        u0 = np.zeros(len(xr))
        u0[xr<0.1]=1.0
        
    # Continuous initial:
    elif initial == 'cont':
        u0 = np.sin((xr-0.1)*np.pi/0.3)**2
        u0[(xr<0.1) | (xr>0.4)] = 0
    
    ur = upw(u0, .995, dx, .65, f, df, outflow)
    
    # Solutions on coarser grids
    N  = np.array([5,10,20,80,160])
    
    if method == 'classical':
        error_upw = np.zeros(len(N))
        error_lxf = np.zeros(len(N))
        error_lxw = np.zeros(len(N))
    elif method == 'high':
        error_nt = np.zeros(len(N))
        error_cuw = np.zeros(len(N))
    j = 0
    for n in N:
        dX = 1/n
        if method == 'classical':
            x  = np.arange(-0.5*dX,1+1.5*dX,dX)
            print(x)
            # Discontinuous initial:
            if initial == 'dis':
                u0 = np.zeros(len(x))
                u0[x<0.1]=1.0
            # Continuous initial:
            elif initial == 'cont':
                u0 = np.sin((x-0.1)*np.pi/0.3)**2
                u0[(x<0.1) | (x>0.4)] = 0
            uu = upw(u0, .995, dX, .65, f, df, outflow)
            uf = lxf(u0, .995, dX, .65, f, df, outflow)
            uw = lxw(u0, .995, dX, .65, f, df, outflow)
            x_e = [False]*1001
            for i in range(len(x[1:-1])):
                x_e[int(dX/dx*(i+0.5))] = True
            ur_e = 0.5*ur[1:] + 0.5*ur[:-1]
            ur_comp = np.zeros(len(x[1:-1]))
            k = 0
            for i in range(len(x_e)):
                if x_e[i]:
                    ur_comp[k] = ur_e[i]
                    k += 1
            uu = uu[1:-1]
            uf = uf[1:-1]
            uw = uw[1:-1]
            error_upw[j] = np.sqrt(dX)*np.linalg.norm(uu - ur_comp, 2)
            error_lxf[j] = np.sqrt(dX)*np.linalg.norm(uf - ur_comp, 2)
            error_lxw[j] = np.sqrt(dX)*np.linalg.norm(uw - ur_comp, 2)
            j += 1
        elif method == "high":
            xh = np.arange(-1.5*dX, 1 + 2.5*dX, dX)
            # Discontinuous initial:
            if initial == 'dis':
                u0 = np.zeros(len(xh))
                u0[xh<0.1]=1.0
                
            # Continuous initial:
            elif initial == 'cont':
                u0 = np.sin((xh-0.1)*np.pi/0.3)**2
                u0[(xh<0.1) | (xh>0.4)] = 0
                
            un = nt(u0, .495, dX, .65, f, df, outflow)
            uc = cuw(u0, .495, dX, .65, f, df, outflow)
            x_e = [False]*4001
            for i in range(len(xh[2:-2])):
                x_e[int(dX/dx*(i+0.5))] = True
            ur_e = 0.5*ur[1:] + 0.5*ur[:-1]
            ur_comp = np.zeros(len(xh[2:-2]))
            k = 0
            for i in range(len(x_e)):
                if x_e[i]:
                    ur_comp[k] = ur_e[i]
                    k += 1
            un = un[2:-2]
            uc = uc[2:-2]
            error_nt[j] = np.sqrt(dX)*np.linalg.norm(un - ur_comp, 2)
            error_cuw[j] = np.sqrt(dX)*np.linalg.norm(uc - ur_comp, 2)
            j += 1
    
    
    if method == 'classical':
        plt.figure()
        plt.loglog([1/i for i in N],error_upw)
        plt.loglog([1/i for i in N],error_lxf)
        plt.loglog([1/i for i in N],error_lxw)
        plt.legend(["UW","LF","LW"])
        plt.ylabel("Error")
        plt.xlabel("N")
        if initial=='dis':
            plt.savefig("Error_classical_disc.pdf")
        elif initial=='cont':
            plt.savefig("Error_classical_cont.pdf")
    elif method == 'high':
        plt.figure()
        plt.loglog(N,error_nt)
        plt.loglog(N,error_cuw)
        plt.legend(["NT","CUW"])
        plt.ylabel("Error")
        plt.xlabel("N")
        if initial=='dis':
            plt.savefig("Error_high_disc.pdf")
        elif initial=='cont':
            plt.savefig("Error_high_cont.pdf")

Error_verification_space('classical','dis')