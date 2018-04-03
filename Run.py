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
# T = 1 gives a continuous solution, T = 0.5 gives a discontinuous solution.
def BL_solution(method, T, M = 1):
    def analytical(u):
        return 1/2*(np.sqrt(np.divide(-2/(T*M)*u + np.sqrt(4/(T*M)*u + 1) - 1, 1/(T*M)*u) + 1) + 1)*np.logical_not(u>((1/2 + 1/np.sqrt(2))*T))
    
    # Flux function
    def f(u):
        return np.divide(np.power(u,2),np.power(u,2) + M*np.power(1-u,2))
    
    def limiter_function(r):
        return np.maximum(0, np.minimum(2*r, np.minimum((r+1)/2, 2)))
    
    s = np.linspace(0,1,1001)
    dfv = max(np.diff(f(s))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    # Reference solution
    dx = 1/2**12
    xr = np.arange(-0.5*dx,1+1.5*dx,dx)
    # Discontinuous solution:
    if T == 0.5:
        u0 = np.zeros(len(xr))
        u0[xr<=0]=1.0
    # Continuous solution:
    elif T == 1:
        u0 = np.ones(len(xr))*analytical(1)
        u0[xr<=0.0]=1.0
        
    #ur = upw(u0, 0.995, dx, T, f, df, outflow)
    
     # Solutions on coarser grids
    N  = 50
    dx = 1/N
    
    if method == 'classical':
       
        x  = np.arange(-0.5*dx,1+1.5*dx,dx)
        # Discontinuous solution:
        if T == 0.5:
            u0 = np.zeros(len(x))
            u0[x<=0] = 1.0
        # Continuous initial:
        elif T == 1:
            u0 = np.ones(len(x))*analytical(1)
            u0[x<=0] = 1.0
        
        uu = upw(u0, .995, dx, T, f, df, outflow)
        uf = lxf(u0, .995, dx, T, f, df, outflow)
        uw = lxw(u0, .995, dx, T, f, df, outflow)
        
        
        # Plot results
        plt.figure()
        plt.plot([1,2,3])
        plt.subplot(131)
        # Reference:
        #plt.plot(xr[1:-1], ur[1:-1], 'o')
        # Analytical:
        plt.plot(xr[1:-1], analytical(xr[1:-1]), color = 'red')
        plt.plot(x[1:-1], uu[1:-1], '.', markersize = 3)
        plt.title("Upwind")
        plt.subplot(132)
        # Reference:
        #plt.plot(xr[1:-1], ur[1:-1], 'o')
        # Analytical:
        plt.plot(xr[1:-1], analytical(xr[1:-1]), color = 'red')
        plt.plot(x[1:-1],uf[1:-1],'.', markersize = 3)
        plt.title("Lax-Friedrichs")
        plt.subplot(133)
        # Reference:
        #plt.plot(xr[1:-1], ur[1:-1], 'o')
        # Analytical:
        plt.plot(xr[1:-1], analytical(xr[1:-1]), color = 'red')
        plt.plot(x[1:-1],uw[1:-1],'.', markersize = 3)
        plt.title("Lax-Wendroff")
        if T == 1:
            plt.savefig("solution_classical_cont.pdf")
        elif T == 0.5:
            plt.savefig("solution_classical_discont.pdf")
    
    elif method == 'high':
        
        xh = np.arange(-1.5*dx, 1 + 2.5*dx, dx)
        
        # Discontinuous solution:
        if T == 0.5:
            u0 = np.zeros(len(xh))
            u0[xh<=0] = 1.0
        # Continuous initial:
        elif T == 1:
            u0 = np.ones(len(xh))*analytical(1)
            u0[xh<=0] = 1.0
            
        uc, phi = cuw(u0, 0.495, dx, T, f, df, outflow)
        un = nt(u0, 0.495, dx, T, f, df, outflow)
        
        #Plot results
        plt.figure()
        plt.plot([1,2])
        plt.subplot(121)
        # Reference:
        #plt.plot(xr[1:-1], ur[1:-1])
        # Analytical:
        plt.plot(xr[1:-1], analytical(xr[1:-1]), color = 'red')
        plt.plot(xh[2:-2],uc[2:-2], '.', markersize = 3)
        plt.title("Central upwind")
        plt.subplot(122)
        # Reference:
        #plt.plot(xr[1:-1], ur[1:-1])
        # Analytical:
        plt.plot(xr[1:-1], analytical(xr[1:-1]), color = 'red')
        plt.plot(xh[2:-2],un[2:-2], '.', markersize = 3)
        plt.title("Nessyahu-Tadmor")
        if T == 0.5:
            plt.savefig("solution_high_discont.pdf")
        elif T == 1:
            plt.savefig("solution_high_cont.pdf")


def Error_verification(method, T, norm):
    def analytical(u):
        return 1/2*(np.sqrt(np.divide(-2/T*u + np.sqrt(4/T*u + 1) - 1, 1/T*u) + 1) + 1)*np.logical_not(u>((1/2 + 1/np.sqrt(2))*T))
  
    # BL flux function:
    def f(u):
        return np.divide(np.power(u,2),np.power(u,2) + np.power(1-u,2))
    
    # Derivative of flux for BL:
    s = np.linspace(0,1,501)
    dfv = max(np.diff(f(s))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    # Reference solution
    dx = 1/2**12
    xr = np.arange(-0.5*dx,1+1.5*dx,dx)
            
    uref = analytical(xr)
    
    # Solutions on coarser grids
    N = np.array([2**i for i in range(1,9)])

    
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
            
            # Discontinuous solution:
            if T == 0.5:
                u0 = np.zeros(len(x))
                u0[x<=0] = 1.0
            # Continuous initial:
            elif T == 1:
                u0 = np.ones(len(x))*analytical(1)
                u0[x<=0] = 1.0
                
            uu = upw(u0, .995, dX, T, f, df, outflow)
            uu = uu[1:-1]
            uf = lxf(u0, .995, dX, T, f, df, outflow)
            uf = uf[1:-1]
            uw = lxw(u0, .995, dX, T, f, df, outflow)
            uw = uw[1:-1]
            uu_cc = [i for i in uu for j in range(int(dX/dx))]
            uf_cc = [i for i in uf for j in range(int(dX/dx))]
            uw_cc = [i for i in uw for j in range(int(dX/dx))]
            error_upw[j] = np.power(dX,1/norm)*np.linalg.norm(uu_cc - uref[1:-1],norm)
            error_lxf[j] = np.power(dX,1/norm)*np.linalg.norm(uf_cc - uref[1:-1],norm)
            error_lxw[j] = np.power(dX,1/norm)*np.linalg.norm(uw_cc - uref[1:-1],norm)
            j += 1
            
        elif method == "high":
            
            xh = np.arange(-1.5*dX, 1 + 2.5*dX, dX)
            
            # Discontinuous solution:
            if T == 0.5:
                u0 = np.zeros(len(xh))
                u0[xh<=0] = 1.0
            # Continuous solution:
            elif T == 1:
                u0 = np.ones(len(xh))*analytical(1)
                u0[xh<=0] = 1.0
                
            un = nt(u0, .495, dX, T, f, df, outflow)
            un = un[2:-2]
            uc, phi = cuw(u0, .495, dX, T, f, df, outflow)
            uc = uc[2:-2]
            phi = phi[2:-2]
            un_cc = [i for i in un for j in range(int(dX/dx))]
            uc_cc = [uc[i]+0.5*(((2*j+1)-int(dX/dx)))/int(dX/dx)*phi[i] for i in range(len(uc)) for j in range(int(dX/dx))]
            error_nt[j] = np.power(dX,1/norm)*np.linalg.norm(un_cc - uref[1:-1],norm)
            error_cuw[j] = np.power(dX,1/norm)*np.linalg.norm(uc_cc - uref[1:-1],norm)
            j += 1
    
    if method == 'classical':
        plt.figure()
        plt.loglog([1/i for i in N],error_upw,'o-')
        plt.loglog([1/i for i in N],error_lxf,'o-')
        plt.loglog([1/i for i in N],error_lxw,'o-')
        plt.legend(["UW","LF","LW"])
        plt.ylabel("Error")
        plt.xlabel("dx")
        if T == 0.5:
            if norm == 1:
                plt.title("Error in 1-norm")
                plt.savefig("Error_classical_disc1.pdf")
            elif norm == 2:
                plt.title("Error in 2-norm")
                plt.savefig("Error_classical_disc2.pdf")
            elif norm == np.inf:
                plt.title("Error in inf-norm")
                plt.savefig("Error_classical_disc_inf.pdf")
        elif T == 1:
            if norm == 1:
                plt.title("Error in 1-norm")
                plt.savefig("Error_classical_cont1.pdf")
            elif norm == 2:
                plt.title("Error in 2-norm")
                plt.savefig("Error_classical_cont2.pdf")
            elif norm == np.inf:
                plt.title("Error in inf-norm")
                plt.savefig("Error_classical_cont_inf.pdf")
    
    elif method == 'high':
        plt.figure()
        plt.loglog([1/i for i in N],error_nt)
        plt.loglog([1/i for i in N],error_cuw)
        plt.legend(["NT","CUW"])
        plt.ylabel("Error")
        plt.xlabel("dx")
        if T == 0.5:
            if norm == 1:
                plt.title("Error in 1-norm")
                plt.savefig("Error_high_disc1.pdf")
            elif norm == 2:
                plt.title("Error in 2-norm")
                plt.savefig("Error_high_disc2.pdf")
            elif norm == np.inf:
                plt.title("Error in inf-norm")
                plt.savefig("Error_high_disc_inf.pdf")
        elif T == 1:
            if norm == 1:
                plt.title("Error in 1-norm")
                plt.savefig("Error_high_cont1.pdf")
            elif norm == 2:
                plt.title("Error in 2-norm")
                plt.savefig("Error_high_cont2.pdf")
            elif norm == np.inf:
                plt.title("Error in inf-norm")
                plt.savefig("Error_high_cont_inf.pdf")
            
     
            
Error_verification('classical', 0.5, np.inf)
BL_solution('classical', 0.5)