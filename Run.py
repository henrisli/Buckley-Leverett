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
from god import god
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
    xr = np.arange(0.5*dx,1+0.5*dx,dx)
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
        plt.plot(xr, analytical(xr), color = 'red')
        plt.plot(x[1:-1], uu[1:-1], '.', markersize = 3)
        plt.title("Upwind")
        plt.subplot(132)
        # Reference:
        #plt.plot(xr[1:-1], ur[1:-1], 'o')
        # Analytical:
        plt.plot(xr, analytical(xr), color = 'red')
        plt.plot(x[1:-1],uf[1:-1],'.', markersize = 3)
        plt.title("Lax-Friedrichs")
        plt.subplot(133)
        
        # Reference:
        #plt.plot(xr[1:-1], ur[1:-1], 'o')
        # Analytical:
        plt.plot(xr, analytical(xr), color = 'red')
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
            
        ug, phi = god(u0, 0.495, dx, T, f, df, outflow)
        
        #Plot results
        plt.figure()
        # Analytical:
        plt.plot(xr, analytical(xr), color = 'red')
        plt.plot(xh[2:-2], ug[2:-2], '.', markersize = 3)
        plt.title("Godunov")
        if T == 0.5:
            plt.savefig("solution_high_discont.pdf")
        elif T == 1:
            plt.savefig("solution_high_cont.pdf")


def Error_verification(T, norm):
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
    xr = np.arange(0.5*dx,1+0.5*dx,dx)
            
    uref = analytical(xr)
  
    # Solutions on coarser grids
    N = np.array([2**i for i in range(3,11)])


    error_upw = np.zeros(len(N))
    error_lxf = np.zeros(len(N))
    error_lxw = np.zeros(len(N))
    error_god = np.zeros(len(N))
    j = 0
    for n in N:
        dX = 1/n
        x  = np.arange(-0.5*dX,1+1.5*dX,dX)
        xg = np.arange(-1.5*dX, 1 + 2.5*dX, dX)
        
        # Discontinuous solution:
        if T == 0.5:
            u0 = np.zeros(len(x))
            u0_g = np.zeros(len(xg))
            u0[x<=0] = 1.0
            u0_g[xg<=0] = 1.0
        # Continuous initial:
        elif T == 1:
            u0 = np.ones(len(x))*analytical(1)
            u0_g = np.ones(len(xg))*analytical(1)
            u0[x<=0] = 1.0
            u0_g[xg<=0] = 1.0
            
        uu = upw(u0, .995, dX, T, f, df, outflow)
        uu = uu[1:-1]
        uf = lxf(u0, .995, dX, T, f, df, outflow)
        uf = uf[1:-1]
        uw = lxw(u0, .995, dX, T, f, df, outflow)
        uw = uw[1:-1]
        ug, phi = god(u0_g, .495, dX, T, f, df, outflow)
        ug = ug[2:-2]
        phi = phi[2:-2]
        # Now create a constant reconstruction to compute error for the three
        # classical schemes.
        uu_c = [i for i in uu for j in range(int(dX/dx))]
        uf_c = [i for i in uf for j in range(int(dX/dx))]
        uw_c = [i for i in uw for j in range(int(dX/dx))]
        # To compute the numerical error in the central upwind scheme, we need
        # to create a piecewise continuous linear reconstruction of the solution.
        # To do this, we use the vector phi and then create the reconstruction.
        ug_c = [ug[i]+0.5*(((2*j+1)-int(dX/dx)))/int(dX/dx)*phi[i] for i in range(len(ug)) for j in range(int(dX/dx))]
        error_upw[j] = np.power(dX,1/norm)*np.linalg.norm(uu_c - uref,norm)
        error_lxf[j] = np.power(dX,1/norm)*np.linalg.norm(uf_c - uref,norm)
        error_lxw[j] = np.power(dX,1/norm)*np.linalg.norm(uw_c - uref,norm)
        error_god[j] = np.power(dX,1/norm)*np.linalg.norm(ug_c - uref,norm)
        j += 1
            
    plt.figure()
    plt.axis('equal')
    plt.loglog(np.divide(1,N),error_upw,'o-')
    plt.loglog(np.divide(1,N),error_lxf,'o-')
    plt.loglog(np.divide(1,N),error_lxw,'o-')
    plt.loglog(np.divide(1,N), error_god, 'o-')
    plt.legend(["UW","LF","LW","God"])
    plt.ylabel("Error")
    plt.xlabel("h")
    if T == 0.5:
        if norm == 1:
            plt.title("Error in 1-norm")
            plt.savefig("Error_disc1.pdf")
        elif norm == 2:
            plt.title("Error in 2-norm")
            plt.savefig("Error_disc2.pdf")
        elif norm == np.inf:
            plt.title("Error in inf-norm")
            plt.savefig("Error_disc_inf.pdf")
    elif T == 1:
        if norm == 1:
            plt.title("Error in 1-norm")
            plt.savefig("Error_cont1.pdf")
        elif norm == 2:
            plt.title("Error in 2-norm")
            plt.savefig("Error_cont2.pdf")
        elif norm == np.inf:
            plt.title("Error in inf-norm")
            plt.savefig("Error_cont_inf.pdf")
     
            
Error_verification(0.5, np.inf)
BL_solution('high', 0.5)