
import numpy as np
import matplotlib.pyplot as plt

# Import schemes:
from upw import upw
from lxf import lxf
from lxw import lxw
from god import god

# Import flux function and analytical solution:
from flux import flux
from analytical import analytical

# The following imports a function for the boundary conditions
from inflow import inflow


# Solution of Buckley-Leverett equation, both with classical
# and Godunov schemes.
# T = 1 gives a continuous solution, T = 0.5 gives a discontinuous solution.
def BL_solution(method, T):
        
    #Here we compute the maximum value of f'(u).
    s = np.linspace(0,1,1001)
    dfv = max(np.diff(flux(s))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    # Grid for the analytical solution
    dx = 1/2**12
    xr = np.arange(0.5*dx,1+0.5*dx,dx)
        
    # Solutions on coarser grids
    N  = 50
    dx = 1/N
    
    if method == 'classical':
        # Coarser grid
        x  = np.arange(-0.5*dx,1+1.5*dx,dx)
        
        # Discontinuous solution:
        if T == 0.5:
            u0 = np.zeros(len(x))
            u0[x<=0] = 1.0
            
        # Continuous initial:
        elif T == 1:
            u0 = np.ones(len(x))*analytical(1,T)
            u0[x<=0] = 1.0
        
        # Compute solutions with the three classical schemes
        uu = upw(u0, 0.995, dx, T, flux, df, inflow)
        uf = lxf(u0, 0.995, dx, T, flux, df, inflow)
        uw = lxw(u0, 0.995, dx, T, flux, df, inflow)
        
        
        # Plot results
        plt.figure()
        plt.plot([1,2,3])
        plt.subplot(131)
        # Analytical solution:
        plt.plot(xr, analytical(xr, T), color = 'red')
        plt.plot(x[1:-1], uu[1:-1], '.', markersize = 3) # We dont want to plot fictitious nodes, thereby the command [1:-1].
        plt.title("Upwind")
        plt.subplot(132)
        # Analytical solution:
        plt.plot(xr, analytical(xr,T), color = 'red')
        plt.plot(x[1:-1], uf[1:-1],'.', markersize = 3)
        plt.title("Lax-Friedrichs")
        plt.subplot(133)
        # Analytical solution:
        plt.plot(xr, analytical(xr,T), color = 'red')
        plt.plot(x[1:-1], uw[1:-1],'.', markersize = 3)
        plt.title("Lax-Wendroff")
        
        if T == 1:
            plt.savefig("solution_classical_cont.pdf")
        elif T == 0.5:
            plt.savefig("solution_classical_discont.pdf")
   
    
    elif method == 'high':
        # Coarser grid, need two fictitious nodes at each end for this scheme.
        xh = np.arange(-1.5*dx, 1 + 2.5*dx, dx)
        
        # Discontinuous solution:
        if T == 0.5:
            u0 = np.zeros(len(xh))
            u0[xh<=0] = 1.0
            
        # Continuous initial:
        elif T == 1:
            u0 = np.ones(len(xh))*analytical(1,T)
            u0[xh<=0] = 1.0
        
        ug, phi = god(u0, 0.495, dx, T, flux, df, inflow)
        
        #Plot results
        plt.figure()
        # Analytical solution:
        plt.plot(xr, analytical(xr, T), color = 'red')
        plt.plot(xh[2:-2], ug[2:-2], '.', markersize = 3)
        plt.title("Godunov")
        if T == 0.5:
            plt.savefig("solution_high_discont.pdf")
        elif T == 1:
            plt.savefig("solution_high_cont.pdf")


def Error_verification(T, norm):

    # Derivative of flux for BL:
    s = np.linspace(0,1,501)
    dfv = max(np.diff(flux(s))/np.diff(s))
    df = lambda u: np.zeros(len(u)) + dfv
    
    # Grid for analytical solution:
    dx = 1/2**12
    xr = np.arange(0.5*dx,1+0.5*dx,dx)
    
    # The analytical solution, which we will find the error relative to.
    uref = analytical(xr,T)
  
    # Solutions on coarser grids
    N = np.array([2**i for i in range(3,11)])


    error_upw = np.zeros(len(N))
    error_lxf = np.zeros(len(N))
    error_lxw = np.zeros(len(N))
    error_god = np.zeros(len(N))
    j = 0
    for n in N:
        dX = 1/n
        # Coarser grids for classical schemes and Godunov scheme.
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
            u0 = np.ones(len(x))*analytical(1,T)
            u0_g = np.ones(len(xg))*analytical(1,T)
            u0[x<=0] = 1.0
            u0_g[xg<=0] = 1.0
            
        uu = upw(u0, 0.995, dX, T, flux, df, inflow)
        uu = uu[1:-1]
        uf = lxf(u0, 0.995, dX, T, flux, df, inflow)
        uf = uf[1:-1]
        uw = lxw(u0, 0.995, dX, T, flux, df, inflow)
        uw = uw[1:-1]
        ug, phi = god(u0_g, 0.495, dX, T, flux, df, inflow)
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
            
    #Rates of convergence estimates
    roc_upw = np.mean(np.log(error_upw[1:] / error_upw[:-1]) / - np.log(2))
    roc_lxf = np.mean(np.log(error_lxf[1:] / error_lxf[:-1]) / - np.log(2))
    roc_lxw = np.mean(np.log(error_lxw[1:] / error_lxw[:-1]) / - np.log(2))
    roc_god = np.mean(np.log(error_god[1:] / error_god[:-1]) / - np.log(2))
        
    plt.figure()
    plt.axis('equal')
    plt.loglog(np.divide(1,N),error_upw,'o-')
    plt.loglog(np.divide(1,N),error_lxf,'o-')
    plt.loglog(np.divide(1,N),error_lxw,'o-')
    plt.loglog(np.divide(1,N), error_god, 'o-')
    plt.legend(["UW, a = {:.2f}".format(roc_upw),
                "LF, a = {:.2f}".format(roc_lxf),
                "LW, a = {:.2f}".format(roc_lxw),
                "God, a = {:.2f}".format(roc_god)], loc = 2)
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


Error_verification(0.5, 1)
#BL_solution('high', 1)