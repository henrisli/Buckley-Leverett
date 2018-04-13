
import numpy as np

def lxw(u0, cfl, dx, T, flux, df, boundary):
    u = np.copy(u0)
    U = np.copy(u0)
    t = 0.0
    dt = cfl*dx/max(abs(df(u0)))
    i = np.arange(1, len(u0)-1, 1)
    while t<T:
        if t+dt > T:
            dt = T-t
        t += dt
        u = boundary(u)
        f = flux(u)
        U[i] = 0.5*(u[i]+u[i+1]) - 0.5*dt/dx*(f[i+1]-f[i]) #Tar vekk 0.5 får mye bedre oppførsel??
        U = boundary(U)
        f = flux(U)
        u[i] = u[i] - dt/dx*(f[i]-f[i-1])
        dt = cfl*dx/max(abs(df(u)))
    return u