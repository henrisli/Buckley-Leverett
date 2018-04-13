
import numpy as np

def god(u0, cfl, dx, T, flux, dflux, boundary):
    def lim(r):
        return np.maximum(0, np.minimum(1.3*r, np.minimum(0.5+0.5*r, 1.3)))
    
    def limiter(a,b):
        return lim(np.divide(b, a + 1e-6))*a
    
    dt = cfl*dx/max(abs(dflux(u0)))
    u = np.copy(u0)
    U = np.copy(u0)
    t = 0.0
    n = len(u0)
    f = np.zeros(n)
    i = np.arange(2,n-2)
    j = np.arange(n-1)
    phi = np.zeros(n)
    while t<T:
        if (t+dt > T):
            dt = T-t
        
        t += dt
        u = boundary(u, 2)
        du = np.diff(u)
        phi[1:-1] = limiter(du[:-1], du[1:]) 
        phi[0] = phi[-2]
        phi[-1] = phi[1]  
        ur = u + 0.5*phi
        fr = flux(ur)
        dfr = dflux(ur)
        mdf = max(dfr)
        f[j] = fr[j]
        U[i] = u[i] - dt/dx*(f[i]-f[i-1])
          
        U = boundary(U,2)
        du = np.diff(U)
        phi[1:-1] = limiter(du[:-1], du[1:])
        phi[0] = phi[-2]
        phi[-1] = phi[1]  
        ur = U + 0.5*phi
        fr = flux(ur)
        dfr = dflux(ur)
        mdf = np.maximum(dfr, mdf)
        mdf = max(mdf)
        f[j] = fr[j]
        u[i] = 0.5*u[i] + 0.5*( U[i]-dt/dx*(f[i]-f[i-1]) )
        dt = cfl*dx/mdf
    u = boundary(u, 2)
    du = np.diff(u)
    phi[1:-1] = limiter(du[:-1], du[1:])
    phi[0] = phi[-2]
    phi[-1] = phi[1]
    return u, phi