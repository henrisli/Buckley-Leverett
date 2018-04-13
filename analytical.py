
import numpy as np

#Analytical solution:
def analytical(x,T):
    return 1/2*(np.sqrt(np.divide(-2*x/T + np.sqrt(4*x/T + 1) - 1, x/T) + 1) + 1)*np.logical_not(x > ((1/2 + 1/np.sqrt(2))*T))    