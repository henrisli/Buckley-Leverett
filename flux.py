
import numpy as np
# BL Flux function
def flux(u):
    return np.divide(np.power(u, 2), np.power(u, 2) + np.power(1 - u, 2))