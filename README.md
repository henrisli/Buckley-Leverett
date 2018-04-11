# Buckley-Leverett
TMA4212
To run files:
Run the functions in Run.py.

Error_verification() computes numerical errors for our four schemes for N = 2^k, k=3,...,10 grid points, and plots these errors. It takes 2 arguments: Time to evaluate solution (1 gives continuous solution, 0.5 gives discontinuous solution) and norm to evaluate error (1, 2, np.inf).

BL_solution() shows the solution of the Buckley-Leverett equation for 0<x<1. It takes 3(2) arguments:
method to use ('classical', 'high'), Time to evaluate solution (1 gives continuous solution, 0.5 gives discontinuous solution) and possibly M, a constant for relative saturations.

Advection_classic_schemes() takes no arguments.

Advection_high_resolution_schemes() takes no arguments.