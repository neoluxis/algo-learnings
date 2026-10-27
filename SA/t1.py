import numpy as np

T: np.float64 = 2e3
dT: np.float64 = 0.99
eps: np.float64 = 1e-14

while (T > eps):
    # An anneal step
    
    T = T * dT
    
