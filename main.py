import numpy as np
def G(row_s, Temp):
    return np.exp(1/Temp*(row_s[:-1]@row_s[1:].T))
