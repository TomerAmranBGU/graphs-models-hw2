import numpy as np

# Exercise 1
def G(row_s, Temp):
    return np.exp(1/Temp * row_s[:-1]@row_s[1:].T)

# Exercise 2
def F(row_s, row_t, Temp):
    return np.exp(1/Temp * row_s@row_t.T)

# exercise 3
# to achive the Ztemp definition we can combine G and F with exponent multipication rules
def Ztemp_brute(Temp):
    # options = np.array(np.meshgrid([-1,1],[-1,1])).T.reshape(-1,2)
    options = np.array([[1,1],[-1,1],[1,-1],[-1,-1],])
    res = 0
    for row_s in options:
        for row_t in options:
            res += G(row_s,Temp)*G(row_t,Temp)*F(row_s,row_t,Temp);
    return res
print(Ztemp_brute(1))
print(Ztemp_brute(1.5))
print(Ztemp_brute(2))