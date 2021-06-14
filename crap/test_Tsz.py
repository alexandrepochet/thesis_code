import pandas as pd
import numpy as np
import time as timepr
from numpy.linalg import multi_dot
import copy


def M(A,B):
    grid = [[0 for row in range(len(A))] for col in range(len(B[0]))]
    for i in range(len(A)):
# Iterating columns matrix B, start vanaf index 0
        for j in range(len(B[0])):
           # Iterating rows matrix B
           for k in range(len(B)):
                   "sommation of results"
                   grid[i][j] += A[i][k] * B[k][j]
    return grid

def Swaprows(A):
    n = len(A)
    IM = [[float(i == j) for i in range(n)] for j in range(n)]
    for j in range(n):
        row = max(range(j, n), key=lambda i: abs(A[i][j]))
        if j != row:
            IM[j], IM[row] = IM[row], IM[j]
    return IM

def SwapM(A):
    IM = Swaprows(A)
    MM= M(IM,A)
    return MM, IM

def G_S(a, b, x, g, omega):   # a is a column of coefficient matrix b augmentation x initial value of iteration g calculation accuracy
    x = x.astype(float)    #Set the precision of x, so that multiple decimals can be displayed in the calculation of x
    m, n = a.shape
    times = 0                    #Iterations
    if (m < n):
        print("There is a solution space.")    # Ensure that the number of equations is greater than the number of unknowns
    while True:
        for i in range(n):
            s1 = 0
            tempx = copy.deepcopy(x)        #Record the answer of the last iteration
            for j in range(0,i):
                s1 -= x[j] * a[i][j]
            for j in range(i+1,n):
                s1 -= tempx[j] * a[i][j]
            x[i] = (b[i] + s1) / a[i][i]*omega + (1-omega)*tempx[i]
            times += 1                                    #Number of iterations plus one
        gap = max(abs(x - tempx))              #Difference from the last answer modulus
        if gap < g:                          #Accuracy meets the requirements, end
            break
        elif times > 10000:          #If the iteration exceeds10000Times, over
            print("No convergence after 10,000 iterations")
            break
    print(times)
    print(x)
    return x

def main():
    a = np.array([[0,3,-1,8],[-1,11,-1,3],[2,-1,10,-1],[10,-1,2,0]])
    a, IM = np.array(SwapM(a))
    b = np.array([15,25,-11,6])
    x = np.array([0,0,0,0])
    g= 0.000001
    omega = 0.8
    x = G_S(a, b, x, g, omega)
    x = np.matmul(IM,np.transpose(x))

if __name__ == '__main__':
    main()
