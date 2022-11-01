import scipy as sp
import math

def identity(matrix: sp.sparse.csc_matrix):
    return matrix

def general_pade(m: int, n: int, matrix: sp.sparse.csc_matrix):
    numerator = sp.sparse.identity(matrix.shape[0], format='csc')
    denominator = sp.sparse.identity(matrix.shape[0], format='csc')
    for i in range(1, m+1):
        numerator += (math.factorial(n+m-i) * math.factorial(m) / (math.factorial(n+m) * math.factorial(i) * math.factorial(m-i))) * (matrix**i)
    for i in range(1, n+1):
        denominator += (math.factorial(n+m-i) * math.factorial(n) / (math.factorial(n+m) * math.factorial(i) * math.factorial(n-i))) * ((-1 * matrix)**i)
    return numerator * sp.sparse.linalg.inv(denominator)

def pade_1_1(matrix: sp.sparse.csc_matrix):
    return general_pade(1, 1, matrix)

def pade_1_2(matrix: sp.sparse.csc_matrix):
    return general_pade(1, 2, matrix)
    

''' for testing purposes
dx = 0.01
dt = 0.001
D = 1e-6
A = sp.sparse.diags([1, -2, 1], [-1, 0, 1], shape = (5, 5))
A = A.tocsc()
A[0, 0] = -1
A[4, 4] = -1
A = D * dt / (dx**2) * A
B = general_pade(2, 2, A)
C = general_pade(4, 4, A)
E = general_pade(1, 1, A)
F = general_pade(0, 0, A)
A = sp.sparse.linalg.expm(A)

print(A.toarray())
print(B.toarray())
print(C.toarray())

print()
print((A-B).toarray())

print()
print(E.toarray())
print((A-E).toarray())
print(F.toarray())
'''