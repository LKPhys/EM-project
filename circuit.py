import numpy as np
import PH388functions as PH388

# Task 4, 5 and 6 - Gaussian elimination, Gauss-Seidel and Jacobi's methods extended to solve the given circuit

R1, R2 = 100., 200.  # known resistance values for the circuit
R4, R6 = 400., 600.
R8, R10, R11 = 800., 1000., 1100.
Ua, Ub = 100., 200. # known voltage values for the circuit
Uc, Ud, Ue = 300., 400., 500.

R = np.array([

[-R1, -R2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[0, R2, 0, R4, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, -R4, 0, -R6, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 0, R6, 0, R8, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 0, -R8, 0, -R10, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 0, -R10, -R11],
[1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
[0, 0, 1, -1, 1, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 0, 1, -1, 1, 0, 0, 0, 0],
[0, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0],
[0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 1]
])

U = np.array([Ua, Ub-Ua, Uc-Ub, Ud-Uc, Ue-Ud, Ue, 0, 0, 0, 0, 0])
I_ge = PH388.gaussian_elimination(R, U)
I_gs = PH388.gauss_seidel(R, U)[0]
I_jc = PH388.jacobi(R, U, 1000, guess=np.ones((len(U),)))

# Checking zero-ness of each loop, all are zero or extremely close to zero
# Gaussian Elimination
a = -R1*I_ge[0] - R2*I_ge[1] - Ua
b = R2*I_ge[1] + R4*I_ge[3] - Ub + Ua
c = -R4*I_ge[3] - R6*I_ge[5] - Uc + Ub
d = R6*I_ge[5] + R8*I_ge[7] - Ud + Uc
e = -R8*I_ge[7] - R10*I_ge[9] - Ue + Ud
f = -R10*I_ge[9] -R11*I_ge[10] - Ue

print('gaussian_elimination solution vector I:\n', I_ge)
print('Zero-ness of each loop:\n', a, b, c, d, e)

print(I_gs)
# Gauss-Seidel
a = -R1*I_gs[0] - R2*I_gs[1] - Ua
b = R2*I_gs[1] + R4*I_gs[3] - Ub + Ua
c = -R4*I_gs[3] - R6*I_gs[5] - Uc + Ub
d = R6*I_gs[5] + R8*I_gs[7] - Ud + Uc
e = -R8*I_gs[7] - R10*I_gs[9] - Ue + Ud
f = -R10*I_gs[9] -R11*I_gs[10] - Ue

print('Gauss-Seidel solution vector I:\n', I_gs)
print('Zero-ness of each loop:\n', a, b, c, d, e)

# Jacobi method
a = -R1*I_jc[0] - R2*I_jc[1] - Ua
b = R2*I_jc[1] + R4*I_jc[3] - Ub + Ua
c = -R4*I_jc[3] - R6*I_jc[5] - Uc + Ub
d = R6*I_jc[5] + R8*I_jc[7] - Ud + Uc
e = -R8*I_jc[7] - R10*I_jc[9] - Ue + Ud
f = -R10*I_jc[9] -R11*I_jc[10] - Ue

print('Jacobi solution vector I:\n', I_jc)
print('Zero-ness of each loop:\n', a, b, c, d, e)
