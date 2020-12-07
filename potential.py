import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import PH388functions as PH388


def finiteDifference(M, rho, epsilon, delta=1,):
    """
    This function sets up the matrices A and b. The overall method for matrix A is done by
    using two nested for loops, the inner loop runs through each column and the outer loop
    runs through each row. These for loops assign the correct number to each specific index
    of the matrix A.

    The procedure for b is much more simple, the charge everywhere is 0 except the middle index
    so the function simply finds the middle index and inserts -1 there.
    (The charge is +1 but the equation moves the charge to the RHS so it is -1 in the code.)
    """

    # Setting up initial matrix and length
    m = np.zeros((M, M))
    n = len(m)
    A = np.zeros((n*n, n*n))

    # Nested for loops assigning correct values to each index
    for i in range(len(A)):
        for j in range(len(A)):
            if i == j:
                A[i][j] = -4
            elif i == j + 1:
                A[i][j] = 1
            elif i == j + n:
                A[i][j] = 1
            elif i == j - 1:
                A[i][j] = 1
            elif i == j - n:
                A[i][j] = 1
            else:
                continue

    for i in range(1, len(A)):
        for j in range(1, len(A)):
            if i % n == 0 and j == i - 1:
                A[i][j] = 0
            elif j % n == 0 and i == j - 1:
                A[i][j] = 0

    # creating grid, this is a bit obsolete here since np.arange does
    # the same job for delta = 1 but including for completeness atm

    xgrid = np.arange(0, (len(m)), delta)
    ygrid = xgrid.copy()

    # Setting up b column vector by finding middle index
    b = np.zeros((n*n, 1))
    midb = int((n*n)/2)
    b[midb] = -rho/epsilon

    return A, b, xgrid, ygrid

# Initial variables
rho = epsilon = delta = 1
M = 21 # desired dimensions

# Function calls
A, b, x, y = finiteDifference(M, rho, epsilon, delta)  # takes M = N value as argument
U, w, i = PH388.SOR(A, b.flatten())
#U2, i2 = PH388.gauss_seidel(A, b.flatten(), 1000)

# Print values
print('Solution Vector: ', U)
print('Optimal Relaxation Parameter: ', w)
print('Iterations for convergence :', i)

# Plots
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(x, y)
U = U.reshape((M, M))
surf = ax.plot_surface(X, Y, U, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.55, aspect=5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('ϕ')
ax.set_title('Potential ϕ(x, y)')
plt.show()
