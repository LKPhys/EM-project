import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import PH388functions as PH388

def finiteDifferenceDirichlet(m, rho, epsilon, delta, topBound, rightBound, leftBound, bottomBound):
    """
    This function sets up the matrices A and b. The overall method for matrix A is done by
    using two nested for loops, the inner loop runs through each column and the outer loop
    runs through each row. These for loops assign the correct number to each specific index
    of the matrix A.

    The procedure for b is to assign boundary conditions to each node. b is created as a
    matrix first so its more easy to visualise, then using nested for loops, like in matrix A,
    it loops through each column of each row and assigns each node its respective boundary value.
    Then b is flattened into an array.
    The boundary conditions can be changed in the function call for other problems to do
    with Dirichlet boundarys.
    """

    # Set up initial matrix A and value n.
    n = len(m)
    A = np.zeros((n*n, n*n))

    # Nested for loop assigning correct values to each index
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

    # Creating grid, this is a bit obsolete here since np.arange does the
    # same job for delta = 1 but including for completeness atm
    xgrid = np.zeros((len(m)))
    delta = 1
    for i in range(1, len(xgrid)):
        xgrid[i] = xgrid[i-1] + delta

    ygrid = xgrid.copy()

    # Vector b is simply the flat version of boundary matrix
    boundaryMatrix = np.zeros((n, n), float)

    # Assign each index of boundary matrix with the correct boundary
    # conditions, loop i through each row, j through each column
    for i in range(n):
        for j in range(n):

            # TOP LEFT
            if i == 0 and j == 0:
                boundaryMatrix[i][j] = 0

            # ONE BOUNDARY TOP ROW
            if i == 0 and j != 0 and j != n-1:
                boundaryMatrix[i][j] = - topBound

            # TWO BOUNDARYS TOP RIGHT
            if i == 0 and i % n == 0 and j == n-1:
                boundaryMatrix[i][j] = - topBound - rightBound

            # ONE BOUNDARY LEFT COLUMN
            if j == 0 and i != 0 and i != n-1:
                boundaryMatrix[i][j] = - leftBound

            # ONE BOUNDARY BOTTOM ROW
            if i == n-1 and j != 0 and j != n-1:
                boundaryMatrix[i][j] = - bottomBound

            # ONE BOUNDARY RIGHT COLUMN
            if i != 0 and i != n-1 and j == n-1:
                boundaryMatrix[i][j] = - rightBound

            # TWO BOUNDARYS BOTTOM LEFT
            if i == n-1 and j == 0:
                boundaryMatrix[i][j] = - bottomBound - leftBound

            # BOTTOM RIGHT
            if i == n-1 and j == n-1:
                boundaryMatrix[i][j] = 0

            # CENTER
            if i == int(n/2) and j == int(n/2):
                boundaryMatrix[i][j] = -rho/epsilon

    # Flatten boundary matrix as b should be an array
    b = boundaryMatrix.flatten()

    return A, b, xgrid, ygrid

# Setup the initial variables
M = N = 11
rho = epsilon = delta = 1
topBound = rightBound = -0.3
bottomBound = leftBound = 0.3
m = np.zeros((M, N))

# Call functions
A, b, x, y = finiteDifferenceDirichlet(m, rho, epsilon, delta, topBound, rightBound, leftBound, bottomBound)
U, w, i = PH388.SOR(A, b.flatten())

midU = int(M**2/2)

# Print values
print('Solution vector: ', U)
print('Optimal Relaxation Parameter: ', w)
print('Iterations Required (SOR): ', i)
print('Peak: ', U[midU])

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
