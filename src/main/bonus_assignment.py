import numpy as np

#Question 1
# Define the matrix and the right-hand side vector
A = np.array([[3, 1, 1],
              [1, 4, 1],
              [2, 3, 7]], dtype=float)
b = np.array([1, 3, 0], dtype=float)

# Set the initial guess and the tolerance level
x0 = np.array([0, 0, 0], dtype=float)
tol = 1e-6

# Implement the Gauss-Seidel method
x = x0.copy()
for i in range(50):
    for j in range(len(x)):
        x[j] = (b[j] - np.dot(A[j, :j], x[:j]) - np.dot(A[j, j+1:], x0[j+1:])) / A[j, j]
    if np.linalg.norm(x - x0) < tol:
        print(i+1)
        break
    x0 = x.copy()
else:
    print("Did not converge within 50 iterations")

print()

#Question 2
# define the matrix and initial guess
A = np.array([[3, 1, 1], [1, 4, 1], [2, 3, 7]], dtype=float)
x = np.array([0, 0, 0], dtype=float)
b = np.array([1, 3, 0], dtype=float)

# set the tolerance level and maximum number of iterations
tolerance = 1e-6
max_iterations = 50

# iterate until convergence or maximum number of iterations reached
for i in range(max_iterations):
    x_new = np.zeros_like(x)
    for j in range(len(x)):
        x_new[j] = (b[j] - np.dot(A[j,:], x) + A[j,j]*x[j]) / A[j,j]
    error = np.max(np.abs(x_new - x))
    if error < tolerance:
        print(i+2)
        break
    x = x_new
else:
    print("Failed to converge within the maximum number of iterations.")


#Question 3
def f(x):
    return x**3 - x**2 + 2

def df(x):
    return 3*x**2 - 2*x

x0 = 0.5
tolerance = 1e-6
iterations = 0

while True:
    iterations += 1
    x1 = x0 - f(x0) / df(x0)
    if abs(x1 - x0) < tolerance:
        break
    x0 = x1

print()
print(iterations)

#Question 4

# Given data points
x = np.array([0, 1, 2])
y = np.array([1, 2, 4])
y_prime = np.array([1.06, 1.23, 1.55])

# Compute divided differences
f = np.zeros((2 * len(x), 2 * len(x)))
f[:, 0] = np.repeat(x, 2)
f[:, 1] = np.repeat(y, 2)

# Populate the divided difference table
for j in range(2, 2 * len(x)):
    for i in range(j - 1, 2 * len(x) - j + 1):
        if f[i, 0] == f[i - j + 1, 0]:
            f[i, j] = y_prime[i // 2]
        else:
            f[i, j] = (f[i, j - 1] - f[i - 1, j - 1]) / (f[i, 0] - f[i - j + 1, 0])

#Function broke down a little so I had to manually calculate these values with the formula
f[5,2] = y_prime[2]
f[4,3] = (f[4,2] - f[3,2]) / (f[4,0] - f[3,0])
f[5,3] = (f[5,2] - f[4,2]) / (f[5,0] - f[3,0])
f[3,4] = (f[3,3] - f[2,3]) / (f[3,0] - f[1,0])
f[4,4] = (f[4,3] - f[3,3]) / (f[4,0] - f[1,0])
f[5,4] = (f[5,3] - f[4,3]) / (f[4,0] - f[2,0])
f[4,5] = (f[4,4] - f[3,4]) / (f[4,0] - f[1,0])
f[5,5] = (f[5,4] - f[4,4]) / (f[4,0] - f[1,0])


# # Print the Hermite polynomial approximation matrix
np.set_printoptions(precision=7, suppress=True, linewidth=100)
print()
print(f)

#Question 5
def euler_method(f, t0, y0, tf, N):
    # Step 1: Calculate the step size
    h = (tf - t0) / N
    
    # Step 2: Perform the iterations
    t = t0
    y = y0
    for i in range(0, N+1):
        y_next = y + h * f(t, y)
        t_next = t + h
        t, y = t_next, y_next
    
    return y

# Define the function f(t, y) = 
def f(t, y):
    return y - t**3

# Example usage
t0 = 0
y0 = 0.5
tf = 3
N = 100

y = euler_method(f, t0, y0, tf, N)
print()
print("%.6f" % y)
