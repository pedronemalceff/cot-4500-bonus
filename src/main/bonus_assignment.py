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
print()

#Question 5
def function(t: float, w: float):
    return w - (t**3)


def do_work(t, w, h):
    basic_function_call = function(t, w)

    incremented_t = t + h
    incremented_w = w + (h * basic_function_call)
    incremented_function_call = function(incremented_t, incremented_w)

    return basic_function_call + incremented_function_call

def modified_eulers():
    original_w = .5
    start_of_t, end_of_t = (0, 3)
    num_of_iterations = 100

    # set up h
    h = (end_of_t - start_of_t) / num_of_iterations

    for cur_iteration in range(0, num_of_iterations):
        # do we have all values ready?
        t = start_of_t
        w = original_w
        h = h

        # create a function for the inner work
        inner_math = do_work(t, w, h)

        # this gets the next approximation
        next_w = w + ( (h / 2) * inner_math )

        if cur_iteration == 99:
            print("%.5f" % next_w)

        # we need to set the just solved "w" to be the original w
        # and not only that, we need to change t as well
        start_of_t = t + h
        original_w = next_w
        
    return None


modified_eulers()
    
