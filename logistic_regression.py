import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

marker_size = mpl.rcParams['lines.markersize'] ** 2


np.random.seed(0)

x = np.random.rand(500) * 22 + 2000
y = np.zeros_like(x)

a, b = 0.5, -2018

def sigmoid(x):

    return( 1 / (1 + np.exp(-x)) )

def dsigmoid(x):

    s = sigmoid(x)

    return( s * (1 - s) )

for i, year in enumerate(x):

    sample = np.random.rand()

    probability = sigmoid(a * (year + b))

    if sample < probability:

        y[i] = 1

    else:

        y[i] = 0



A = np.linspace(0.4, 2.3, 30)
B = np.linspace(-2019, -2017, 30)
A, B = np.meshgrid(A, B)
Z = sum([(sigmoid(A * (x[i] + B)) - y[i])**2 for i in range(len(x))])

fig, ax = plt.subplots()

ax.contour(A, B, Z, 20, zorder=0)

U, V = np.meshgrid(A, B)

U = sum([(x[i] + B) * dsigmoid(A * (x[i] + B)) * 2 * (sigmoid(A * (x[i] + B)) - y[i]) for i in range(len(x))])
V = sum([A * dsigmoid(A * (x[i] + B)) * 2 * (sigmoid(A * (x[i] + B)) - y[i]) for i in range(len(x))])

U = -U
V = -V

q = ax.quiver(A, B, U, V, zorder=5)


av = 1.5
bv = -2017.1

trace_a = [av]
trace_b = [bv]

eta = 0.03

for i in range(12):

    da = sum([(x[i] + bv) * dsigmoid(av * (x[i] + bv)) * 2 * (sigmoid(av * (x[i] + bv)) - y[i]) for i in range(len(x))])
    db = sum([av * dsigmoid(av * (x[i] + bv)) * 2 * (sigmoid(av * (x[i] + bv)) - y[i]) for i in range(len(x))])

    av = av - eta * da
    bv = bv - eta * db

    trace_a.append(av)
    trace_b.append(bv)

    ax.set_ylabel("$b$")
    ax.set_xlabel("$a$")
    ax.plot(trace_a, trace_b, c="green")

    plt.savefig(f"./logistic_regression_gd{i}.png")
