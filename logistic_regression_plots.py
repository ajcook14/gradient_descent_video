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



x_true = np.linspace(2000.3, 2021.7, 100)
y_true = sigmoid(a * (x_true + b))

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

    fig, ax = plt.subplots()

    ax.scatter(x, y, s=marker_size/4)
    ax.set(yticklabels=[], yticks=[])
    ax.set_ylabel("petrol                                                       electric")
    ax.set_xlabel("year")
    ax.set_title("Car type for sales from year 2000")

    y_predicted = sigmoid(av * (x_true + bv))
    ax.plot(x_true, y_true, c="red")
    ax.plot(x_true, y_predicted, c="green")

    plt.savefig(f"./logistic_regression{i}.png")
