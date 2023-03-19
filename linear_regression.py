import numpy as np
import matplotlib.pyplot as plt

data_size = 20

a, b = 0.4, -1.2

x = np.linspace(-10., 10., data_size) + np.random.normal(0., 0.5, data_size)

y = a * x + b

errors = np.random.normal(0., 2., data_size)

y += errors

fig, ax = plt.subplots()

ax.scatter(x, y)
ax.plot(x, a*x + b, c='red')
ax.set_ylabel('$y$')
ax.set_xlabel('$x$')

plt.savefig("./beamer/linear_regression.png")
