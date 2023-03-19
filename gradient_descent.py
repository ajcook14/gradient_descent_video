import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D


a, b = 1, 100
X = np.linspace(-2, 2, 50)
Y = np.linspace(-1, 3, 50)
X, Y = np.meshgrid(X, Y)
Z = ((a - X)**2 + b * (Y - X**2)**2)



"""
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

plt.savefig("./gd.png")
"""



fig, ax = plt.subplots()

ax.contour(X, Y, Z, list(map(lambda x: x**2, range(20))), zorder=0)

plt.savefig("./gd_contour.png")


X = np.linspace(-2, 2, 20)
Y = np.linspace(-1, 3, 20)
X, Y = np.meshgrid(X, Y)
U, V = np.meshgrid(X, Y)

U = -2 * (a - X) - 4 * X * b * (Y - X**2)
V = 2 * b * (Y - X**2)

#U = -U
#V = -V

q = ax.quiver(X, Y, U, V, zorder=5)

plt.savefig("./gd_contour_grad.png")
