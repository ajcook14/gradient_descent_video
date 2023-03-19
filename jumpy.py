import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

marker_size = mpl.rcParams['lines.markersize'] ** 2


np.random.seed(0)

c = 2.
d = 5.

A = np.linspace(-2, 2, 30)
B = np.linspace(-2, 2, 30)
A, B = np.meshgrid(A, B)
Z = sum([c * A**2 + d * B**2 for i in range(500)])

fig2, ax2 = plt.subplots()

ax2.contour(A, B, Z, 20, zorder=0)

U, V = np.meshgrid(A, B)

U = 2 * c * A
V = 2 * d * B

U = -U
V = -V

q = ax2.quiver(A, B, U, V, zorder=5)


av = 1.
bv = 1.

trace_a = [av]
trace_b = [bv]

eta = 0.205

for i in range(12):

    da = 2 * c * av
    db = 2 * d * bv

    av = av - eta * da
    bv = bv - eta * db

    trace_a.append(av)
    trace_b.append(bv)

    ax2.set_ylabel("$b$")
    ax2.set_xlabel("$a$")
    ax2.plot(trace_a, trace_b, c="green")


plt.savefig(f"./jumpy.png")
