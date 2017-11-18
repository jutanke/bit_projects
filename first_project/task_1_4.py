
import numpy as np
import pylab
import matplotlib.pyplot as plt

# based on https://glowingpython.blogspot.de/2011/04/plotting-p-norm-unit-circles-with.html?m=1
def drawUnitCircle(p):
    X = []
    Y = []
    epsilon = 0.1
    for i in range(100000):
        x = np.random.rand()*2-1
        y = np.random.rand()*2-1
        vec = np.asarray([x, y], dtype=float)
        pNorm = np.linalg.norm(vec, p)
        if pNorm >= 1 - epsilon and pNorm <= 1 + epsilon:
            X.append(x)
            Y.append(y)

    plt.scatter(x=X,y=Y,s=0.2)
    plt.xlim(-2., 2.)
    plt.ylim(-2., 2.)
    plt.show()


if __name__ == '__main__':
    drawUnitCircle(0.5)