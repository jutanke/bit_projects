
import numpy as np
import pylab
import matplotlib.pyplot as plt

# based on https://glowingpython.blogspot.de/2011/04/plotting-p-norm-unit-circles-with.html?m=1
def drawUnitCircle(p):
    X = []
    Y = []
    for i in range(5000):
        x = np.random.rand()*2-1
        y = np.random.rand()*2-1
        vec = np.asarray([x, y], dtype=float)
        if np.linalg.norm(vec, p) <= 1:
            X.append(x)
            Y.append(y)

    plt.scatter(x=X,y=Y)
    plt.xlim(-2., 2.)
    plt.ylim(-2., 2.)
    plt.show()


if __name__ == '__main__':
    drawUnitCircle(0.5)