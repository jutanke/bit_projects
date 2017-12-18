import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
import numpy.polynomial as pol
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    # read data as 2D array of data type 'object'
    data = np.loadtxt('../whData.dat',dtype=np.object,comments='#',delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)

    # 1D arrays for weight height and gender
    w = np.copy(X[:,0])
    h = np.copy(X[:,1])
    g = data[:,2]
    

    h_sort_indices = np.argsort(h)
    w = w[h_sort_indices] 
    h = h[h_sort_indices]
    g = g[h_sort_indices]
    

    weight_greater_than_zero = np.where( w > 0 )
    weight_less_than_zero = np.where( w < 0 )

    h_n = h[weight_less_than_zero] # height vals where weight is -1
    w   = w[weight_greater_than_zero]
    h   = h[weight_greater_than_zero]
    g   = g[weight_greater_than_zero]

    X = np.linspace(np.amin(h), np.amax(h), 50)
    Y = np.linspace(np.amin(w), np.amax(w), 50)
    X, Y = np.meshgrid(X, Y)

    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    h_w = np.vstack((h,w)).T
    mu = np.mean(h_w, axis=0)
    cov = np.cov(h_w, rowvar=0)

    F = multivariate_normal(mu, cov)
    Z = F.pdf(pos)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=1, antialiased=True, cmap=cm.viridis)
    ax.scatter(h, w, np.zeros(21), c='r', marker='o')

    ax.set_xlabel('\n' + 'height', linespacing=4)
    ax.set_ylabel('\n' + 'weight', linespacing=4)
    ax.set_zlabel('\n' + 'joint probability', linespacing=4)
    ax.set_zlim(0,0.003)
    ax.set_zticks(np.linspace(0,0.003,5))
    ax.view_init(27, -21)

    plt.savefig('result.png')
    #plt.show()