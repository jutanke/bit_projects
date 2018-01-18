import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def applyPCA(X,k):
    m, n = X.shape

    # Step 1: Mean normalization
    mu = np.reshape(np.mean(X, axis=1), newshape=(m, 1))
    print("Shape of mu: ", mu.shape)

    X_normalized = X - mu

    # Step 2: Compute covariance matrix
    X_Cov = np.cov(X_normalized, rowvar=1, ddof=1)

    print("Shape of covariance matrix: ", X_Cov.shape)

    # Step 3: Compute eigenvctors and eigenvalues of the covariance matrix
    # Note: If using np.eig() instead of np.eigh() projections will be complex numbers
    # Ascending order
    eigenVals, eigenDecoder = np.linalg.eigh(X_Cov)
    # Descending order
    eigenVals = np.flip(eigenVals, axis=-1)
    eigenDecoder = np.flip(eigenDecoder,axis=1)

    # Take frst k columns of the decoder
    decoder = eigenDecoder[:, :k]

    print("Shape of eigen vector matrix: ", eigenDecoder.shape)
    print("Shape of decoder: ", decoder.shape)

    # Poject into R^2
    Z = np.matmul(decoder.T, X_normalized)

    print("Shape of projected data: ", Z.shape)

    return Z

if __name__ == '__main__':
    pathX = '../data/third_project/data-dimred-X.csv'
    pathY = '../data/third_project/data-dimred-y.csv'
    X = np.loadtxt(fname=pathX,delimiter=',',dtype=float)
    Y = np.loadtxt(fname=pathY, dtype=float)
    k = 2

    print("Shape of X: ",X.shape)
    print("Shape of Y: ", Y.shape)

    fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    projected = applyPCA(X,k)
    # print(projected)

    plt.scatter(projected.T[:,0],projected.T[:,1],c=Y)
    # Axes3D.scatter(xs=projected,ys=Y)
    plt.show()




