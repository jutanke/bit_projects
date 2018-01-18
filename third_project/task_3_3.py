import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


def multiClassLDA(X,Y,k):
    mu = np.reshape(np.mean(X, axis=1), newshape=(X.shape[0], 1))
    uniqueLabels = np.unique(Y)

    m,n = X.shape
    S_B = np.zeros(shape=(m,m))
    S_W = np.zeros(shape=(m,m))

    print(S_B.shape)

    for l in uniqueLabels:
        label_indices = np.ndarray.flatten(np.argwhere((Y == l) * 1.))
        X_l = X.T[label_indices].T
        mu_l = np.reshape(np.mean(X_l, axis=1), newshape=(X_l.shape[0], 1))
        cov_l = np.cov(X_l, rowvar=1, ddof=1)
        S_B += np.matmul(mu_l-mu,(mu_l-mu).T)
        S_W += cov_l

    print("Shape of S_B: ",S_B.shape)
    print("Shape of S_W: ", S_W.shape)

    eigenVals, eigenDecoder = np.linalg.eigh(np.matmul(np.linalg.pinv(S_W),S_B))
    eigenVals = np.flip(eigenVals, axis=-1)
    eigenDecoder = np.flip(eigenDecoder, axis=1)
    # Take frst k columns of the decoder
    decoder = eigenDecoder[:, :k]

    Z = np.matmul(decoder.T, X)

    return Z

# def multiClassLDA(X,Y,k):
#     label_1_indices = np.ndarray.flatten(np.argwhere((Y == 1) * 1.))
#     label_2_indices = np.ndarray.flatten(np.argwhere((Y == 2) * 1.))
#     label_3_indices = np.ndarray.flatten(np.argwhere((Y == 3) * 1.))
#     X_1 = X.T[label_1_indices].T
#     X_2 = X.T[label_2_indices].T
#     X_3 = X.T[label_3_indices].T
#
#     mu = np.reshape(np.mean(X, axis=1), newshape=(X.shape[0], 1))
#     mu_1 = np.reshape(np.mean(X_1, axis=1), newshape=(X_1.shape[0], 1))
#     mu_2 = np.reshape(np.mean(X_2, axis=1), newshape=(X_2.shape[0], 1))
#     mu_3 = np.reshape(np.mean(X_3, axis=1), newshape=(X_3.shape[0], 1))
#
#     X_Cov_1 = np.cov(X_1, rowvar=1, ddof=1)
#     X_Cov_2 = np.cov(X_2, rowvar=1, ddof=1)
#     X_Cov_3 = np.cov(X_3, rowvar=1, ddof=1)
#
#     S_B = np.matmul(mu_1-mu,(mu_1-mu).T) + np.matmul(mu_2-mu,(mu_2-mu).T) + np.matmul(mu_3-mu,(mu_3-mu).T)
#     S_W = X_Cov_1 + X_Cov_2 + X_Cov_3
#
#     print("Shape of S_B: ",S_B.shape)
#     print("Shape of S_W: ", S_W.shape)
#
#     eigenVals, eigenDecoder = np.linalg.eigh(np.matmul(np.linalg.pinv(S_W),S_B))
#     eigenVals = np.flip(eigenVals, axis=-1)
#     eigenDecoder = np.flip(eigenDecoder, axis=1)
#     # Take frst k columns of the decoder
#     decoder = eigenDecoder[:, :k]
#
#     Z = np.matmul(decoder.T, X)
#
#     return Z

def applyPCA(X,k):
    m, n = X.shape

    # Step 1: Mean normalization
    mu = np.reshape(np.mean(X, axis=1), newshape=(m, 1))
    print("Shape of mu: ", mu.shape)

    X_normalized = X - mu

    # Step 2: Compute covariance matrix
    X_Cov = np.cov(X_normalized, rowvar=1, ddof=1)

    print("Shape of covariance matrix: ", X_Cov.shape)

    # Step 3: Compute eigenvectors and eigenvalues of the covariance matrix
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

    # Poject into R^k
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
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # PCA
    projectedPCA = applyPCA(X, k)
    # print(projected)
    ax1.scatter(projectedPCA.T[:, 0], projectedPCA.T[:, 1], c=Y)



    # Multiclass LDA
    projectedLDA = multiClassLDA(X, Y,k)
    ax2.scatter(projectedLDA.T[:, 0], projectedLDA.T[:, 1], c=Y)


    # Axes3D.scatter(xs=projected,ys=Y)
    plt.show()




