from scipy.cluster.vq import kmeans
from third_project.task_3_1 import macQueenClustering, hartiganClustering
import numpy as np
import matplotlib.pyplot as plt
from scipy import  cluster
from sklearn.metrics.pairwise import euclidean_distances

beta= 1.

def spectralClusterig(data):
    global beta
    diff = (euclidean_distances(data, data) ** 2) * beta
    S = np.exp(diff)
    D = np.matmul(np.sum(S,axis=1),np.identity(S.shape[0]))
    L = D-S
    eigenvalues,eigenvectors = np.linalg.eig(L)
    # Sort ascending order
    sortedIndex = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sortedIndex]
    eigenvectors = eigenvectors[sortedIndex]

    fiedlerVector = eigenvectors[1]
    clusterAssignments = np.ones(200) - ((fiedlerVector>0)*1.)

    return clusterAssignments


if __name__ == '__main__':

    dataPath = '../data/third_project/data-clustering-2.csv'
    data = np.loadtxt(dataPath, dtype=np.object, comments='#', delimiter=',').astype('float32')
    data = data.T
    k = 2

    fig = plt.figure(figsize=(10, 10))
    plot1 = fig.add_subplot(221)
    plot2 = fig.add_subplot(222)
    plot3 = fig.add_subplot(223)
    plot4 = fig.add_subplot(224)

    lloydsCentroids,distortion = kmeans(data, k_or_guess=k)
    hartigansCentroids = hartiganClustering(k, data)
    macQueenCentroids = macQueenClustering(k, data)


    assignment, cdist = cluster.vq.vq(data, lloydsCentroids)
    plot1.scatter(data[:, 0], data[:, 1], c=assignment)
    plot1.scatter(lloydsCentroids[:, 0], lloydsCentroids[:, 1], c='r')

    assignment, cdist = cluster.vq.vq(data, hartigansCentroids)
    plot2.scatter(data[:, 0], data[:, 1], c=assignment)
    plot2.scatter(hartigansCentroids[:, 0], hartigansCentroids[:, 1], c='r')

    assignment, cdist = cluster.vq.vq(data, macQueenCentroids)
    plot3.scatter(data[:, 0], data[:, 1], c=assignment)
    plot3.scatter(macQueenCentroids[:, 0], macQueenCentroids[:, 1], c='r')

    spectralAssignments = spectralClusterig(data)
    plot4.scatter(data[:, 0], data[:, 1], c=spectralAssignments)
    # plot4.scatter(macQueenCentroids[:, 0], macQueenCentroids[:, 1], c='r')

    plt.show()