import matplotlib.pyplot as plt
import numpy as np
from scipy import  cluster
from scipy.cluster.vq import kmeans
import math
import sys

'''
Each function makes some change to a figure: Creates a figure, creattes a plotting area, plots some lines in a plotting
area etc.
Various states (e.g. current figure) are preserved across function calls.
Axes: Its the plot. The region of the image with the data space.
Plotting functions are directed to the current axes.
'''


def macQueenClustering(k,data):
    # initialize centroids
    centroids_indices = np.random.randint(0,len(data),size=k)
    centroids = data[centroids_indices]
    nmbrElementsInClusters = np.zeros(shape=k)

    # compute winner clusters for each data point
    for dp in data:
        assignment, cdist = cluster.vq.vq([dp], centroids)
        assignment = assignment[0]
        # update cluster size and centroid
        nmbrElementsInClusters[assignment] += 1
        c_u_i = centroids[assignment]
        c_u_i = c_u_i + 1./nmbrElementsInClusters[assignment] * (dp-c_u_i)
        centroids[assignment] = c_u_i

    return centroids


def computeCentroid(data):
    mean_x = np.sum(data[:, 0]) / data.shape[0]
    mean_y = np.sum(data[:, 1]) / data.shape[0]

    return np.array([mean_x, mean_y])

'''
k: Number of clusters
'''
def hartiganClustering(k,data):
    clusterAssignments = np.random.randint(0,k,size=data.shape[0])
    clusters = []
    centroids = []
    assignmentDict = dict()

    # Compute initial centroids
    for k_i in range(k):
        indicies = np.argwhere(clusterAssignments==k_i).flatten()
        cluster_i = data[indicies]
        for dp in cluster_i:
            assignmentDict[tuple(dp)] = k_i
        print("Length: ", len(cluster_i))
        clusters.append(cluster_i)
        centroids.append(computeCentroid(cluster_i))

    centroids = np.array(centroids)
    clusters = np.array(clusters)

    converged = -1

    countIter = 0
    countRemove = 0
    while converged != 1:
        converged = 1
        # For each point computer winner cluster c_w
        for data_i in data:
            clusterNmbr = assignmentDict[tuple(data_i)]
            cluster_i = clusters[clusterNmbr]

            if len(cluster_i) > 1:
                removeIndex = np.where(np.all(cluster_i == data_i, axis=1))
                # Remove data point from cluster
                cluster_i = np.delete(cluster_i,removeIndex,0)
                c_u_i = computeCentroid(cluster_i)
                clusters[clusterNmbr] = cluster_i
                centroids[clusterNmbr] = c_u_i
                countRemove += 1
            else:
                print("only one: ", countIter)
                countIter += 1
                # sys.exit(0)
            clusterNmbr_winner, d_w_dsit = cluster.vq.vq([data_i], centroids)
            clusterNmbr_winner = clusterNmbr_winner[0]

            if clusterNmbr_winner != clusterNmbr:
                converged = 0

            assignmentDict[tuple(data_i)] = clusterNmbr_winner
            cluster_winner = clusters[clusterNmbr_winner]
            data_i = np.array([data_i])
            np.append(cluster_winner,data_i,axis=0)
            clusters[clusterNmbr_winner] = cluster_winner

            # sys.exit(0)

        print("converged")

        return centroids





if __name__ == '__main__':
    dataPath = '../data/third_project/data-clustering-1.csv'
    data = np.loadtxt(dataPath, dtype=np.object, comments='#', delimiter=',').astype('float32')
    data = data.T

    k = 3

    fig = plt.figure(figsize=(10,10))
    plot1 = fig.add_subplot(221)
    plot2 = fig.add_subplot(222)
    plot3 = fig.add_subplot(223)

    # Lloyd's algorithm
    centroids, distortion = kmeans(data, k_or_guess=k)
    print(centroids)
    # print(centroids)
    '''
    Assigns a code from a code book to each observation. Each observation vector in 
    the ‘M’ by ‘N’ obs array is compared with the centroids in the code book and assigned 
    the code of the closest centroid
    '''
    assignment, cdist = cluster.vq.vq(data, centroids)
    plot1.scatter(data[:, 0], data[:, 1], c=assignment)
    plot1.scatter(centroids[:, 0], centroids[:, 1], c='r')


    # Hartigan's algorithm
    centroidsHartigan = hartiganClustering(k,data)
    print(centroidsHartigan)
    # print(centroidsHartigan)
    # print(centroidsHartigan)
    assignment, cdist = cluster.vq.vq(data, centroidsHartigan)
    plot2.scatter(data[:, 0], data[:, 1], c=assignment)
    plot2.scatter(centroidsHartigan[:, 0], centroidsHartigan[:, 1], c='r')


    # MacQueens's algorithm
    centroidsMacQueen = macQueenClustering(k,data)
    assignment, cdist = cluster.vq.vq(data, centroidsMacQueen)
    plot3.scatter(data[:, 0], data[:, 1], c=assignment)
    plot3.scatter(centroidsMacQueen[:, 0], centroidsMacQueen[:, 1], c='r')

    plt.show()