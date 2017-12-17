
import numpy as np
import matplotlib.pyplot as plt
import timeit

def k_nearest_neighbors(X, k, x_new):
    argMinSorted = np.argsort(np.sum((X-x_new)**2,axis=1),axis=0)
    return argMinSorted[:k]


if __name__ == '__main__':
    trainDataPath = '../data/second_project/data2-train.dat'
    testDataPath = '../data/second_project/data2-test.dat'
    trainData = np.loadtxt(trainDataPath, dtype=float, delimiter=' ')
    testData = np.loadtxt(testDataPath, dtype=float, delimiter=' ')

    X = trainData[:, :2]
    Y = trainData[:,2]

    X_Test = testData[:, :2]
    Y_Test = testData[:,2]

    K = [1,3,5]
    accuracies = []

    for k in K:
        start = timeit.default_timer()
        predictions = []
        for x_T in X_Test:
            kNearestNeighbors = k_nearest_neighbors(X, k, x_T)
            predictions.append(1. if np.sum(Y[kNearestNeighbors],axis=0)>0 else -1.)

        accuracy = np.sum(
            np.array([1. if pred == Y_Test[i] else 0. for i, pred in enumerate(predictions)]), axis=0) / float(
            len(Y_Test))
        stop = timeit.default_timer()
        accuracies.append(accuracy)
        print "k: %d  accuracy: %f  runtime: %f" % (k,accuracy,stop - start)

    plt.bar(K, accuracies, width=0.4, color='r')
    plt.ylabel('Accuracy')
    plt.title('k Nearest Neighbors')
    plt.ylim(0,1.)
    plt.show()

