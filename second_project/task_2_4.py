import itertools
from itertools import chain, combinations
import numpy as np


def createFeatureVec(S):
    pSet = powerset(S)
    pSet = [[1] if not list(subSet) else list(subSet) for subSet in pSet]
    featureVec = []

    for subSet in pSet:
        val = 1
        for x in subSet:
            val *= x
        featureVec.append(val)

    return featureVec

'''https://stackoverflow.com/questions/374626/how-can-i-find-all-the-subsets-of-a-set-with-exactly-n-elements'''
def powerset(iterable):
    xs = list(iterable)
    # note we return an iterator rather than a list
    return chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1))

def predict(X_design, theta):
    prediction = np.matmul(X_design, theta)
    return prediction


def computeThetaMLE(X_design, Y):
    X_T_X = np.matmul(np.transpose(X_design),X_design)
    inverse = np.linalg.inv(X_T_X)
    pseudoInverse = np.matmul(inverse,np.transpose(X_design))
    theta_MLE = np.matmul(pseudoInverse,Y)

    return theta_MLE

if __name__ == '__main__':
    n = 3
    tupleList = reversed(list(itertools.product([0, 1], repeat=n)))
    X_design = [list(tuple) for tuple in tupleList]
    X_design =  np.array([[1. if float(x) == 1 else -1. for x in subList] for subList in X_design])

    rule = 110
    targetString = np.binary_repr(rule, width=2**3)
    Y = np.array([float(x) for x in targetString])

    theta_MLE_110 = computeThetaMLE(X_design=X_design,Y=Y)

    prediction_110 = predict(X_design,theta_MLE_110)

    print "theta_MLE_110: ", theta_MLE_110
    print "prediction_110: ", prediction_110

    print

    rule = 126
    targetString = np.binary_repr(rule, width=2 ** 3)
    Y = np.array([float(x) for x in targetString])

    theta_MLE_126 = computeThetaMLE(X_design=X_design, Y=Y)

    prediction_126 = predict(X_design, theta_MLE_126)

    print "theta_MLE_126: ", theta_MLE_126
    print "prediction_126: ", prediction_126

    print '-------------------------'


    # Compute theta based on feature based design matrix
    n = 3
    S = [i +1 for i in range(n)]
    tupleList = reversed(list(itertools.product([0, 1], repeat=n)))
    X_design = [list(tuple) for tuple in tupleList]
    X_design = np.array([[1. if float(x) == 1 else -1. for x in subList] for subList in X_design])
    feature_X_Design = np.array([createFeatureVec(S) for S in X_design])

    rule = 110
    targetString = np.binary_repr(rule, width=2 ** 3)
    Y = np.array([float(x) for x in targetString])

    theta_feature_MLE_110 = computeThetaMLE(X_design=feature_X_Design, Y=Y)

    prediction_feature_110 = predict(feature_X_Design, theta_feature_MLE_110)

    print "theta_feature_MLE_110: ", theta_feature_MLE_110
    print "prediction_feature_110: ", prediction_feature_110

    print

    rule = 126
    targetString = np.binary_repr(rule, width=2 ** 3)
    Y = np.array([float(x) for x in targetString])

    theta_feature_MLE_126 = computeThetaMLE(X_design=feature_X_Design, Y=Y)

    prediction_feature_126 = predict(feature_X_Design, theta_feature_MLE_126)

    print "theta_feature_MLE_126: ", theta_feature_MLE_126
    print "prediction_feature_126: ", prediction_feature_126





