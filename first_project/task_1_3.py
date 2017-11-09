import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
import sys
import csv

# delta L / delta k
def computeFirstPartialDervK(h, k, alpha):
    N = len(h)

    # compute sum of logs
    sumLogs = 0.
    sum = 0.
    for d in h:
        sumLogs += np.log(d)
        sum += ((d / alpha) ** k) * np.log(d / alpha)

    derivative = N/k - N*np.log(alpha) + sumLogs - sum

    return derivative

# delta L / delta alpha
def computeFirstPartialDervAlpha(h, k, alpha):
    N = len(h)
    sum = 0.

    for d in h:
        sum += (d/alpha)**k

    deriative = k/alpha*(sum-N)
    return deriative

# delta L / delta k * delta k
def computeSecondPartialDervK(h, k, alpha):
    N = len(h)
    sum = 0.

    for d in h:
        sum += ((d / alpha) **k) * (np.log(d / alpha)**2)

    derivative = (-N / k**2) - sum

    return derivative

# delta L / delta alpha * delta alpha
def computeSecondPartialDervAlpha(h, k, alpha):
    N = len(h)
    sum = 0.
    for d in h:
        sum += (d/alpha)**k

    derivative = (k/alpha**2) * (N - (k+1)*sum)

    return derivative

def computePartialDervKAlpha(h, k, alpha):
    N = len(h)

    sum = 0.
    sumTwo = 0.
    for d in h:
        sum+= (d/alpha)**k
        sumTwo += ((d/alpha)**k) * np.log(d/alpha)

    derivative = 1./alpha * sum + (k/alpha) * sumTwo - N/alpha
    return derivative

def computeGradient(h,k,alpha):
    firstDervK = computeFirstPartialDervK(h, alpha, k)
    firstDervAlpha = computeFirstPartialDervAlpha(h, k, alpha)
    gradient = np.array([firstDervK,firstDervAlpha],dtype=float)

    return gradient


def computeHessian(h,k,alpha):
    secondDerivK = computeSecondPartialDervK(h,k,alpha)
    secondDervAlpha = computeFirstPartialDervAlpha(h,k,alpha)
    derivKalpha = computePartialDervKAlpha(h,k,alpha)

    hessian = np.array([[secondDerivK,derivKalpha],[derivKalpha,secondDervAlpha]],dtype=float)

    return hessian

def computeKAndAlpha(h,k,alpha):
    vecT = np.array([k,alpha],dtype=float)
    hessian = computeHessian(h,k,alpha)
    inverseHessian = np.linalg.inv(hessian)
    gradient = computeGradient(h,k,alpha)
    gradient = -1. * gradient

    vecT1 = np.add(vecT,np.matmul(inverseHessian,gradient))

    # print "Hessian dim: ", hessian.shape
    # print "Inverse Hessian dim: ", inverseHessian.shape
    # print "Gradient dim: ", gradient.shape
    # print "inverseHessian*gradient shape: ", (inverseHessian * gradient).shape
    # print "Vect1 dim: ", vecT1.shape

    return vecT1[0],vecT1[1]

if __name__ == "__main__":
    dataPath = '../data/first_project/myspace.csv'
    h = []
    x = []
    dataInterest = []
    with open(dataPath, 'rb') as csvfile:
        lines = csv.reader(csvfile, delimiter=',')
        for line in lines:
            value = int(line[1])
            if value == 0:
                continue
            dataInterest.append(line)

        for n,d in enumerate(dataInterest):
            value = int(d[1])
            h.append(value)
            x.append(n+1)

    # the histogram of the data
    # bins = plt.hist(h)
    #
    # print bins
    # print len(bins)
    #
    # plt.xlabel('bins')
    # plt.ylabel('frequency')
    # plt.title('Histogram of h')
    # # plt.axis([40, 160, 0, 0.03])
    # plt.grid(False)
    # plt.show()


    k = 1.
    alpha =1.
    for i in range(2):
        k,alpha = computeKAndAlpha(h,k=k,alpha=alpha)
        print "Iter: %s  k:%d  alpha:%d" % (i+1,k,alpha)





