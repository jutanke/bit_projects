import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt

import csv

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
    bins = plt.hist(h)

    print bins
    print len(bins)

    plt.xlabel('bins')
    plt.ylabel('frequency')
    plt.title('Histogram of h')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(False)
    plt.show()




