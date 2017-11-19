#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


# 1st alternative for reading multi-typed data from a text file
def read_data_alt_1(filter_neg=False):
    # define type of data to be read and read data from file
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    data = np.loadtxt('whData.dat', dtype=dt, comments='#', delimiter=None)

    if filter_neg:
        data = np.array(list(filter(lambda x: float(x[0]) > 0 and float(x[1]) > 0, data)))

    # read height, weight and gender information into 1D arrays
    ws = np.array([d[0] for d in data])
    hs = np.array([d[1] for d in data])
    gs = np.array([d[2] for d in data])

    return ws, hs, gs


# 2nd alternative for reading multi-typed data from a text file
def read_data_alt_2(filter_neg=False):
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    if filter_neg:
        data = np.array(list(filter(lambda x: float(x[0]) > 0 and float(x[1]) > 0, data)))

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)

    # read gender data into 1D array (i.e. into a vector)
    y = data[:,2]

    # let's transpose the data matrix
    X = X.T
    return X, y


def plot_data(w1,h1, w2, h2):
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(w1, h1, 'ro')
    ax1.set_title("Plot with outliers (negative values)")
    ax1.set_xlabel('Weight')
    ax1.set_ylabel('Height')

    ax2 = fig.add_subplot(122)
    ax2.plot(w2, h2, 'ro')
    ax2.set_title("Plot with no outliers")
    ax2.set_xlabel('weight')
    ax2.set_ylabel('Height')

    plt.show()


def main():
    print("Task 1.1")
    print("    Import, clean the data and plot height vs weight")

    # Import data, as it is, without filtering negative values
    X, y = read_data_alt_2()

    # Copy information rows of X into 1D arrays
    w1 = np.copy(X[0,:])
    h1 = np.copy(X[1,:])

    # Import the same data as above, but negative values removed
    w2, h2, _ = read_data_alt_1(filter_neg=True)

    # Plot the raw data and clean data side by side
    print("    Plot data with outlier vs with no outlier")
    plot_data(w1, h1, w2, h2)
