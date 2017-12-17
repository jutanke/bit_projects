#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# Reading multi-typed data from a text file
def read_data():
    # define type of data to be read and read data from file
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    data = np.loadtxt('whData.dat', dtype=dt, comments='#', delimiter=None)

    # Task 1.1 ==> Filter out the entry with negative weight or size
    data = np.array(list(filter(lambda x: float(x[0]) > 0 and float(x[1]) > 0, data)))

    # read height, weight and gender information into 1D arrays
    ws = np.array([d[0] for d in data])
    hs = np.array([d[1] for d in data])
    gs = np.array([d[2] for d in data])

    return ws, hs, gs


def get_mean(arr):
    return np.sum(arr) / float(len(arr))


def get_stddev(arr, mean=None):
    if not mean:
        mean = get_mean(arr)
    arr = arr - mean
    arr = arr ** 2
    return np.sqrt(np.sum(arr) / float(len(arr)))


def fit_norm_dist(arr):
    mu = get_mean(arr)
    stddev = get_stddev(arr, mu)
    return mu, stddev


def gaussian_1d(x, mean, stddev):
    index = (x - mean) / stddev
    index_sq = index ** 2
    return (1.0 / (stddev * np.sqrt(2.0*np.pi))) * np.exp(- 0.5 * index_sq)


def plot_norm_fit(arr):
    mean, stddev = fit_norm_dist(arr)
    assert mean == np.mean(arr)
    assert stddev == np.std(arr)

    print("  Empirical mean: %f" % mean)
    print("  Empirical std dev: %f" % stddev)

    # generate (x,y) points from the gaussian function parameterised by mean, stddev
    # Range (mean - 4 * stddev, mean + 4 * stddev) should capture more than 95% of the distribution
    gauss_x = np.arange(mean-4*stddev, mean+4*stddev, 0.1)
    gauss_y = gaussian_1d(gauss_x, mean, stddev)

    fig = plt.figure()
    axs = fig.add_subplot(111)

    # plot the original data points on the x-axis
    axs.plot(arr, np.zeros_like(arr), 'bo', label='data', alpha=0.3)

    # plot the points of the gaussian function
    axs.plot(gauss_x, gauss_y, 'r-', label='normal')

    xmin = gauss_x.min()
    xmax = gauss_x.max()
    axs.set_xlim(xmin, xmax)
    axs.set_ylim(0, gauss_y.max()+0.001)
    leg = axs.legend(loc='upper right', shadow=False, fancybox=False, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    plt.show()
    plt.close()


def main():
    print("Task 1.2")
    print("    Fitting a normal distribution to 1D data")
    ws, hs, gs = read_data()
    plot_norm_fit(hs)
