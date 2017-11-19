#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:09:40 2017

@author: utkrist
"""

#  Incomplete

def weibull_density(x,k,a):
    xa = np.divide(x,a)
    xak_1 = np.power(xa, k-1)
    exk = np.exp(-np.multiply(xak_1, xa))
    return (k/a) * np.multiply(xak_1, exk)

def weilbul_fit():
    data = np.loadtxt('myspace.csv',dtype=np.object,comments='#',delimiter=",")

    # read interest data into 1D array
    interest = data[:,1].astype(np.float)

    # filter the zero entries
    h = np.array(list(filter(lambda x: x != 0 , interest)))

    # Get the values for the x axis
    x = np.arange(1., len(h)+1)/(len(h)/2.)

    weib_y = weibull_density(x,k=5,a=1)
    scale = max(h) / max(weib_y)

    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # plot the original data points on the x-axis
    axs.plot(x, h, color='black', label='data', alpha=0.3)
    axs.plot(x, weib_y * scale, 'r-', label='weibul')
    plt.show()