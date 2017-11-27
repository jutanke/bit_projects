# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plotData2D(X, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)
    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')

    # set x and y limits of the plotting area
    xmin = X[0,:].min()
    xmax = X[0,:].max()
    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(-2, X[1,:].max()+10)
    axs.set_ylabel('Height')
    axs.set_xlabel('Weight')
    

    # set properties of the legend of the plot
    leg = axs.legend(loc='lower left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()
    

if __name__ == "__main__":
    #######################################################################
    # 1st alternative for reading multi-typed data from a text file
    #######################################################################
    # define type of data to be read and read data from file
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    data = np.loadtxt('/Users/POOJA/Desktop/Pattern Recognition/whData.dat', dtype=dt, comments='#', delimiter=None)

    # read height, weight and gender information into 1D arrays
    ws = np.array([d[0] for d in data])
    hs = np.array([d[1] for d in data])
    gs = np.array([d[2] for d in data]) 


    ##########################################################################
    # 2nd alternative for reading multi-typed data from a text file
    ##########################################################################
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)

    # read gender data into 1D array (i.e. into a vector)
    y = data[:,2]
    
    # let's transpose the data matrix 
    X = X.T

    # now, plot weight vs. height using the function defined above
    plotData2D(X,'plotWH.pdf')

    #######################################################################
    # Plot the data without the outliers (Plot only positive weight entries)
    #######################################################################                   
    weight=X[0,:]
    height=X[1,:]
    gender=y
    i=1
    l = []
    for i in zip((weight>0)*1,(height>0)*1):
        l.append(i[0]and i[1])
    WH_mask=np.array(l).nonzero()
    weight=weight[WH_mask]
    height=height[WH_mask]
    gender=gender[WH_mask]
    combined = np.vstack((weight,height))
    plotData2D(combined,'Withoutoutliers_plotWH.pdf')
   
