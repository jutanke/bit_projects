import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm

def plotHeighDataAndGaussian(X, mean, std, plot_gaussian=None, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # see what happens, if you uncomment the next line
    #axs.set_aspect('equal')
    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')
    
    # add a gaussian plot if flag is true
    if plot_gaussian == True:
	    x = np.linspace(mean[0] - 3*std[0],  200, mean[0] + 3*std[0])
	    p = norm.pdf(x, mean[0], std[0])
	    axs.plot(x,p, label='gaussian')

    # set x and y limits of the plotting area
    xmin = X[0,:].min()
    xmax = X[0,:].max()

    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(0, 0.1)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    plt.title(filename)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == "__main__":

    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    # removing outliers. i.e. the rows with negative values of weight and size.
    positive_data = list()
    for row in data:
        if float(row[0]) >= 0.0 and float(row[1]) >= 0.0:
            positive_data.append(row.tolist())
            pass
        pass
    data = np.array(positive_data)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)

    # getting the data such that height is in one column, and 0 in second column for 2D plot.
    for row in X:
        row[0] = row[1]
        row[1] = 0
        pass

    # calculating the mean and standard deviation of the height (and the zero column)
    meann = np.mean(X, axis=0)
    stdd  = np.std(X, axis=0)

    # read gender data into 1D array (i.e. into a vector)
    y = data[:,2]

    # let's transpose the data matrix 
    X = X.T

    # now, plot weight vs. height using the function defined above
    plotHeighDataAndGaussian(X, meann, stdd, True, "gaussianAndHeights.pdf")
