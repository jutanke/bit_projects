
import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt

'''
This script contains the code for task 1.1 and 1.2
'''
def plotData2D(X, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # see what happens, if you uncomment the next line
    # axs.set_aspect('equal')
    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')

    # set x and y limits of the plotting area
    #a[:,a.min(axis=0)>=0
    xmin = X[0,:].min()
    xmax = X[0,:].max()
    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(-2, X[1,:].max()+10)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        dataOut = '../data/first_project/out/' + filename
        plt.savefig(dataOut, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()




if __name__ == "__main__":
    #######################################################################
    # 1st alternative for reading multi-typed data from a text file
    #######################################################################
    # define type of data to be read and read data from file
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    dataPath = '../data/first_project/whData.dat'
    data = np.loadtxt(dataPath, dtype=dt, comments='#', delimiter=None)

    # read height, weight and gender information into 1D arrays
    ws = np.array([d[0] for d in data])
    hs = np.array([d[1] for d in data])
    gs = np.array([d[2] for d in data])



    ##########################################################################
    # 2nd alternative for reading multi-typed data from a text file
    ##########################################################################
    # read data as 2D array of data type 'object'
    data = np.loadtxt(dataPath,dtype=np.object,comments='#',delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)
    # remove all negative wights
    X_New = []
    for row in X:
        hasNegativeWeight = False
        for col in row:
            if col < 0.:
               hasNegativeWeight = True
        if hasNegativeWeight == False:
            X_New.append(row)

    X_New = np.asarray(X_New,dtype=np.float)
    X = X_New
    # read gender data into 1D array (i.e. into a vector)
    y = data[:,2]
    
    # let's transpose the data matrix 
    X = X.T

    # now, plot weight vs. height using the function defined above
    plotData2D(X, 'plotWH.pdf')

    # next, let's plot height vs. weight 
    # first, copy information rows of X into 1D arrays
    w = np.copy(X[0,:])
    h = np.copy(X[1,:])
    
    # second, create new data matrix Z by stacking h and w
    Z = np.vstack((h,w))

    # third, plot this new representation of the data
    plotData2D(Z, 'plotHW.pdf')

    # fit normal distribution
    '''For the normal distribution, the sample mean ( which is what np.mean() calculates ) is the maximum likelihood
     estimator of the population ( parametric ) mean. This is not true of all distributions, though.
     source: https://glowingpython.blogspot.de/2012/07/distribution-fitting-with-scipy.html
    '''
    mu, std = norm.fit(hs)
    print("Empirical mean: %f " % np.mean(hs))
    print("Empirical sd: %f " % np.std(hs))
    print("Fitted mean: ", mu)
    print("Fitted std: ", std)
    # plot heights into the figure. Points shall lie on x axis -> all y values are 0
    plt.scatter(x=hs,y=[0. for i in range(len(hs))])

    # set x/y-axis limits for plot
    plt.xlim(140,200)
    plt.ylim(0,0.06)

    # create 400 points from the intervall [0,200]
    x = np.linspace(0, 200, 400)
    # get the f(x) for the 400 points
    p = norm.pdf(x, mu, std)

    # plot data points
    plt.plot(x, p, 'k', linewidth=2)
    title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    plt.title(title)
    dataOut = '../data/first_project/out/normal_dist.pdf'
    plt.savefig(dataOut, facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()