
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def plotData2D(X, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # see what happens, if you uncomment the next line
    # axs.set_aspect('equal')
    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')

    # set x and y limits of the plotting area
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
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
# 1st alternative for reading multi-typed data from a text file
def read_data_alt_1():
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


# 2nd alternative for reading multi-typed data from a text file
def read_data_alt_2():
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat',dtype=np.object,comments='#',delimiter=None)

    #-------------------------------------------------------------------------
    # Task 1.1 ==> Filter out the entry with negative weight or size
    #-------------------------------------------------------------------------
    data = np.array(list(filter(lambda x: float(x[0]) > 0 and float(x[1]) > 0, data)))

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)

    # read gender data into 1D array (i.e. into a vector)
    y = data[:,2]
    
    # let's transpose the data matrix 
    X = X.T
    return X, y


def demo_task(X, y):
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


def assert_valid_array(arr):
    assert isinstance(arr, np.ndarray), "Provide an array of real numbers"
    assert len(arr) > 0, "Provide a non-empty array of numbers."
    
def get_mean(arr):
    assert_valid_array(arr)
    return np.sum(arr) / float(len(arr))

def get_stddev(arr, mean=None):
    assert_valid_array(arr)    
    if not mean:
        mean = get_mean(arr)
    arr = arr - mean
    arr = arr ** 2
    return np.sqrt(np.sum(arr) / float(len(arr)))
    
def gaussian_1d(x, mean, stddev):
    index = (x - mean) / stddev
    index_sq = index ** 2
    return (1.0 / (stddev * np.sqrt(2.0*np.pi))) * np.exp(- 0.5 * index_sq)

def fit_norm_dist(arr):
    assert_valid_array(arr)
    mu = get_mean(arr)
    stddev = get_stddev(arr, mu)
    return mu, stddev

def plot_norm_fit(arr):
    assert_valid_array(arr)
    mean, stddev = fit_norm_dist(arr)
    
    # generate (x,y) points from the gaussian function parameterised by mean, stddev
    # Range (mean - 4 * stddev, mean + 4 * stddev) should capture more than 95% of the distribution
    gauss_x = np.arange(mean-4*stddev, mean+4*stddev, 0.1)
    gauss_y = gaussian_1d(gauss_x, mean, stddev)
    
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)
    
    # plot the original data points on the x-axis 
    axs.plot(arr, np.zeros_like(arr), 'bo', label='data', alpha=0.3)
    
    # plot the points of the gaussian function 
    axs.plot(gauss_x, gauss_y, 'r-', label='normal')
    
    # set x and y limits of the plotting area
    xmin = gauss_x.min()
    xmax = gauss_x.max()
    axs.set_xlim(xmin, xmax)
    axs.set_ylim(0, gauss_y.max()+0.001)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper right', shadow=False, fancybox=False, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # either show figure on screen or write it to disk
    plt.show()
    plt.close()
   

if __name__ == "__main__":
    # Contains: task 1.1
    ws, hs, gs = read_data_alt_1()
    X, y = read_data_alt_2()
    
    # demo_task(X,y)

    # Task 1.2 Fitting normal distribution to body size data:
    plot_norm_fit(hs)

    # Task 1.3 Fitting a Weibull distribution to 1D data
   

    

