
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.misc as msc
import scipy.ndimage as img

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


def unit_circle_func(x, p):
    return np.power(1.-np.power(np.abs(x),p), 1./p)

def draw_unit_circles():

    # Generate x-axis points in range [-1,0]
    # This corresponds to top left quadrant
    x1 = np.arange(-1, 0, 0.00001)

    x1_rev = list(reversed(x1))
    # Rest of the x-axis points can be derived from symmetry
    x2 = np.multiply(-1, x1_rev)
    x3 = list(reversed(x2))
    x4 = x1_rev

    for p in [0.5, 1, 2, 3, 4]:
        # Get the Corresponding y-coordinates
        y1 = unit_circle_func(x1, p)

        # Since the unit circle symmetric about x and y axis
        # we simply generate rest of the points by reflecction
        y2 = list(reversed(y1))
        y3 = np.multiply(-1, list(reversed(y2)))
        y4 = np.multiply(-1, y2)

        x = np.concatenate((x1, x2, x3, x4))
        y = np.concatenate((y1, y2, y3, y4))

        fig = plt.figure()
        axs = fig.add_subplot(111)
        axs.set_aspect('equal')
        axs.plot(x,y)
        plt.show()


def foreground2BinImg(f):
    d = img.filters.gaussian_filter(f, sigma=0.50, mode='reflect') - \
        img.filters.gaussian_filter(f, sigma=1.00, mode='reflect')
    d = np.abs(d)
    m = d.max()
    d[d< 0.1*m] = 0
    d[d>=0.1*m] = 1
    return img.morphology.binary_closing(d)

if __name__ == "__main__":
    # Contains: task 1.1
    #ws, hs, gs = read_data_alt_1()
    #X, y = read_data_alt_2()

    # demo_task(X,y)

    # Task 1.2 Fitting normal distribution to body size data:
    #plot_norm_fit(hs)

    # Task 1.3 Fitting a Weibull distribution to 1D data
    #weilbul_fit()

    #Task 1.4 Unit circle
    #draw_unit_circles()

    # Task 1.5 Fractal Dimension
    imgName = 'tree-'
    f = msc.imread(imgName+'.png', flatten=True).astype(np.float)
    g = foreground2BinImg(f)
    #g = g * 1

    #msc.imshow(g*1)
    log_si = []
    log_ni = []

    L = 9
    h = w = 512
    for i in range(1, L-1):
        si = 1./2.**i
        wi = int(si * w)
        hi = wi

        r = 0
        c = 0
        ni = 0
        while r + wi <= w and c + hi <= h:
            if True in g[r:r+wi+1, c:c+hi+1]:
                ni += 1
            r = r+wi
            c = c+hi

        log_si.append(np.log(1.0/si))
        log_ni.append(np.log(ni))

    y = np.expand_dims(log_si, axis=1)
    x = np.expand_dims(log_ni, axis=1)

    X = np.concatenate((x, np.ones((x.shape[0],1))), axis=1)

    # Least square fitting
    param = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
    print(param)

    _x = np.arange(0,3,0.001)
    _y = list(map(lambda xi: param[0]*xi+param[1],_x))

    plt.figure()
    plt.plot(x,y)
    plt.plot(_x, _y)
    plt.show()

    #param =  np.dot(np.linalg.inv(np.dot(X.T,X)), X)
    #np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X), y)
    #print(param)
"""
    m = np.random.rand()
    c = np.random.rand()
    n = float(len(log_si))
    learning_rate = 0.0001

    errors = []
    err = 500
    max_iter = 1000
    while err > 0.3 and i < max_iter:
        e = y - (m*x+c)
        err = (1.0/n * np.dot(e.T, e))[0][0]
        errors.append(err)

        d_m = -2./n * np.sum(np.multiply(err, x))
        d_c = -2./n * np.sum(err)

        m -= learning_rate * d_m
        c -= learning_rate * d_c
        i +=1

    plt.plot(range(len(errors)), errors)
    plt.show()
    print(m,c)
"""





