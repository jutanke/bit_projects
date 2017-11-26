
import numpy as np
import scipy.misc as msc
import scipy.ndimage as img
import matplotlib.pyplot as plt
from scipy.stats import linregress
# from scipy.ndimage import convolve
from scipy.signal import convolve2d

def computeDataPoints(binaryImg):
    S = [1. / (2. ** (i + 1)) for i in range(9 - 2)]
    logNs = []
    logS = []

    for s in S:
        n = countBoxes(binaryImg, s)
        logNs.append(np.log(n))
        logS.append(np.log(1. / s))
        # print "log(n):%f log(1/s):%f" % (np.log(n), np.log(1. / s))

    return logS,logNs


# def countBoxes(binaryImg, scaleFactor):
#     a = 512 * scaleFactor
#     # Problem: No paramteter that defines how to slide over image
#     # kernel = np.ones(shape=(a,a))
#     # convolve2d(binaryImg, kernel)

# Julians solution
def countBoxes(g, s_i):
    h, w = g.shape
    assert (h == w)
    sub_w = int(w * s_i)
    times_w = int(w / sub_w)

    n = 0
    for _x in range(0, times_w):
        for _y in range(0, times_w):
            # Define subrows
            x = _x * sub_w
            # Define sub-columns
            y = _y * sub_w
            # Extract sub-image
            I = g[y:y + sub_w, x:x + sub_w]
            count = np.sum(I)
            if count > 0:
                n += 1
    return n

def foreground2BinImg(f):
    # Apply Gaussian filter
    d = img.filters.gaussian_filter(f, sigma=0.5, mode='reflect') - \
        img.filters.gaussian_filter(f, sigma=1, mode='reflect')

    # Compute absolute values
    d = np.abs(d)
    # Get biggest value
    m = d.max()

    # d<0.1*m: Determine all background pixels
    d[d<0.1*m] = 0
    # d>= 0.1 * m: Determine all foreground pixels
    d[d>= 0.1 * m] = 1

    # Transform into bool values
    return img.morphology.binary_closing(d)

imgPath = '../data/first_project/lightning-3.png'
lightening = msc.imread(imgPath, flatten=True).astype(np.float)
light_trans = foreground2BinImg(lightening)

imgPath = '../data/first_project/tree-2.png'
tree = msc.imread(imgPath, flatten=True).astype(np.float)
tree_trans = foreground2BinImg(tree)

# Fit line
xLight,yLight = computeDataPoints(light_trans)
slopeLight, interceptLight, _, _, _ = linregress(x=xLight, y=yLight)
estimatedYLight = [slopeLight * x + interceptLight for x in xLight]

xTree,yTree = computeDataPoints(tree_trans)
slopeTree, interceptTree, _, _, _ = linregress(x=xTree, y=yTree)
estimatedYTree = [slopeTree * x + interceptTree for x in xTree]

fig = plt.figure(figsize=(5,5))
plot = fig.add_subplot(111)
plot.set_xlabel('log(1/s)')
plot.set_ylabel('log(n)')
plot.scatter(xLight, estimatedYLight, label='lightning')
plot.scatter(xTree, estimatedYTree, label='tree')
plt.legend()

fig2 = plt.figure(figsize=(10,10))
fig2.add_subplot(221).imshow(lightening)
fig2.add_subplot(222).imshow(light_trans)
fig2.add_subplot(223).imshow(tree)
fig2.add_subplot(224).imshow(tree_trans)
plt.show()

