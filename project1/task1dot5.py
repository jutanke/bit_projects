#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:24:03 2017

@author: utkrist
"""


# ERRRRRRRR
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as msc
import scipy.ndimage as img


def foreground2BinImg(f):
    d = img.filters.gaussian_filter(f, sigma=0.50, mode='reflect') - \
        img.filters.gaussian_filter(f, sigma=1.00, mode='reflect')
    d = np.abs(d)
    m = d.max()
    d[d< 0.1*m] = 0
    d[d>=0.1*m] = 1
    return img.morphology.binary_closing(d)


def countForegroundPixels(img):
    """
    img: Binary image containing "True" for foreground pixels,
         and "False" for background pixels
    """
    h, w = img.shape
    assert h==w
    L = int(np.log2(w))

    scale_factors = []
    num_boxes = []
    for i in range(1, L-2):

        # Scaling Factor
        si = 1./2.**i

        # Height and width of the box
        wi = int(si * w)
        hi = wi

        # Top left corner of a box
        px = 0

        # Number of boxes containing foreground pixel
        ni = 0
        while px + wi <= w:
            py = 0
            while py + hi <= h:
                if True in img[px:px+wi, py:py+hi]:
                    ni += 1
                py = py + hi
            px = px + wi
        scale_factors.append(si)
        num_boxes.append(ni)

    return scale_factors, num_boxes


def get_fractal_dim(imgname):
    # Read image
    file = msc.imread(imgname+'.png', flatten=True).astype(np.float)

    # Binarize Image
    img = foreground2BinImg(file)

    # Divide the image into boxes and count the ones containing foregroud pixels
    scale_factors, num_boxes = countForegroundPixels(img)
    print("    Num boxes: " + str(num_boxes))


    x = np.log2(1.0/np.array(scale_factors))
    y = np.log(num_boxes)

    # Perform least square fit for the equation
    # D * log(1/s_i) + b = log(n_i)
    # D * x + b = y
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    X = np.concatenate((x, np.ones((x.shape[0],1))), axis=1)

    # Least square fitting
    # param = (X^T * X)^(-1) * X^T * y
    # param[0] = D, param[1] = b
    param = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

    # Plot the line of best fit along with the data
    _x = np.arange(0, max(x),0.001)
    _y = list(map(lambda xi: param[0]*xi+param[1],_x))

    fig, ax = plt.subplots()
    plt.scatter(x,y, color='red')
    plt.plot(_x, _y, color='blue')
    plt.legend(('Line of best fit', 'Data points'))
    plt.xlabel('log(1/si)')
    plt.ylabel('log(ni)')
    plt.title('log(1/si) vs log(ni) for %s' % imgname)
    plt.show()

    return param[0]


def main():
    print("Task 1.5")
    print("  Estimating the dimension of fractal objects in an image")
    
    img1 = 'lightning-3'
    img2 = 'tree-2'

    print("  Image: %s" % img1)
    fdim1 = get_fractal_dim(img1)
    print("    Fractal dimension: %f" % fdim1)


    print("  Image: %s" % img2)
    fdim2 = get_fractal_dim(img2)
    print("    Fractal dimension: %f" % fdim2)

