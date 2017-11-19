#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def unit_circle_func(x, p):
    """
    Args:
        x: a number of array of numbers
        p: a single non negative numbers
    Output:
        y = f(x) = (1-|x|^p)^(1/p)
    """
    return np.power(1.-np.power(np.abs(x),p), 1./p)


def draw_unit_circle(x,y,p):
    """
    Args:
        x,y : list of x and y coordindates
        p: a non negative number to denote L-p norm
    """
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.set_aspect('equal')
    plt.title('Unit circle for p=%.1f' % p)
    axs.plot(x,y)
    plt.show()


def main():
    """ The unit circle is symmetric about x and y axis
        So it is enough to generate points in range [-1,0]
        and get other points using reflections
    """

    print("Task 1.4:")
    print("    Drawing unit circles")
    # Generate points on x-axis in range [-1,0]
    # This corresponds to top left quadrant
    lo_x = -1
    hi_x  = 0
    delta = 0.00001
    x1 = np.arange(lo_x, hi_x, delta)

    x1_rev = list(reversed(x1))

    # Top right quadrant
    x2 = np.multiply(-1, x1_rev)

    # Bottom right quadrant
    x3 = list(reversed(x2))

    # Bottom left quadrant
    x4 = x1_rev

    for p in [0.5, 1, 2, 3, 4]:
        # Get the Corresponding y-coordinates
        y1 = unit_circle_func(x1, p)
        y2 = list(reversed(y1))
        y3 = np.multiply(-1, list(reversed(y2)))
        y4 = np.multiply(-1, y2)

        x = np.concatenate((x1, x2, x3, x4))
        y = np.concatenate((y1, y2, y3, y4))

        draw_unit_circle(x,y,p)