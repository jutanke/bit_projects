import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lnalg

if __name__ == "__main__":
    # read data as 2D array of data type 'object'
    data = np.loadtxt('../whData.dat',dtype=np.object,comments='#',delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    X = data[:,0:2].astype(np.float)

    w = np.copy(X[:,0])
    h = np.copy(X[:,1])
    g = data[:,2]

    weight_greater_than_zero = np.where( w > 0 )
    weight_less_than_zero = np.where( w < 0 )

    h_n = h[weight_less_than_zero] # height vals where weight is -1
    w   = w[weight_greater_than_zero]
    h   = h[weight_greater_than_zero]
    g   = g[weight_greater_than_zero]

    maxh = np.amax(h)
    minh = np.amin(h)
    x = np.copy(h)/maxh
    y = np.array([w]).T/maxh
    h_n = h_n/maxh
    # y = np.copy([w]).T

    print x
    print y

    degrees = [1,5,10]
    for d in degrees:
        # z = np.polyfit(x, y, 3)
        A = np.vander(x, d+1)
        coeffs = np.squeeze(np.dot(lnalg.pinv(A), y))
        f = np.poly1d(coeffs)
        y_est = f(np.linspace( 0, 1, 10000))

        print "d = %f" % d
        for h_n_idx in h_n:
            print "  h = %f -> w = %f" % (h_n_idx*maxh,f(h_n_idx)*maxh)
            pass

        plt.plot(h_n*maxh, f(h_n)*maxh, "o")
        # create plot
        plt.plot(x*maxh, y*maxh, '.', label = 'original data', markersize=5)
        plt.plot(np.linspace( 0, 1, 10000)*maxh, y_est*maxh, 'o-', label = 'estimate', markersize=1)

        axes = plt.gca()
        axes.set_xlim([150,185])
        axes.set_ylim([30,100])
        plt.xlabel('height')
        plt.ylabel('weight')
        plt.title('least squares fit of degree '+ str(d))
        plt.savefig('fit_degree_' + str(d) + '.png')
        plt.close()
        pass