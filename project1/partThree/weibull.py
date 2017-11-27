import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import norm

def normalize(x):
    return (x-min(x))/(max(x)-min(x))

if __name__ == "__main__":

    # read data as 2D array of data type 'object'
    data = np.loadtxt("myspace.csv", delimiter=",", dtype={'names': ('dates', 'value'),'formats': ('|S30', np.float)})
    print "Task 1.3"

    # removing rows with 0 as value. 
    positive_data = list()
    for row in data:
        if float(row[1]) > 0.0 :
            positive_data.append(row.tolist())
            pass
        pass
    data = np.array(positive_data)

    h = data[:,1].astype(np.float)
    x = np.array(range(0, h.shape[0]))

    # print h.shape[0] # 453

    kappa = 4
    alpha = 2
    N = h.shape[0]

    for itrr in xrange(0,10):
        
        # 1) L/K
        dLikelihood_dKappa   = (N/kappa) - N*np.log(alpha) + np.sum(np.log(h)) + np.sum( np.power( (h/alpha),kappa ) *  np.log(h/alpha) )

        # 2) L/A
        dLikelihood_dAlpha   = (kappa/alpha)* np.sum(np.power((h/alpha),kappa)-N)

        # 3) 2L/K2
        d2Likelihood_dKappa2 = -(N/np.power(kappa,2)) * np.sum( np.power( (h/alpha),kappa ) * np.power( np.log(h/alpha),2) )

        # 4) 2L/A2
        d2Likelihood_dAlpha2 = (kappa/np.power(alpha,2)) * ( N - (kappa+1)* np.sum(np.power((h/alpha),kappa)))

        # 5) 2L/KA
        d2Likelihood_dKappaAlpha = (1/alpha)*np.sum(np.power((h/alpha),kappa)) + (kappa/alpha)*np.sum(np.power((h/alpha),kappa)*np.log(h/alpha))-(N/alpha)
        
        # kappa_new, alpha_new = A + (B^-1) * C

        # A = [kappa, alpha]^T
        A = np.array([kappa, alpha])

        # B =   [ d2Likelihood_dKappa2       d2Likelihood_dKappaAlpha    ]
        #       [ d2Likelihood_dKappaAlpha   d2Likelihood_dAlpha2        ]
        B = np.array([[d2Likelihood_dKappa2, d2Likelihood_dKappaAlpha], [d2Likelihood_dKappaAlpha, d2Likelihood_dAlpha2]])

        # C = [-dLikelihood_dKappa, -dLikelihood_dAlpha]^T
        C = np.array([-dLikelihood_dKappa, -dLikelihood_dAlpha])

        # kappa_new, alpha_new
        new_vals = A.T + np.dot(np.linalg.inv(B),C.T)
        kappa = new_vals[0]
        alpha = new_vals[1]

        pass

    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # the weibull distributeion
    pdf = (kappa/alpha) * np.power((x/alpha),(kappa-1)) * np.exp(- np.power((x/alpha), kappa))

    # plot pdf
    axs.plot(x, pdf, 'r', label="weibull")

    # plot histogram
    axs.hist(h, 50, color = "skyblue", edgecolor='black', normed=True)

    plt.title = "Task 1.3: Weibull pdf and google data histogram"

    plt.show()
    plt.close()