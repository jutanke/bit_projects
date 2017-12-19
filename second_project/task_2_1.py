
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import sys

def predict(x,w):
    x_vec = np.array([x ** i for i in range(len(w))])
    prediction = np.matmul(w, np.transpose(x_vec))
    return prediction

def leastSquares(X_design,Ys):

    X_T_X = np.matmul(np.transpose(X_design), X_design)
    inverse = np.linalg.inv(X_T_X)
    theta_MLE = np.matmul(np.matmul(inverse, np.transpose(X_design)), Ys)

    return theta_MLE

if __name__ == '__main__':
    dataPath = '../data/first_project/whData.dat'
    data = np.loadtxt(dataPath, dtype=np.object, comments='#', delimiter=None)
    ws = []
    hs = []
    gs = []

    for d in data:
        ws.append(d[0])
        hs.append(d[1])
        gs.append(d[2])

    ws = np.array(ws,dtype=float)
    hs = np.array(hs,dtype=float)
    gs = np.array(gs)

    wIndex = ((ws > 0) * 1).nonzero()
    ws = ws[wIndex]
    hs = hs[wIndex]
    gs = gs[wIndex]

    Ds = [1,5,10]
    # fig = plt.figure(figsize=(8, 8))
    # ax1 = fig.add_subplot(111)
    # ax1.set_ylim([-200, 200])
    # ax1.scatter([float(d[1]) for d in data], [float(d[0]) for d in data], label='Data')
    #
    # for d in Ds:
    #     w = leastSquares(hs,ws,d=d)
    #     l = 'Polynomial of degree: '+ str(d)
    #     ax1.scatter([float(d[1]) for d in data], [predict(float(d[1]), w) for d in data], label=l)
    #
    # plt.legend()
    # plt.show()

    d = 5
    X_design = np.array([[x ** (i) for i in range(d + 1)] for x in hs])
    theta_MLE = leastSquares(X_design=X_design,Ys=ws)
    for x_i in data:
        height = float(x_i[1])
        prediction = predict(height,theta_MLE)
        print("height: %f  predicted weight: %f" % (height,prediction))

