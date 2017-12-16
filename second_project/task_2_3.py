
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def predict(x,w):
    x_vec = np.array([x**i for i in range(len(w))])
    prediction = np.matmul(w,np.transpose(x_vec))

    return prediction

def bayesianRegression(X_design, y, sigmaSquare, sigma_0_square):
    X_T_X = np.matmul(np.transpose(X_design),X_design)
    regulariser = sigmaSquare/sigma_0_square
    I_regularised = regulariser*np.identity(X_T_X.shape[0])
    inverse = np.linalg.inv(np.add(X_T_X,I_regularised))
    theta_MAP = np.matmul(np.matmul(inverse,np.transpose(X_design)),y)

    return theta_MAP

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

    ws = np.array(ws, dtype=float)
    hs = np.array(hs, dtype=float)
    gs = np.array(gs)

    wIndex = ((ws > 0) * 1).nonzero()
    ws = ws[wIndex]
    hs = hs[wIndex]
    gs = gs[wIndex]

    d = 5
    X_design = np.array([[x ** (i) for i in range(d + 1)] for x in hs])
    sigma_0_square = 3.
    sigmaSquare = 1.

    theta_MAP = bayesianRegression(X_design=X_design, y=ws, sigmaSquare=sigmaSquare, sigma_0_square=sigma_0_square)
    print "theta_MAP: ", theta_MAP

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([-200, 200])
    # ax1.scatter([float(d[1]) for d in data], [float(d[0]) for d in data], label='Data')

    for x_i in data:
        height = float(x_i[1])
        prediction = predict(height,theta_MAP)
        print "height: %f  predicted weight: %f" % (height,prediction)
