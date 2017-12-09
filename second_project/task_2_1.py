
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

def predict(x,w):
    prediction = 0.
    for i, w_i in enumerate(w):
        prediction += (w_i*(x**i))
    return prediction

def leastSquares(Xs,Ys,d):
    X = [[x**(i) for i in range(d+1)] for x in Xs]
    X = np.asarray(X,dtype=float)
    pseudoInverse = np.matmul(inv(np.matmul(np.transpose(X), X)),np.transpose(X))
    w = np.matmul(pseudoInverse,Ys)
    return w

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
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    ax1.set_ylim([-200, 200])
    ax1.scatter([float(d[1]) for d in data], [float(d[0]) for d in data], label='Data')

    for d in Ds:
        w = leastSquares(hs,ws,d=d)
        l = 'Polynomial of degree: '+ str(d)
        ax1.scatter([float(d[1]) for d in data], [predict(float(d[1]), w) for d in data], label=l)

    plt.legend()
    plt.show()

