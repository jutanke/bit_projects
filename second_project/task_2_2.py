
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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

    X = np.transpose(np.array([ws,hs],dtype=float))
    print X.shape

    mean = np.mean(X, axis=0)
    print mean
    cov = np.cov(X, rowvar=0)
    print cov
    heights = np.linspace(100, 200,100)
    weights= np.linspace(50, 120, 100)
    samples = np.transpose(np.array([heights, weights]))
    print samples
    pdf = multivariate_normal.pdf(samples,mean=mean,cov=cov)
    # drawnSamples = np.random.multivariate_normal(mean=mean,cov=cov)
    plt.plot(samples, pdf, 'k', linewidth=2)
    plt.show()