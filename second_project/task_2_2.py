
import numpy as np

if __name__ == '__main__':

    # Load data
    dataPath = '../data/first_project/whData.dat'
    data = np.loadtxt(dataPath, dtype=np.object, comments='#', delimiter=None)
    ws = data[:,0]
    hs = data[:,1]
    gs = data[:,2]

    ws = np.array(ws, dtype=float)
    hs = np.array(hs, dtype=float)
    gs = np.array(gs)

    # Handle outliers
    wIndex = ((ws > 0) * 1).nonzero()
    wIndexOutliers = ((ws < 0) * 1).nonzero()


    hsOut = hs[wIndexOutliers]
    ws = ws[wIndex]
    hs = hs[wIndex]
    gs = gs[wIndex]


    # Shape = (#variables,#values)
    X = np.array([ws,hs],dtype=float)

    # Compute parameters of bivariate Gaussian
    meanWeight = np.mean(ws)
    meanHeight = np.mean(hs)
    sdWeight = np.sqrt(np.var(ws,ddof=1))
    sdHeight = np.sqrt(np.var(hs,ddof=1))
    cov_h_w = np.cov(X,bias=False)[0,1]
    pearsonCor = cov_h_w/(sdHeight*sdWeight)


    # Predict weights for outliers
    print('-------Unbiased-------')
    for hOut in hsOut:
        predWeight = meanWeight + pearsonCor*(sdWeight/sdHeight)*(hOut-meanHeight)
        print("height of outlier: %f  predicted weight: %f" % (hOut, predWeight))


    # Second Version: SDs are biased
    meanWeight = np.mean(ws)
    meanHeight = np.mean(hs)
    sdWeight = np.std(ws)
    sdHeight = np.mean(hs)
    cov_h_w = np.cov(X, bias=True)[0, 1]
    pearsonCor = cov_h_w / (sdHeight * sdWeight)

    print()
    print('------------------')
    for hOut in hsOut:
        predWeight = meanWeight + pearsonCor*(sdWeight/sdHeight)*(hOut-meanHeight)
        print("height of outlier: %f  predicted weight: %f" % (hOut, predWeight))


