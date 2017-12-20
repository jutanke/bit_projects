
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import sys



def predict(X_design, theta):
    predictions = np.matmul(X_design,theta)
    return predictions

def leastSquaresUnstable(X_design, Y):
    X_T_X = np.matmul(X_design.T, X_design)
    inverse = np.linalg.inv(X_T_X)
    theta_MLE = np.matmul(np.matmul(inverse, X_design.T), Y)

    return theta_MLE

def leastSquaresStable(X_design, Y):
    theta_MLE = np.matmul(np.linalg.pinv(X_design), Y)

    return theta_MLE

def commputeDesignX(X,d):
    X_design = np.array([[x ** (i) for i in range(d + 1)] for x in X])

    return X_design

# def leastSquares(X_design,Ys):
#
#     X_T_X = np.matmul(np.transpose(X_design), X_design)
#     inverse = np.linalg.inv(X_T_X)
#     theta_MLE = np.matmul(np.matmul(inverse, np.transpose(X_design)), Ys)
#
#     return theta_MLE

if __name__ == '__main__':
    dataPath = '../data/first_project/whData.dat'
    data = np.loadtxt(dataPath, dtype=np.object, comments='#', delimiter=None)
    ws = data[:, 0].astype('int32')
    hs = data[:, 1].astype('int32')
    gs = data[:, 2]

    wsAll = np.array(ws, dtype=float)
    hsAll = np.array(hs, dtype=float)
    gsAll = np.array(gs)

    # Remove outliers
    wIndex = ((ws > 0) * 1).nonzero()
    wIndexOutliers = ((ws < 0) * 1).nonzero()

    ws = wsAll[wIndex]
    hs = hsAll[wIndex]
    gs = gsAll[wIndex]

    hsOut = hs[wIndexOutliers]

    Ds = [1,5,10]
    colors = ['r--','g--','y--']
    colorsScatter = ['r', 'g', 'y']
    xs = np.linspace(150, 200, 1000)
    plt.figure(figsize=(10,10))
    plt.scatter(hsAll, wsAll, label='Data')

    for i,d in enumerate(Ds):
        X_design = commputeDesignX(X=hs, d=d)
        X_designAll = commputeDesignX(X=hsAll, d=d)
        theta_MLE_unstable = leastSquaresUnstable(X_design=X_design,Y=ws)
        theta_MLE_Stable = leastSquaresStable(X_design=X_design,Y=ws)
        print("d: %d    theta_MLE_unstable: %s \n" % (d, theta_MLE_unstable))
        print("d: %d    theta_MLE_Stable: %s \n" % (d,theta_MLE_Stable))
        # Predictions for outliers
        X_design_Out = commputeDesignX(hsOut,d=d)
        predsOutliers = predict(X_design=X_design_Out, theta=theta_MLE_Stable)
        for j,predO in enumerate(predsOutliers):
            print("height: %f  predicted weight: %f" % (hsOut[j], predsOutliers[j]))
        print('------------------')

        predictions = predict(X_design=X_designAll, theta=theta_MLE_Stable)
        X_design_xs = commputeDesignX(xs, d=d)
        ys = predict(X_design=X_design_xs, theta=theta_MLE_Stable)

        # Plot
        plt.xlabel('Height')
        plt.ylabel('Weight')
        label = 'd=' + str(d)
        plt.ylim([-10, 200])
        plt.scatter(hsAll, predictions, color=colorsScatter[i],label=label)
        plt.plot(xs, ys, colors[i])
        plt.legend()

    plt.show()


