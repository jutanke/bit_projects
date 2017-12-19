
import numpy as np
import matplotlib.pyplot as plt
import sys
def predict(X_design, theta):
    predictions = np.matmul(X_design,theta)
    return predictions

def bayesianRegression(X_design, Y, sigmaSquare, sigma_0_square):
    X_T_X = np.matmul(X_design.T,X_design)
    regulariser = sigmaSquare/sigma_0_square
    I_regularised = regulariser*np.identity(X_T_X.shape[0])
    inverse = np.linalg.inv(np.add(X_T_X,I_regularised))
    theta_MAP = np.matmul(np.matmul(inverse,X_design.T), Y)

    return theta_MAP

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

if __name__ == '__main__':

    # Load data
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

    ws = wsAll[wIndex]
    hs = hsAll[wIndex]
    gs = gsAll[wIndex]

    # Apply Bayessian regression
    sigma_0_square = 3.
    sigmaSquare = 1.

    X_design_hsAll_d_5 = commputeDesignX(X=hsAll, d=5)
    X_design = commputeDesignX(X=hs,d=5)
    thetaMLE_Unstable = leastSquaresUnstable(X_design=X_design, Y=ws)
    thetaMLE_Stable = leastSquaresStable(X_design=X_design, Y=ws)

    # Select sigmaSquare
    X_design_d_1 = commputeDesignX(X=hs,d=1)
    theta_MLE_d_1 = leastSquaresStable(X_design=X_design_d_1,Y=ws)
    predictions_d_1 = predict(X_design=X_design_d_1, theta=theta_MLE_d_1)

    residuals = predictions_d_1 - ws
    varianceResiduals = np.var(residuals)

    sigmaSquare = varianceResiduals
    SigmaSquare  = [varianceResiduals,1.,3000.]

    theta_MAP = bayesianRegression(X_design=X_design, Y=ws, sigmaSquare=sigmaSquare, sigma_0_square=sigma_0_square)
    print("theta_MLE_unstable: %s \n" % thetaMLE_Unstable)
    print("theta_MLE_Stable: %s \n" % thetaMLE_Stable)
    print("theta_MAP: %s \n" % theta_MAP)

    predictions = predict(X_design_hsAll_d_5, theta_MAP)

    print("-----------Predictions based on Sigma^2= Variance of residuals-----------")
    for i,height in enumerate(hsAll):
        print("height: %f  predicted weight: %f" % (height, predictions[i]))

    # Plot results
    fig, axs = plt.subplots(2,2,figsize=(10, 10))
    axs = axs.ravel()
    axs[0].set_ylim([-10, 200])

    xs = np.linspace(150, 200, 1000)
    X_design_xs = commputeDesignX(xs,d=5)

    ys = predict(X_design=X_design_xs,theta=theta_MAP)

    axs[0].set_xlabel('Height')
    axs[0].set_ylabel('Weight')
    axs[0].scatter(hsAll, wsAll, label='Data')

    axs[0].plot(hs, predictions_d_1, 'C2',label='Least Squares-1d')
    axs[0].legend()

    for i,sigmaSquare in enumerate(SigmaSquare):
        currentPlt = i+1
        theta_MAP_current = bayesianRegression(X_design=X_design, Y=ws, sigmaSquare=sigmaSquare, sigma_0_square=sigma_0_square)
        predictions = predict(X_design=X_design_hsAll_d_5,theta=theta_MAP_current)
        axs[currentPlt].set_ylim([-10, 200])
        label = 'MAP'
        axs[currentPlt].set_xlabel('Height')
        axs[currentPlt].set_ylabel('Weight')
        axs[currentPlt].scatter(hsAll, wsAll, label='Data')
        axs[currentPlt].scatter(hsAll, predictions, label=label)
        ys = predict(X_design=X_design_xs, theta=theta_MAP_current)
        axs[currentPlt].plot(xs, ys, 'C1--')
        title = 'Fitting based on sigma^2 = ' + str(round(sigmaSquare,2))
        axs[currentPlt].set_title(title)
        axs[currentPlt].legend()
    plt.show()


