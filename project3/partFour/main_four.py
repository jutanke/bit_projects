import numpy as np 
import matplotlib.pyplot as plt
import random 

# def f(x,w,theta):
# 	return 2 * np.exp(-0.5* (np.dot(w.T,x)-theta)**2) - 1

# def E_one(x, y, w, theta):
# 	return 0.5 * (f(x,w, theta)-y)**2

def f_acc(X,w,theta):
	return 2 * np.exp(-0.5* (np.dot(X,w)-theta)**2) - 1


def E(X, Y, w, theta):
	return 0.5 * np.sum((f_acc(X,w, theta)-Y)**2)

def dE_dW(X, Y, w, theta):
	temp = np.multiply((f_acc(X,w,theta)-Y), np.exp(-0.5 * (np.dot(X,w)-theta)**2 ))
	temp = np.reshape(( np.multiply( -2 * temp , np.dot(X,w)-theta) ),(X.shape[0],1))
	return np.sum(temp*X, axis=0)

def dE_dTheta(X, Y, w, theta):
	temp = np.multiply((f_acc(X,w,theta)-Y), np.exp(-0.5 * (np.dot(X,w)-theta)**2 ))
	temp = np.reshape(( np.multiply( 2 * temp , np.dot(X,w)-theta) ),(X.shape[0],1))
	return np.sum(temp, axis=0)

if __name__ == "__main__":
	X = np.genfromtxt('../xor-X.csv', delimiter=',').T
	Y = np.genfromtxt('../xor-y.csv', delimiter=',')

	X_pos = X[Y ==  1 ]
	X_neg = X[Y == -1 ]

	weights 	= np.random.uniform(-1,1,2)
	theta 		= np.random.uniform(-1,1,1)
	eta_theta 	= 0.001
	eta_weights = 0.005

	errors = np.array([])
	epoch_range = range(0,500)
	for ep in epoch_range:
		new_order = np.random.permutation(X.shape[0])
		X_r = X[new_order]
		Y_r = Y[new_order]

		weights = weights - eta_weights*dE_dW(X_r, Y_r, weights, theta)
		theta 	= theta - eta_theta*dE_dTheta(X_r, Y_r, weights, theta)
		error = E(X_r,Y_r,weights,theta)
		errors = np.concatenate( (errors, [error]) )
	


	###### visualizations #####
	f = plt.figure(1)
	plt.scatter(X_pos[:,0], X_pos[:,1],c='b',marker='o')
	plt.scatter(X_neg[:,0], X_neg[:,1],c='orange',marker='o')
	plt.grid(True)
	plt.savefig('original data')

	g = plt.figure(2)
	test_X = np.random.uniform(-1.5,1.5,10000)
	test_Y = np.random.uniform(-1.5,1.5,10000)
	test_XY= np.stack((test_X,test_Y)).T
	w_sums = f_acc(test_XY,weights,theta)
	plt.scatter(test_X, test_Y, c=['b' if i > 0 else 'orange' for i in w_sums], marker='.')
	plt.savefig('test')
	
	h = plt.figure(3)
	plt.plot(epoch_range, errors)
	plt.savefig('error')
	plt.show()

	print X.shape
	print Y.shape
