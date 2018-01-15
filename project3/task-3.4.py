import numpy as np
import matplotlib.pyplot as plt

X = np.loadtxt('./data/xor-X.csv', delimiter=',')
Y = np.loadtxt('./data/xor-y.csv', delimiter=',');
X = np.array(zip(X[0], X[1]))

theta = np.random.uniform(-1, 1)
w = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

def plot(y): 
    mat = np.array(zip(*X)).reshape([2, 200])
    plt.scatter(mat[0], mat[1], c=['b' if i > 0  else 'r' for i in y])

def calculate_exp(x, w, theta):
    return np.exp(-1 / 2 * np.square(np.subtract(np.matmul(w.transpose(), x), theta)))

def activation_function(x, w, theta):
    return 2 * calculate_exp(x, w, theta) - 1

'''
1. initialize w and theta
2. calculate deltaW and deltaT for all 200 samples
3. update deltaW, T and go over all examples again for 1000 times 
'''

learning_curve = 0.001
errors = []
epochs = 10000
for epoch in range(epochs):
    print epoch
    DT, DW, ET = 0, 0, 0
    for i in range(200):
        x = X[i]
        diffY = activation_function(X[i], w, theta) - Y[i]        
        exp = calculate_exp(x, w, theta)
        DT += diffY * 2 * exp * (np.matmul(x, w) - theta) 
        DW += diffY * 2 * exp * (np.matmul(x, w) - theta) * x * -1
        ET +=  0.5 * (diffY ** 2)
        
    w = w - learning_curve * DW
    theta = theta - learning_curve * DT
    errors.append(ET)
    
plt.grid(True)
# plt.plot(range(epoch)[10::100], errors[10::100], 'bo')
y_ = [activation_function(X[i], w, theta) for i in range(len(Y))]
plot(y_)
plt.show()
