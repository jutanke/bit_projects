import numpy as np
import matplotlib.pyplot as plt

x = np.loadtxt('./data/xor-X.csv', delimiter=',')
y = np.loadtxt('./data/xor-y.csv', delimiter=',');

colors = [int(i % 10) for i in y]
plt.scatter(x[0], x[1], c=['b' if i == 1 else 'r' for i in y])
plt.show()
