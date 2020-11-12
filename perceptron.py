import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap

class Perceptron(object):
    def __init__(self, itr=10):
        self.itr = itr
        self.errors = []
        self.weights = np.zeros(1 + x.shape[1])
    
    def fit(self, x, y, l2):
        self.weights = np.zeros(1 + x.shape[1])
        for i in range(self.itr):
            error = 0
            for xi, target in zip(x, y):
                update = 1 * (target - 2 * l2 * self.predict(xi))
                self.weights[1:] += update*xi
                self.weights[0] += update
                error += int(update != 0)
            self.errors.append(error)
            print(self.weights)
        return self
    
    def net_input(self, x):
        return np.dot(x, self.weights[1:]) + self.weights[0]
    
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)

y = pd.read_csv("train.data", header=None)

# # Class 1 and class 2
# x = y.loc[np.r_[0:80], [0, 1, 2, 3]].values
# y = y.loc[np.r_[0:80], 4].values

# # Class 2 and class 3
# x = y.loc[np.r_[40:120], [0, 1, 2, 3]].values
# y = y.loc[np.r_[40:120], 4].values

# # Class 1 and class 3
# x = y.loc[np.r_[0:40, 80:120], [0, 1, 2, 3]].values
# y = y.loc[np.r_[0:40, 80:120], 4].values

# # Class 1, 2, and 3
# x = y.loc[np.r_[0:120], [0, 1, 2, 3]].values
# y = y.loc[np.r_[0:120], 4].values

# y = np.where(y == 'class-1', -1, 1)
# y = np.where(y == 'class-2', -1, 1)
# y = np.where(y == 'class-3', -1, 1)

Classifier = Perceptron(itr=20)
Classifier.fit(x, y, 1)
plt.plot(range(1, len(Classifier.errors) + 1), Classifier.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of misclassifications')
plt.show()

q = pd.read_csv("test.data", header=None)

# p = q.loc[0:9, [0, 1, 2, 3]]
# p = q.loc[10:19, [0, 1, 2, 3]]
# p = q.loc[20:29, [0, 1, 2, 3]]

a = Classifier.predict(p)

# b = q.loc[0:9, 4]
# b = q.loc[10:19, 4]
# b = q.loc[20:29, 4]

# b = np.where(b == 'class-1', -1, 1)
# b = np.where(b == 'class-2', -1, 1)
# b = np.where(b == 'class-3', -1, 1)

print(a)
print(b)
print((np.sum(a == b)/len(b))*100)

l2_regularization = [0.01, 0.1, 1.0, 10.0, 100.0]
for a in l2_regularization:
    Classifier = Perceptron(itr=20)
    Classifier.fit(x, y, a)
    plt.plot(range(1, len(Classifier.errors) + 1), Classifier.errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of misclassifications')
    plt.show()
    
    q = pd.read_csv("test.data", header=None)

    # p = q.loc[0:9, [0, 1, 2, 3]]
    # p = q.loc[10:19, [0, 1, 2, 3]]
    # p = q.loc[20:29, [0, 1, 2, 3]]

    a = Classifier.predict(p)

    # b = q.loc[0:9, 4]
    # b = q.loc[10:19, 4]
    # b = q.loc[20:29, 4]

    # b = np.where(b == 'class-1', 1, -1)
    # b = np.where(b == 'class-2', 1, 1)
    # b = np.where(b == 'class-3', -1, 1)

    print(a)
    print(b)
    print((np.sum(a == b)/len(b))*100)
