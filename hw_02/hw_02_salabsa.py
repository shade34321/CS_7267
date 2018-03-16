import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Most of this code was taken directly from your code in class.
"""
def solveLinearRegresion(X, y):
    """
        Finds the optimal B given the data X and Y
    """
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()),y)

def predict(X, b, threshold):
    """
        Determines which classification X is using B based on some threshold
    """
    return np.array(np.dot(X, b) > threshold)

def accuracy(X, y):
    """
        Determines how many we guessed correctly
    """
    return (sum(X == y) / float(len(y))) * 100

def min_max_norm(X):
    """
        Simple min-max norm to normalize the data. Not really sure this is needed but got an extra 2% in my GD accuracy.
    """
    return X / 255.0

def cost(X, y, b):
    """
        Calculates the Ordinary least Squares
    """
    return np.sum((np.dot(X,b) - np.array(y))**2)

def GD_LR(X, y, b):
    """
        Calculates gradient descient linear regression
    """
    return -np.dot(X.transpose(), y) + np.dot(np.dot(X.transpose(), X), b)

#Reads in the data
test_data = pd.read_csv('MNIST_test.csv', skiprows=[0], header=None)
training_data = pd.read_csv('MNIST_training.csv', skiprows=[0], header=None)

#Pulls out the labels which determines whats true -> you usually call y_test groundtruth 
y_training = training_data.iloc[:,0]
y_test = test_data.iloc[:,0]

# Grabs all the feature data
X_training = training_data.iloc[:,1:]
X_test = test_data.iloc[:,1:]

# Solves for b_opt using the training data
b_opt = solveLinearRegresion(X_training, y_training)

#Predicts on the testing data
predictions = predict(X_test, b_opt, .5)
print "b_opt: "
print b_opt

print "Linear regression accuracy is %f%%" % accuracy(predictions, y_test)

"""
  This figures out the GD using non normalized data. Somebody mentioned in class that in class it was stated GD data always needs to be normalized after I did this. I found it interesting how close the two are.
"""
# Pulls out the size of the matrix. We only need the number of columns so we don't bother with the rows.
_, p = X_training.shape
#Gives us an array of 0's for the coefficient. 
b_est = np.zeros(p)
# Purposely set the learning rate really low. If you set it higher it overflows. Tis interesting
learning_rate = 1e-10
bs = [b_est]
costs = [cost(X_training, y_training, b_est)]
# finds B-est so we can check the graph and ensure it converges
for i in range(0, 100):
    b_est = b_est - learning_rate * GD_LR(X_training, y_training, b_est)
    b_cost = cost(X_training, y_training, b_est)
    bs.append(b_est)
    costs.append(b_cost)

plt.plot(costs)
plt.show()
print b_est


# Predict on the test data
gd_prediction = predict(X_test, b_est, 0.5)
print "Non-normalized gradient descent b_est"
print b_est

#How accurate are we?
print "Non-normalized gradient descent accuracy is %f%%" % accuracy(gd_prediction, y_test)

# Differences between the two different b from LR and GD
total_diff = sum(abs(b_opt-b_est))
print "Non-normalized differenc"
print total_diff

"""
   The code below is exactly the same as the GD code above. But this time we use normalize data. It gave me an extra 2% and I could start with a bigger learning rate.
"""

X_training_norm = min_max_norm(X_training)
X_test_norm = min_max_norm(X_test)
_, p = X_training_norm.shape
b_est = np.zeros(p)
learning_rate = 1e-4
bs = [b_est]
costs = [cost(X_training_norm, y_training, b_est)]
for i in range(0, 100):
    b_est = b_est - learning_rate * GD_LR(X_training_norm, y_training, b_est)
    b_cost = cost(X_training_norm, y_training, b_est)
    bs.append(b_est)
    costs.append(b_cost)

gd_prediction = predict(X_test_norm, b_est, 0.5)
print "Normalized gradient descent b_est"
print b_est

print "Normalized gradient descent accuracy is %f%%" % accuracy(gd_prediction, y_test)

total_diff = sum(abs(b_opt-b_est))
print "Normalized differenc"
print total_diff
