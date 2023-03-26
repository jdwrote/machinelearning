# # Outline
# - [ 1 - Packages ](#1)
# - [ 2 - Logistic Regression](#2)
#   - [ 2.1 Problem Statement](#2.1)
#   - [ 2.2 Loading and visualizing the data](#2.2)
#   - [ 2.3  Sigmoid function](#2.3)
#   - [ 2.4 Cost function for logistic regression](#2.4)
#   - [ 2.5 Gradient for logistic regression](#2.5)
#   - [ 2.6 Learning parameters using gradient descent ](#2.6)
#   - [ 2.7 Plotting the decision boundary](#2.7)
#   - [ 2.8 Evaluating logistic regression](#2.8)
# - [ 3 - Regularized Logistic Regression](#3)
#   - [ 3.1 Problem Statement](#3.1)
#   - [ 3.2 Loading and visualizing the data](#3.2)
#   - [ 3.3 Feature mapping](#3.3)
#   - [ 3.4 Cost function for regularized logistic regression](#3.4)
#   - [ 3.5 Gradient for regularized logistic regression](#3.5)
#   - [ 3.6 Learning parameters using gradient descent](#3.6)
#   - [ 3.7 Plotting the decision boundary](#3.7)
#   - [ 3.8 Evaluating regularized logistic regression model](#3.8)

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
from Coursera_ML_C1_W3_ProgramingAssignment_LogisticRegression_part1 import compute_cost, compute_gradient, \
    compute_gradient_reg_test, gradient_descent

# - [ 3 - Regularized Logistic Regression](#3)
# In this part of the exercise, you will implement regularized logistic regression to predict
# whether microchips from a fabrication plant passes quality assurance (QA).
# During QA, each microchip goes through various tests to ensure it is functioning correctly.

# 3.1 Problem Statement
# Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests.
# From these two tests, you would like to determine whether the microchips should be accepted or rejected.
# To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.

# load dataset
X_train, y_train = load_data("./ex2data2.txt")

# print X_train
print("X_train:", X_train[:5])
print("Type of X_train:",type(X_train))

# print y_train
print("y_train:", y_train[:5])
print("Type of y_train:",type(y_train))

print ('The shape of X_train is: ' + str(X_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))

# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

# Set the y-axis label
plt.ylabel('Microchip Test 2')
# Set the x-axis label
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show()

#   - [ 3.3 Feature mapping](#3.3)

# One way to fit the data better is to create more features from each data point.
# In the provided function map_feature, we will map the features into all polynomial terms of  𝑥1 and  𝑥2
# up to the sixth power.

print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)

# Let's also print the first elements of X_train and mapped_X to see the tranformation.
print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])



# While the feature mapping allows us to build a more expressive classifier, it is also more susceptible to overfitting.
# In the next parts of the exercise, you will implement regularized logistic regression to fit the data
# and also see for yourself how regularization can help combat the overfitting problem.


#   - [ 3.4 Cost function for regularized logistic regression](#3.4)

# UNQ_C5
def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar, float) Controls amount of regularization
    Returns:
      total_cost : (scalar)     cost
    """

    m, n = X.shape

    # Calls the compute_cost function that you implemented above
    cost_without_reg = compute_cost(X, y, w, b)

    # You need to calculate this value
    reg_cost = 0.

    ### START CODE HERE ###
    for j in range(n):
        reg_cost += (w[j] ** 2)

    reg_cost = (lambda_ / (2 * m)) * reg_cost

    ### END CODE HERE ###

    # Add the regularization cost to get the total cost
    total_cost = cost_without_reg + reg_cost

    return total_cost

X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)

# Please complete the compute_gradient_reg function below to modify the code below to calculate the following term

#   - [ 3.5 Gradient for regularized logistic regression](#3.5)

def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Computes the gradient for logistic regression with regularization

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model
      lambda_ : (scalar,float)  regularization constant
    Returns
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b.
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w.

    """
    m, n = X.shape

    dj_db, dj_dw = compute_gradient(X, y, w, b)

    ### START CODE HERE ###

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_ / m) * w[j]

    ### END CODE HERE ###

    return dj_db, dj_dw


X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5

lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )

# UNIT TESTS
compute_gradient_reg_test(compute_gradient_reg)

#   - [ 3.6 Learning parameters using gradient descent](#3.6)

# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ (you can try varying this)
lambda_ = 0.01

# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b,
                                    compute_cost_reg, compute_gradient_reg,
                                    alpha, iterations, lambda_)

#   - [ 3.7 Plotting the decision boundary](#3.7)

plot_decision_boundary(w, b, X_mapped, y_train)
# Set the y-axis label
plt.ylabel('Microchip Test 2')
# Set the x-axis label
plt.xlabel('Microchip Test 1')
plt.legend(loc="upper right")
plt.show()

#   - [ 3.8 Evaluating regularized logistic regression model](#3.8)
# You will use the predict function that you implemented above to calculate the accuracy
# of the regularized logistic regression model on the training set

#Compute accuracy on the training set
p = predict(X_mapped, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))