# CPSC 4420 Project 5
# Nick Gustafson and Lake Summers

# learning.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement linear and logistic regression
using the gradient descent method, as well as the binary perceptron algorithm. 
To complete the assignment, please modify the linear_regression(), binary_perceptron(), 
and logistic_regression() functions. 

The package `matplotlib` is needed for the program to run.
You should also use the 'numpy' library to vectorize 
your code, enabling a much more efficient implementation of 
linear and logistic regression. You are also free to use the 
native 'math' library of Python. 

All provided datasets are extracted from the scikit-learn machine learning library. 
These are called `toy datasets`, because they are quite simple and small. 
For more details about the datasets, please see https://scikit-learn.org/stable/datasets/index.html

Each dataset is randomly split into a training set and a testing set using a ratio of 8 : 2. 
You will use the training set to learn a regression model. Once the training is done, the code
will automatically validate the fitted model on the testing set.  
"""

# use math and/or numpy if needed
import math
import numpy as np

def linear_regression(x, y, logger=None):
    """
    Linear regression using full batch gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by x^T w, where x is a column vector. 
    The intercept term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    You should use as learning rate alpha=0.0001. If you scale the cost function by 1/#samples, use alpha=0.001  

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the target value for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.
    
    Returns
    -------
    w: a 1D array
       linear regression parameters
    """
    #Initialize alpha and weights
    alpha=0.0001
    w = np.zeros(len(x[0]))

    def getGuess(inputs):
        #Compute dot between weights and inputs
        return np.dot(w.T, inputs)
    
    def getError():
        #Set error to 0
        error = 0
        #For each input/output pair
        for inputs, output in zip(x, y):
            #Compute error
            error = error + (getGuess(inputs) - output)**2
        return error / 2
            
    def getDifference(index):
        #Set difference to 0
        difference = 0
        #For each input/output pair
        for inputs, output in zip(x, y):
            difference = difference + ((getGuess(inputs) - output) * inputs[index])
        return difference
    
    #Set past/current error
    pastError = 0
    currentError = -1
    j = 0
    #Keep running until error threshhold and beyond 500 iterations
    while currentError - pastError > 0.01 or j < 500:
        #For each weight
        for i, _ in enumerate(w):
            #Update weight
            w[i] = w[i] - alpha * getDifference(i)
        #Increment J
        j = j + 1
        #Update past error
        pastError = currentError
        #Get new error
        currentError = getError()

    #Return Weights
    return w

def binary_perceptron(x, y, logger=None):
    """
    Binary classifaction using a perceptron. 
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by
        h = (x^T w) 
    with the decision boundary:
        h >= 0 => x in class 1
        h < 0  => x in class 0
    where x is a column vector. 
    The intercept/bias term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    
    
    Parameters
    ----------
    x: a 2D array with the shape [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array with the shape [N]
       It is the ground truth value for each sample in x
    logger: a logger instance through which plotting loss
       Usage: Please do not use the logger in this function.
    
    Returns
    -------
    w: a 1D array
       binary perceptron parameters
    """
    #Initialize Vars
    w = [0] * len(x[0])
    dotVal = 0
    yPrediction = None
    convergence = False

    #While weights haven't converged
    while not convergence:
        #Set flag for convergence to true so it can be changed later
        convergence = True
        #For the number of samples
        for n in range(len(x)):
                #Get dotVal (dot of x[n]^T and w)
                dotVal = np.dot(np.transpose(x[n]), w)
                #Set prediction
                if dotVal >= 0:
                    yPrediction = 1
                else:
                    yPrediction = 0
                #Compare prediction to y
                if yPrediction != y[n]:
                    #Adjust w by x^T
                    w = w + ((y[n]-yPrediction) * np.transpose(x[n]))
                    #Set flag for convergence to false
                    convergence = False        
    #Return weights
    return w


def logistic_regression(x, y, logger=None):
    """
    Logistic regression using batch gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by p = sigmoid(x^T w)
    with the decision boundary:
        p >= 0.5 => x in class 1
        p < 0.5  => x in class 0
    where x is a column vector. 
    The intercept/bias term can be ignored due to that x has been augmented by adding '1' as an extra feature. 
    In gradient descent, you should use as learning rate alpha=0.001    

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the ground truth label for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.
        
    Returns
    -------
    w: a 1D array
       logistic regression parameters
    """
    alpha=0.0001
    w = np.zeros(len(x[0]))

    def getGuess(inputs):
        # Get normal guess
        guess = np.dot(w.T, inputs)
        # Raise it to the e power
        guess = math.e**(-1 * guess)
        # Return the actual guess
        return 1 / (1 + guess)
    
    def getError():
        # Go through all points and figure out error
        error = 0
        for i, inputs in enumerate(x):
            if y[i] == 0:
                error += np.log(1 - getGuess(inputs))
            else:
                error += np.log(getGuess(inputs))

        return -1 * error
            
    def getDifference(index):
        difference = 0
        for inputs, output in zip(x, y):
            # Finds the difference per point 
            difference = difference + ((getGuess(inputs) - output) * inputs[index])
        return difference
    
    pastError = 0
    currentError = -1
    j = 0
    # Making sure the algoritem goes past 500 iterations and then checks for convergence
    while currentError - pastError > 0.03 or j < 500:
        # Figure out new values of weights
        for i, _ in enumerate(w):
            w[i] = w[i] - alpha * getDifference(i)
        j = j + 1
        # Setup the comparison of the error
        pastError = currentError
        currentError = getError()

    return w

if __name__ == "__main__":
    import os
    import tkinter as tk
    from app.regression import App

    import data.load
    dbs = {
        "Boston Housing": (
            lambda : data.load("boston_house_prices.csv"),
            App.TaskType.REGRESSION
        ),
        "Diabetes": (
            lambda : data.load("diabetes.csv", header=0),
            App.TaskType.REGRESSION
        ),
        "Handwritten Digits": (
            lambda : (data.load("digits.csv", header=0)[0][np.where(np.equal(data.load("digits.csv", header=0)[1], 0) | np.equal(data.load("digits.csv", header=0)[1], 1))],
                      data.load("digits.csv", header=0)[1][np.where(np.equal(data.load("digits.csv", header=0)[1], 0) | np.equal(data.load("digits.csv", header=0)[1], 1))]),
            App.TaskType.BINARY_CLASSIFICATION
        ),
        "Breast Cancer": (
            lambda : data.load("breast_cancer.csv"),
            App.TaskType.BINARY_CLASSIFICATION
        )
     }

    algs = {
       "Linear Regression (Batch Gradient Descent)": (
            linear_regression,
            App.TaskType.REGRESSION
        ),
        "Logistic Regression (Batch Gradient Descent)": (
            logistic_regression,
            App.TaskType.BINARY_CLASSIFICATION
        ),
        "Binary Perceptron": (
            binary_perceptron,
            App.TaskType.BINARY_CLASSIFICATION
        )
    }

    root = tk.Tk()
    App(dbs, algs, root)
    tk.mainloop()
