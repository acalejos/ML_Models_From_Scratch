#!/usr/bin/env python
import sys
import pandas as pd
import numpy as np
import copy
from Models.progressBar import progressBar
np.set_printoptions(threshold=sys.maxsize)
#############################################


#################Logistic Regression####################################
# Runs logistic regression and tests model against training and test sets
def lr(trainingSet, testSet):
    model = train_lr(trainingSet)
    test(trainingSet, model, 1, .5, "Training")
    test(testSet, model, 1, .5, "Testing")

# Trains LR model using training set


def train_lr(trainingSet):
    X = trainingSet.drop("decision", axis=1)  # Features
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = trainingSet['decision']  # Decision
    y = y[:, np.newaxis]
    # Given in homework prompt
    weights = np.zeros((X.shape[1], 1))
    Lambda = 0.01
    eta = 0.01
    iterations = 500
    tol = np.exp(-6)
    return logisticGradientDescent(X, y, weights, eta, Lambda, iterations, tol)

# Sigmoid function


def sigmoid(x):
    return 1/(1+np.e**(-x))

# Computes gradient of logistic loss function and returns dw


def gradient(X, y, weights, Lambda):
    z = np.dot(X, weights)
    y_hat = sigmoid(z)
    error = (y_hat-y)
    dw = (np.dot(X.T, error))+(Lambda*weights)
    return dw

# Measures L2 norm between two vectors


def dist(x, y):
    dist = np.linalg.norm(x-y)
    #dist =  np.sqrt(np.sum((x-y)**2))
    return dist

# Performs gradient descent


def logisticGradientDescent(X, y, weights, eta, Lambda, iterations, tol):
    for _ in range(iterations):
        dw = gradient(X, y, weights, Lambda)
        old = copy.copy(weights)
        weights = weights - (eta*dw)
        if (dist(weights, old) < tol):
            break
    return weights

####################SVM##########################################


def svm(trainingSet, testSet):
    model = train_svm(trainingSet)
    test(trainingSet, model, 2, .5, "Training Set")
    test(testSet, model, 2, .5, "Test Set")


def train_svm(trainingSet):
    X = trainingSet.drop("decision", axis=1)  # Features
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = trainingSet['decision']  # Decision
    y = y[:, np.newaxis]
    weights = np.zeros((X.shape[1], 1))
    Lambda = 0.5
    eta = 0.5
    iterations = 500
    tol = np.exp(-6)
    return svmGradientDescent(X, y, weights, eta, Lambda, iterations, tol)


def svm_gradient(weights, X, y, reg):
    dw = np.zeros(weights.shape)
    #num_train = X.shape[0]
    predictions = np.dot(X, weights)
    # Matrix which measures correct vs incorrect classifications
    yi_predictions = (y == predictions).astype(int)
    error = predictions - yi_predictions
    error[error < 1] = -1
    error[error >= 1] = 1
    error[y, np.arange(0, predictions.shape[1])] = 1
    error[y, np.arange(0, predictions.shape[1])] = -1 * np.sum(error, axis=0)
    dw = np.dot(X.T, error)
    # Average over number of training examples
    dw = dw / X.shape[0]
    return dw


def svmGradientDescent(X, y, weights, eta, Lambda, iterations, tol):
    for step in range(iterations):
        dw = svm_gradient(weights, X, y, Lambda)
        old = copy.copy(weights)
        weights = weights + (eta*dw)
        progressBar("SVM Training", step, iterations)
        if (dist(weights, old) < tol):
            break
    sys.stdout.flush()
    print("\n")
    return weights

# LR Prediction


def makePredictions(model, X, threshold):
    z = np.dot(X, model)
    return sigmoid(z) >= threshold

# SVM Prediction


def predict_svm(weights, X):
    return np.sign(np.dot(X, weights))

# Runs test using given model and predictive technique


def test(testSet, model, modelIdx, threshold, name):
    X = testSet.drop("decision", axis=1)  # Features
    y = testSet['decision']  # Decision
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    # Logistic Regression
    if modelIdx == 1:
        predictions = makePredictions(model, X, threshold)
        accuracy = round(100*np.mean(y == predictions), 2)
        print(name + " Accuracy LR: " + str(accuracy))
        return accuracy
    elif modelIdx == 2:
        predictions = predict_svm(model, X)
        accuracy = round(100*np.mean(y == predictions), 2)
        print(name + " Accuracy SVM: " + str(accuracy))
        return accuracy
    else:
        print(
            "Usage: python lr_svm.py [trainingDataFilename] [testDataFilename] [model (1 or 2)]")


# Main function
def main(argv):
    trainingDataFilename = argv[0]
    testDataFilename = argv[1]
    modelIdx = int(argv[2])
    trainingSet = pd.read_csv(trainingDataFilename)
    testSet = pd.read_csv(testDataFilename)
    if modelIdx == 1:
        lr(trainingSet, testSet)
    elif modelIdx == 2:
        svm(trainingSet, testSet)
    else:
        print(
            "Usage: python lr_svm.py [trainingDataFilename] [testDataFilename] [model (1 or 2)]")


if __name__ == '__main__':
    if len(sys.argv) > 4:
        print(
            "Usage: python lr_svm.py [trainingDataFilename] [testDataFilename] [model (1 or 2)]")
        # sys.exit(0)
    else:
        main(sys.argv[1:])
