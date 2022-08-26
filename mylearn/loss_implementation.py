import numpy as np
from typing import Callable
from .tools import addones

def binary_cross_entropy_iterable(Y, Y_predicted):
    n = Y.shape[0]
    total_cost = 0
    for i in range(n):
        cost = Y[i]*np.log(Y_predicted[i])+(1-Y[i])*np.log(1-Y_predicted[i])
        total_cost += cost
    total_cost = -(1 / n) * total_cost
    return total_cost

def binary_cross_entropy_derivative_iterable(X, Y, Y_predicted):
    # there are two derivatives - for bias and for weights
    # calculating bias derivative
    db = 0
    for i, y in enumerate(Y):
        db += Y_predicted[i] - Y[i]
    db = db/X.shape[0]

    # calculating weights
    W = np.zeros(X.shape[1])
    for i, Xrow in enumerate(X.values):
        for j, x in enumerate(Xrow):
            W[j] += x*(Y_predicted[i] - Y[i])

    for i, w in enumerate(W):
        W[i] = w/X.shape[0]

    return np.insert(W,0,db,axis=0)

def binary_cross_entropy_matrix(Y, Y_predicted):
    m = Y.shape[0]
    total_cost = -(1 / m) * np.sum(
        Y * np.log(Y_predicted) + (1 - Y) * np.log(
            1 - Y_predicted))
    return total_cost

def binary_cross_entropy_derivative_matrix(X, Y, Y_predicted):
    m = Y.shape[0]
    return (1 / m) * np.dot(addones(X).T, Y_predicted - Y)
