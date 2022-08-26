import numpy as np
from typing import Callable

from .tools import addones

'''
Base GD implementation
'''
def learn_base_gd(X, Y, n_epoch: int, learn_rate: float, stop_rate: float, linear_func: Callable, loss_func: Callable, derivative_func: Callable):
    W = np.zeros(X.shape[1]+1)
    epoch_passed = 0

    loss = 0
    for enum in range(n_epoch):
        epoch_passed += 1
        Y_predicted = linear_func(X,W)
        #print("Y_pred: {}".format(Y_predicted))
        nloss = loss_func(Y,Y_predicted)
        diff = abs(loss - nloss)
        if(diff <= stop_rate):
            break
        loss = nloss

        gradients = derivative_func(X,Y,Y_predicted)
        W = W-learn_rate*gradients

    print(gradients)
    print(W)
    return W, epoch_passed

