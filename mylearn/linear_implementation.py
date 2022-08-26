import numpy as np
from .tools import addones

def linear_regression_matrix(X,W):
    """Matrix version of linear regression model: f(y) = w0+w1*x1+w2*x2+...+wn*xn

    Args:
        X (list): Matrice NxM, where N is number of strings and M is number of features.
            It is important that every row of X must be prepended with 1, for example:
            [[1,0.5,2,5],
             [1,0.2,3,6],
             [1,2  ,5,3],
             ...
             [1,0.1,1,2]]
        W (list): Vector of weights, including bias. For example, for X given above
            weights vector might be:
            [0.34,-0.25,1.1,0.2]

    Returns:
        list: a list of strings representing the header columns
    """
    return np.dot(addones(X),W)


def linear_regression_iterable(X,W):
    """Iterative form of linear regression model: f(y) = w0+w1*x1+w2*x2+...+wn*xn

    Args:
        X (list): Matrice NxM, where N is number of strings and M is number of features.
            For example:
            [[0.5,2,5],
             [0.2,3,6],
             [2  ,5,3],
             ...
             [0.1,1,2]]
        W (list): Vector of weights, including bias. For example, for X given above
            weights vector might be:
            [0.34,-0.25,1.1,0.2]

    Returns:
        list: a list of strings representing the header columns
    """
    Y = np.zeros(X.shape[0])
    for rown,Xrow in enumerate(X.values):
        y = 0
        for i in range(1,len(Xrow)):
            x = Xrow[i]
            y += W[i]*x
        y += W[0]
        Y[rown] = y
    return np.array(Y)

