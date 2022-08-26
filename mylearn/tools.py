import numpy as np

'''
adds one to features matrix
'''
def addones(X):
    return np.hstack((np.ones((X.shape[0],1)),X))
