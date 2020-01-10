#  MATH BASIS FOR DETRENDING:
#  To detrend / flatten a 2d array, we need to understand the problem. We need to find a plane of best fit by regression.
#  Regression is done between the angle of rotation for each point to fit the height data the best. Thus:
#  X * Theta = Y must be solved.
#  X:= coordinate matrix (first column is a vector of ones; second and third are the matrix coordinate)
#  Theta:= angle of rotation of the plane
#  Y:= height data, unraveled into a 2d array with only 1 column.
#
#  X * Theta = Y
#  X_T * X * Theta = X_T * Y
#  Theta = (X_T * X) ^ -1 * X_T * Y
#  CODE
#

import numpy as np

def plane_fit(arr):
    '''m, n describes the shape of the 2d array'''
    out = np.copy(arr)
    m,n = out.shape
    

    #  Creating X
    X1, X2 = np.mgrid[:m,:n]
    X = np.hstack( (np.ones((m*n,1)),np.reshape(X1,(m*n,1))) )
    X = np.hstack( (X, np.reshape(X2,(m*n,1))) )

    #  Creating Y
    Y = np.reshape(out,(m*n,1))

    #  Fitting for theta
    theta = np.dot(   np.dot( np.linalg.pinv(np.dot(X.transpose(),X)) , X.transpose() )   ,   Y)
    plane = np.reshape(np.dot(X,theta) , (m,n))
    return plane

def subtract_plane(arr):
    plane = plane_fit(arr)
    return arr - plane