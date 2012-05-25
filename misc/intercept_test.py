import numpy as np
from numpy import linalg as la

def fit(X, y): 
    return np.dot(np.dot(X.T, la.pinv(np.dot(X, X.T))), y)

def intercept(Xmean, ymean, b):
    return ymean - np.dot(Xmean, b)

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([sum(x)*2 + 1 for x in X])       

T = np.array([[7, 8], [9, 10], [2, 1]])
Ty = np.array([sum(t)*2 + 1 for t in T])       

print X, y
print T, Ty

Xmean = X.mean(axis=0)
ymean = y.mean()
Xc = np.c_[np.ones(len(X)), X] #- Xmean
yc = y #- ymean

b = fit(Xc, yc)
b, b0 = b[1:], b[0]
#b0 = intercept(Xmean, ymean, b)

print
#print b, b0

print np.dot(X, b) + b0
print np.dot(T, b) + b0


