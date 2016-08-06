import numpy as np

from numpy.linalg import pinv, svd, inv

def main():
    """
    """
    
    A = np.array([[1,-2],[3,5]])
    A = A.T
    n, p = A.shape
    
    # print A
    
    res = svd(A)
    u = res[0]
    s = res[1]
    v = res[2]
    
    A_pinv = pinv(A)
    A_inv = inv(A)
    
    print A_inv
    
    return
    
    A_pinv2 = v.T.dot(np.diag(1/s)).dot(u.T)
    
    print "intermediate"
    print v.T.dot(np.diag(1/s))
    
    print "final"
    print A_pinv2
    
    # R1 = A.dot(A_inv).dot(A)
    # R2 = A.dot(A_pinv).dot(A)
    # R3 = A.dot(A_pinv2).dot(A)
    
    # R1 = A.dot(A_inv)
    # R2 = A.dot(A_pinv)
    # R3 = A.dot(A_pinv2)
    
    # print R1
    # print R2
    # print R3
    
    # print s
    
    # print u
    # print v

if __name__ == '__main__':
    main()