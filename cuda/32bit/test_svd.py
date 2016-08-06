import numpy as np

from numpy.linalg import svd

def main():
    """
    """
    
    A = np.array([[1,-2],[3,5]])
    
    print A
    
    res = svd(A)
    u = res[0]
    s = res[1]
    v = res[2]
    
    print s

if __name__ == '__main__':
    main()