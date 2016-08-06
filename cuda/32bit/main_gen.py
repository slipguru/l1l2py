import numpy as np

import os

from glados.utils.gendata import grouped

def main():
    
    n = 200;
    
    # print len(range(100,10000, 50))
    
    # ps = range(50,200,10)
    # ps += range(200,1000, 50)
    # ps += range(1000, 20000, 100)

    # ps = [20000]
    # ns = 4000
    
    
    # ns = range(50, 4050, 50)
    # ps = range(100, 20100, 100)
    
    ns = [600]
    ps = [100000]

    main_folder = 'data_c'

    for n in ns:
        for p in ps:
            X, Y = grouped(n,p)
            
            X_fname = os.path.join(main_folder, 'X_%d_%d.csv' % (n,p))
            Y_fname = os.path.join(main_folder, 'Y_%d_%d.csv' % (n,p))
            
            np.savetxt(X_fname, X, fmt = '%.10e')
            np.savetxt(Y_fname, Y, fmt = '%.10e')
    
    # pass

if __name__ == '__main__':
    main()
