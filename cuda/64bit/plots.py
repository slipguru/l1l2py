import numpy as np

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt

def main():
    res_py = np.genfromtxt('results_python.txt')
    res_cpp = np.genfromtxt('results_cpp.txt')
    
    P = res_py[:,0]
    
    init_time_py = res_py[:,1]
    comp_time_py = res_py[:,2]
    
    init_time_cpp = res_cpp[:,1]
    comp_time_cpp = res_cpp[:,2]
    
    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].scatter(P, init_time_py)
    axarr[0, 0].set_title('Initialization time (python)')

    # axarr[0, 1].scatter(x, y)
    # axarr[0, 1].set_title('Axis [0,1]')
    # 
    # axarr[1, 0].plot(x, y ** 2)
    # axarr[1, 0].set_title('Axis [1,0]')
    # 
    # axarr[1, 1].scatter(x, y ** 2)
    # axarr[1, 1].set_title('Axis [1,1]')
    
    plt.show('testplot.pdf')
    
    

if __name__ == '__main__':
    main()