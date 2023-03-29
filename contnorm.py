#contnorm.py

from numpy import sort,int 

def contnorm(d0,fraction = None, axis = None):
    
    '''
    Determines the normalization factor of a numpy array of up to 4 dimensions by sorting
    the array, and locating the value for which the number of entries below the value
    correspond the inputed fraction.

    Default fraction value is 0.85

    Note that normalizing N-dimensional data cube along given axes can be accomplished
    more efficiently with other functions. 

    Written by Tom Schad - 10 March 2016

    '''
    if fraction == None:
        fraction = 0.85

    if axis == None:
        return (sort(d0,axis = axis))[int(fraction*d0.size)]

    if axis != None:
        print d0.ndim,axis
        if d0.ndim <= axis:
            raise ValueError("Input array does not have selected axis")
        if axis == 0:
            if d0.ndim == 1: return (sort(d0,axis = axis))[int(fraction*d0.shape[0])]
            if d0.ndim == 2: return (sort(d0,axis = axis))[int(fraction*d0.shape[0]),:]
            if d0.ndim == 3: return (sort(d0,axis = axis))[int(fraction*d0.shape[0]),:,:]
            if d0.ndim == 4: return (sort(d0,axis = axis))[int(fraction*d0.shape[0]),:,:,:]
        if axis == 1:
            if d0.ndim == 2: return (sort(d0,axis = axis))[:,int(fraction*d0.shape[1])]
            if d0.ndim == 3: return (sort(d0,axis = axis))[:,int(fraction*d0.shape[1]),:]
            if d0.ndim == 4: return (sort(d0,axis = axis))[:,int(fraction*d0.shape[1]),:,:]
        if axis == 2:
            if d0.ndim == 3: return (sort(d0,axis = axis))[:,:,int(fraction*d0.shape[2])]
            if d0.ndim == 4: return (sort(d0,axis = axis))[:,:,int(fraction*d0.shape[2]),:]
    
def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2 

if __name__ == "__main__":
    print('testing contnorm')
    import numpy as np
    import matplotlib.pyplot as plt
    plt.ion()
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
    z = np.zeros((100,200,50))
    z0 = func(grid_x,grid_y)
    for iz in range(0,50):z[:,:,iz] = z0 * (-1)**iz
    print contnorm(z)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(131)
    ax1.imshow(contnorm(z,fraction = 0.75,axis = 0))
    ax2 = fig1.add_subplot(132)
    ax2.imshow(contnorm(z,fraction = 0.75,axis = 1))
    ax3 = fig1.add_subplot(133)
    ax3.imshow(contnorm(z,fraction = 0.85,axis = 2))

