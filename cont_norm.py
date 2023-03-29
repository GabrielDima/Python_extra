def contnorm(d0,fraction = None, axis = None):

    '''
    Determines the normalization factor of a numpy array of up to 4 dimensions by sorting
    the array, and locating the value for which the number of entries below the value
    correspond to the inputed fraction.

    Default fraction value is 0.85

    Note that normalizing N-dimensional data cube along given axes can be accomplished
    more efficiently with other functions.

    Written by Tom Schad - 10 March 2016

    '''

    from numpy import sort,int

    if fraction == None:
        fraction = 0.85

    if axis == None:
        return (sort(d0,axis = axis))[int(fraction*d0.size)]

    if axis != None:
        #print d0.ndim,axis
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
