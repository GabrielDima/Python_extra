# shc.py 

import numpy as np 
import cv2 

######################################################
def shc(d0,d1):
    '''
    Find the linear shift between two 1 or 2d arrays 
    using the fourier crosscorrelation method and
    interpolation for sub-pixel shifts. 

    This is a python implementation of the IDL shc.pro procedure, 
    originally written by P.Suetterlin of KIS.

    The implementations does not include an edge filter for the fft 
    for now.

    Two versions are available in python.  One based on Numpy FFT 
    and the other on OPENCV.  The default version is the OPENCV
    implementation, as it is in general faster.

    Warning...zero padding effects the results when there is 
    low constrast.  Not sure how to fix this yet.  NUMPY fft 
    should have zero padding, but its unclear how this is done 
    differently than how I am doing it. 

    Use of FFTW might happen in the future.  Initially tests with 
    the pyfftw interfaces module did show speedups of the fftw 
    compared to numpy, but only having multiple runs.  This has 
    something to do with the planning of the FFT.  

    Written by Tom Schad - 24 Feb 2016

    '''  

    print(' Note:  Need to implement this module as a try/exception type ')
    print(' Much like the get_ffts module..this will always default to fastest available ') 
    print(' Also try to make it use the axis keyword and/or cv2rows flag for fast 2d/1d app')
    print(' This same type of operation can be made to be the base of the destretch routines')

    return shc_opencv(d0,d1)

######################################################
def shc_numpy(d0,d1):

    '''
    Read DocString for SHC
    '''

    d0 = d0 - d0.mean()
    d1 = d1 - d1.mean()
    
    if d0.shape != d1.shape:
        raise ValueError("Input shapes do not match")

    if d0.ndim > 2:
        raise ValueError("Only 1-d or 2-d data supported")
        
    ## one dimensional case 
    if d0.ndim == 1: 
        cc = np.fft.fftshift(np.abs(np.fft.ifft(np.fft.fft(d0) * np.fft.fft(d1).conjugate())))
        xmax = cc.argmax()
        ## Polyfit of degree 2 for three points, extremum
        c1 = (cc[xmax+1]-cc[xmax-1])/2.
        c2 = cc[xmax+1]-c1-cc[xmax]
        xmax = xmax - c1/c2/2.
        return  xmax - d0.shape[0]/2
        
    ## two dimensional case
    if d0.ndim == 2:
        ## find the maximize correlation via the FFT method
        cc = np.fft.fftshift(np.abs(np.fft.ifft2(np.fft.fft2(d0) * np.fft.fft2(d1).conjugate())))
        indices = np.where(cc == cc.max())
        rmax = (indices[0])[0]
        cmax = (indices[1])[0]
        ## Interpolate to sub-pixel accuracy 
        if (rmax*cmax >= 0) and (rmax < d0.shape[0]) and (cmax < d0.shape[1]):
            ## interpolate to sub-pixel accuracy
            ## we use a quadratic estimator as in Tian and Huhns (1986) pg 222
            denom = 2.*(2.*cc.max() - cc[rmax+1,cmax] - cc[rmax-1, cmax])
            rfra = (rmax) + (cc[rmax+1,cmax] -cc[rmax-1,cmax])/denom
            denom = 2.*(2.*cc.max() - cc[rmax,cmax+1] - cc[rmax, cmax-1])
            cfra = (cmax) + (cc[rmax,cmax+1] -cc[rmax,cmax-1])/denom
            rmax = rfra
            cmax = cfra
        
        return np.array([rmax - d0.shape[0]/2. , cmax - d0.shape[1]/2.])

######################################################
def shc_opencv(d0,d1):

    '''
    Read DocString for SHC
    '''
    
    d0 = np.float32(d0 - d0.mean())
    d1 = np.float32(d1 - d1.mean())
    
    if d0.shape != d1.shape:
        raise ValueError("Input shapes do not match")

    if d0.ndim > 2:
        raise ValueError("Only 1-d or 2-d data supported")
        
    ## one dimensional case 
    if d0.ndim == 1:
        ## Manually Pad Arrays for DFT Optimization
        rows = (d0.shape)[0]
        nrows = cv2.getOptimalDFTSize(rows)
        d0n = np.zeros((nrows))
        d1n = np.zeros((nrows))
        d0n[:rows] = d0
        d1n[:rows] = d1
        ## Fourier Phase Correlation
        d0_dft = cv2.dft(d0, flags = cv2.DFT_COMPLEX_OUTPUT)
        d1_dft = cv2.dft(d1, flags = cv2.DFT_COMPLEX_OUTPUT)
        d01 = cv2.mulSpectrums(d0_dft,d1_dft,flags = 0, conjB = True)
        id01 = cv2.idft(d01, flags = cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        cc = np.fft.fftshift(np.abs(id01))
        xmax = cc.argmax()
        ## Polyfit of degree 2 for three points, extremum
        c1 = (cc[xmax+1]-cc[xmax-1])/2.
        c2 = cc[xmax+1]-c1-cc[xmax]
        xmax = xmax - c1/c2/2.
        return  xmax - d0.shape[0]/2
        
    ## two dimensional case
    if d0.ndim == 2:    
        ## Manually Pad Arrays for DFT Optimization
        rows, cols = d0.shape
        nrows = cv2.getOptimalDFTSize(rows)
        ncols = cv2.getOptimalDFTSize(cols)
        d0n = np.zeros((nrows,ncols))
        d1n = np.zeros((nrows,ncols))
        d0n[:rows,:cols] = d0
        d1n[:rows,:cols] = d1
        ## Fourier Phase Correlation
        d0_dft = cv2.dft(np.float32(d0n), flags = cv2.DFT_COMPLEX_OUTPUT)
        d1_dft = cv2.dft(np.float32(d1n), flags = cv2.DFT_COMPLEX_OUTPUT)
        d01 = cv2.mulSpectrums(d0_dft,d1_dft,flags = 0, conjB = True)
        id01 = cv2.idft(d01, flags = cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        cc = np.fft.fftshift(np.abs(id01))
        indices = np.where(cc == cc.max())
        rmax = (indices[0])[0]
        cmax = (indices[1])[0]
        ## Interpolate to sub-pixel accuracy 
        if (rmax*cmax >= 0) and (rmax < d0n.shape[0]) and (cmax < d0n.shape[1]):
            ## interpolate to sub-pixel accuracy
            ## we use a quadratic estimator as in Tian and Huhns (1986) pg 222
            denom = 2.*(2.*cc.max() - cc[rmax+1,cmax] - cc[rmax-1, cmax])
            rfra = (rmax) + (cc[rmax+1,cmax] -cc[rmax-1,cmax])/denom
            denom = 2.*(2.*cc.max() - cc[rmax,cmax+1] - cc[rmax, cmax-1])
            cfra = (cmax) + (cc[rmax,cmax+1] -cc[rmax,cmax-1])/denom
            rmax = rfra
            cmax = cfra
        return np.array([rmax - d0n.shape[0]/2. , cmax - d0n.shape[1]/2.])
        
######################################################
  
if __name__ == "__main__":
    from idlpy import IDL
    import numpy as np 
    import time
    a  = np.random.rand(1024)
    b  = np.random.rand(1024)
    print "IDL 1d SHC return: ",IDL.shc(a,b) 
    print "Python 1d SHC numpy return:", shc_numpy(a,b)
    print "Python 1d SHC opencv return:", shc_opencv(a,b)
    aa = np.random.rand(1248,4238)
    aa[10:20,20:45] = 100.
    bb = np.random.rand(1248,4238)
    bb[15:25,21:46] = 100.
    print "IDL 2D SHC return:",IDL.shc(aa,bb)
    print "Python 2D SHC numpy return:",shc_numpy(aa,bb)
    print "Python 2D SHC opencv return:",shc_opencv(aa,bb)
    start = time.time()
    shc_numpy(aa,bb)
    end = time.time()
    print("numpy time:",end - start)
    start = time.time()
    shc_opencv(aa,bb)
    end = time.time()
    print("opencv time:",end - start)
    
