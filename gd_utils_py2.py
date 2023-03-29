def save_dict(file,**kwargs):
    """
    Save the value of some data in a file.
    Usage: save('misdatos.pypic',a=a,b=b,test=test)
    """
    import pickle
    f=open(file,"wb")
    pickle.dump(kwargs,f,protocol=2)
    f.close


def restore_dict(file):
    """
    Read data saved with save function.
    Usage: datos = restore('misdatos.pypic')
    """
    import pickle
    f=open(file,"rb")
    result = pickle.load(f)
    f.close
    return result


def d_shiftscale(data_to_fit, median_profile, MEDI_DERIV = None, WEIGHTS = None, SMTH = None, TERMS = 2):
    """
    Fit an array of data points with a function f by assuming the data is well fit by a function of the type
    F(x) = c1 * f(x) + c2 * f'(x) + c3.

    The parameter TERMS controls which componenents to include in the fit.
    TERMS = 2 includes (c1, c2) terms in fit
    TERMS = 3 includes (c1, c2, c3) terms in fit
    TERMS = 4 includes (c1, c3) terms in fit
    """

    import numpy as np
    from astropy.convolution import Box1DKernel
    import numpy.linalg

    # Make sure arrays are floats.
    data_to_fit = np.asarray(data_to_fit, dtype=float)
    median_profile = np.asarray(median_profile, dtype=float)

    # Number of pixels.
    numpix = len(data_to_fit)

    # If the user does not specify a weights array, set all weights = 1
    if WEIGHTS == None:
        WEIGHTS = np.ones(numpix)

    # Calculate a smoothed defivative of the median_profile function, unless it is provided.
    if MEDI_DERIV == None:
        if SMTH == None:
            MEDI_DERIV = np.gradient(median_profile)
        else:
            MEDI_DERIV_TMP = np.convolve(np.gradient(median_profile), Box1DKernel(SMTH), mode='same')
            # The edges of the array are affected by edge effects.
            # Replace the affected pixels with the original values.
            MEDI_DERIV_TMP[0: SMTH/2] = MEDI_DERIV[0: SMTH/2]
            MEDI_DERIV_TMP[numpix - SMTH/2: numpix] = MEDI_DERIV[numpix - SMTH/2: numpix]

    # Calculate the fit to the data.
    # terms == 2 I have profile and derivative
    # terms == 3 I have profile, derivative and constant term
    # terms == 4 I have profile and constant term
    if TERMS == 2:
        # Calculate the least-square coefficients
        D = [ sum(data_to_fit * median_profile * WEIGHTS), sum(data_to_fit * MEDI_DERIV * WEIGHTS)  ]
        D = np.matrix(D, dtype = float)

        C = [ [ sum(median_profile * median_profile * WEIGHTS**2), sum(median_profile * MEDI_DERIV * WEIGHTS**2) ] , \
              [ sum(median_profile * MEDI_DERIV     * WEIGHTS**2), sum(MEDI_DERIV    * MEDI_DERIV * WEIGHTS**2) ]   ]
        C = np.matrix(C, dtype = float)

        # Calculate the c1 and c2 values
        c_vec = numpy.linalg.inv(C) * D.transpose()

        # Calculate the profile
        fit_profile = c_vec[0] * median_profile + c_vec[1] * MEDI_DERIV
        fit_profile = np.asarray(fit_profile)
        fit_profile = fit_profile.flatten()

    elif TERMS == 3:
        # Calculate the least-square coefficients
        D = [ sum(data_to_fit * median_profile * WEIGHTS), \
              sum(data_to_fit * MEDI_DERIV     * WEIGHTS), \
              sum(data_to_fit *                  WEIGHTS)  ]
        D = np.matrix(D, dtype = float)

        C = [ [ sum(median_profile * median_profile * WEIGHTS**2), sum(median_profile * MEDI_DERIV * WEIGHTS**2), sum(median_profile * WEIGHTS) ] , \
              [ sum(median_profile * MEDI_DERIV     * WEIGHTS**2), sum(MEDI_DERIV     * MEDI_DERIV * WEIGHTS**2), sum(MEDI_DERIV     * WEIGHTS) ] , \
              [ sum(median_profile *                  WEIGHTS   ), sum(MEDI_DERIV     * WEIGHTS                ), sum(                 WEIGHTS) ]  ]
        C = np.matrix(C, dtype=float)

        # Calculate the c1, c2 and c3 values
        c_vec = numpy.linalg.inv(C) * D.transpose()

        # Calculate the profile
        fit_profile = c_vec[0] * median_profile + c_vec[1] * MEDI_DERIV +c_vec[2]
        fit_profile = np.asarray(fit_profile)
        fit_profile = fit_profile.flatten()

    elif TERMS == 4:
        # Calculate the least-square coefficients
        D = [ sum(data_to_fit * median_profile * WEIGHTS), sum(data_to_fit * WEIGHTS)  ]
        D = np.matrix(D, dtype = float)

        C = [ [ sum(median_profile * median_profile * WEIGHTS**2), sum(median_profile * WEIGHTS) ] , \
              [ sum(median_profile *                  WEIGHTS   ), sum(                 WEIGHTS) ]   ]
        C = np.matrix(C, dtype = float)

        # Calculate the c1 and c3 values
        c_vec = numpy.linalg.inv(C) * D.transpose()

        # Calculate the profile
        fit_profile = c_vec[0] * median_profile + c_vec[1]
        fit_profile = np.asarray(fit_profile)
        fit_profile = fit_profile.flatten()

    return fit_profile, c_vec


def gaussian(x, a, b, c):
    """
    Define a Gaussian function to use during fitting.
    :param x: independent variable
    :param a: amplitude
    :param b: center offset
    :param c: standard deviation
    :return val: value at the required position
    """
    import numpy as np

    val = a * np.exp(-(x - b)**2 / (2.0 * c**2))
    return val


def reject_outliers(data, m = 2.):
    """
    Reject outlying data from a sample data
    :param data: list of numberst to reject from.
    :param m: how far from a sigma deviation to go
    """
    import numpy as np

    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]


def itos(int_val, no_elements):
    """
    Generate a string of the type 0002 where 2 is an example int_value and
    no_elements is 4 in this case. Basically generate a sting padded with 0's and
    containing the value int_val on the far right
    """
    import numpy as np

    # Make sure int_val is integer
    int_val = int(int_val)

    final_s = ''

    if int_val < 10:
        for i in range(0,no_elements-1):
            final_s = final_s + '0'
        final_s = final_s + str(int_val)
    elif int_val < 100:
        for i in range(0,no_elements-2):
            final_s = final_s + '0'
        final_s = final_s + str(int_val)
    elif int_val < 1000:
        for i in range(0,no_elements-3):
            final_s = final_s + '0'
        final_s = final_s + str(int_val)

    return final_s


class TestApp(object):

    def __init__(self, master, *args):
        import Tkinter as Tk
        self.master = master
        master.title("A Simple GUI")

        self.label = Tk.Label(master, text="This is 1st GUI")
        self.label.grid(row=0, column=0, columnspan=2, sticky='ewns')

        self.greet_button = Tk.Button(master, text='Greet', command=args[0])
        self.greet_button.grid(row=1, column=0, sticky='ewns')

        self.close_but = Tk.Button(master, text='Close', command=self._quit)
        self.close_but.grid(row=1, column=1, sticky='ewns')

    def _quit(self):
        """
        Cleanly exists the program and closes the GUI.
        """
        self.master.quit()     # stops mainloop
        self.master.destroy()  # this is necessary on Windows to prevent
                               # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    #def greet():
    #    print 'hello'


def combine_logs(dir_path):
    """
    Combine the log files located in the folder dir_path
    """

    import os
    from os import listdir
    import natsort
    from datetime import date
    import numpy as np

    # ~~~ Read all the files located in the folder ~~~
    # Check if the file_path exists.
    if os.path.exists(dir_path):
        # Quick check if the directory path contains a "/" at the end.
        if not (dir_path[-1] == "/"):
            dir_path += "/"

        filenames = natsort.natsorted(listdir(dir_path))

    else:
        sys.exit("That directory does not exist.")

    # Select only filenames that have the filetype .log
    for i, f_name in enumerate(filenames):
        if f_name.split('.')[-1] != 'log':
            filenames.pop(i)

        if f_name == 'master_log.txt':
            filenames.pop(i)

    # Sort the filenames
    all_files = np.empty(len(filenames), dtype=[('fname', 'S15'), ('days', 'S10')])
    all_files[:]['fname'] = filenames

    for i, f_name in enumerate(all_files['fname']):
        # Split the date into pieces.
        mm_dd_yyyy = f_name.decode().split('.')[0].split('_')
        mm = int(mm_dd_yyyy[0])
        dd = int(mm_dd_yyyy[1])
        yyyy = int(mm_dd_yyyy[2])

        d0 = date(2010, 1, 1)
        d1 = date(yyyy, mm, dd)
        delta = d1 - d0

        # Calculate the number of days.
        all_files[i]['days'] = delta.days

    all_files.sort(order=['days'])
    filenames = all_files[::-1]['fname']

    # Overwrite existing master file
    master_log = open(dir_path + 'master_log.txt', 'w')

    # ~~~ Loop over each filename and add to a master log file ~~~
    for i, f_name in enumerate(filenames):
        f_name = f_name.decode()
        # Check if the file is a valid log file by its suffix.
        if f_name.split('.')[-1] == 'log':
            # Open the file for reading
            f_file = open(dir_path + f_name, 'r')
            # Write the date of the log and one space
            master_log.write('############## ' + f_name.split('.')[0] + ' ############## \n\n')
            # Write the lines in the logs
            for line in f_file:
                master_log.write(line)
            # Add a new line if needed.
            #if line != '\n':
            #    master_log.write('\n')
            # Close the log file.
            master_log.write('\n\n')
            f_file.close()

    master_log.close()


def parse_folder_old(dir_path):
    """
    Combine all the files names in a folder into a single file called file_descriptions.txt
    """

    import os
    from os import listdir
    import natsort

    # ~~~ Read all the files located in the folder ~~~
    # Check if the file_path exists.
    if os.path.exists(dir_path):
        filenames = natsort.natsorted(listdir(dir_path))

        # Quick check if the directory path contains a "/" at the end.
        if not (dir_path[-1] == "/"):
            dir_path += "/"
    else:
        sys.exit("That directory does not exist.")

    # If the master file already exists then append to it.

    if os.path.exists(dir_path+'file_descriptions.txt'):
        master_log = open(dir_path+'file_descriptions.txt', 'a')
    else:
        master_log = open(dir_path+'file_descriptions.txt', 'w')

    # ~~~ Loop over each filename and add the name to the file_descriptions.txt
    for i, f_name in enumerate(filenames):
        master_log.write(f_name + '\n')

    master_log.close()


def parse_folder(dir_path="", file_ext = 'all'):
    """
    Combine all the files names in a folder into a single file called file_descriptions_[extension].txt
    where [extension] is added if only a particular type of files are targeted.
    If the keyword file_ext is given then use it to only parse files with that name.
    """

    import os
    from os import listdir
    import natsort
    import tkMessageBox
    import sys

    # ~~~ Determine the types of files parsed ~~~ #
    if file_ext == 'all':
        output_fname = 'file_descriptions.txt'
    else:
        output_fname = 'file_descriptions_' + file_ext + '.txt'

    # ~~~ Read all the files located in the folder ~~~ #
    # Check if the directory exists.
    if dir_path == "":
        # Select the current working directory if no directory is specified.
        dir_path = os.getcwd()
    elif not os.path.exists(dir_path):
        sys.exit("That directory does not exist.")

    # Quick check if the directory path contains a "/" at the end.
    if not (dir_path[-1] == "/"):
        dir_path += "/"

    # Find all the filenames in the directory.
    filenames = natsort.natsorted(listdir(dir_path))

    # ~~~ Determine if the output file already exists ~~~ #

    # Ask if to overwrite the file or append to it.
    if os.path.exists(dir_path + output_fname):
        result_ask = tkMessageBox.askyesno(title=" already exists", message=output_fname + ' already exists. Overwrite?' +
                                           ' Default is appending to file.')
        if result_ask:
            # User wants to overwrite the file entirely.
            master_log = open(dir_path + output_fname, 'w')
        else:
            # Default is to
            master_log = open(dir_path + output_fname, 'a')
            master_log.write('\n')
    else:
        # File never exists initially so just create it.
        master_log = open(dir_path + output_fname, 'w')

    # ~~~ Loop over filenames ~~~ #

    # Make a decision based on the file extension selected.
    if file_ext == 'all':
        # Add just the filenames to the complete file.
        for i, f_name in enumerate(filenames):
            master_log.write('# ' + f_name + '\n')

    elif file_ext == 'py':
        # If parsing python files look for the comment string at the top of each file.
        # If not found indicate that it is missing

        for i, f_name in enumerate(filenames):
            if f_name.split('.')[-1] == file_ext:
                master_log.write('# ' + f_name + '\n\n')

                # Open the file and read the first line.
                with open(dir_path + f_name, 'r') as f:
                    first_line = f.readline().strip('\n')

                end_comment = False
                if first_line == '"""':
                    with open(dir_path + f_name, 'r') as f:
                        for lines in f:
                            lines = lines.strip('\n')
                            master_log.write(lines + '\n')
                            if lines == '"""' and end_comment == False:
                                end_comment = True
                            elif lines == '"""' and end_comment == True:
                                master_log.write('\n')
                                break

    # Close the file.
    master_log.close()


def calc_disc_calibration(obs, cal, time_cal, time_obs, radius_cal, radius_obs):
    """
    Calculate the observation intensity in millionths of the disc intensity according to the formula
    Calibrated_int = (obs / cal) * (time_cal / time_obs) * (radius_cal / radius_obs)**2
    :param obs: Observation intensity, can also be an array.
    :param cal: Disc intensity, can also be an array but must have the same size as obs.
    :param time_cal: Disc exposure time
    :param time_obs: Observation exposure time
    :param radius_cal: Calibration exposure aperture radius
    :param radius_obs: Observation exposure aperture radius
    :return:
    """

    cal_int = (float(obs) / float(cal)) * \
              (float(time_cal) / float(time_obs)) * \
              (float(radius_cal) / float(radius_obs))**2

    print cal_int
    print "Calibrated intensity is: ", cal_int / 1.e-6


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


def shc_numpy(d0,d1, tol=None, maxval=False):

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

    GD 08/27/2018 : Return both the index and maximum correlation value by setting maxval=True
                    tol gives the pixel window over which to search the maximum correlation location.
    '''

    import numpy as np

    d0 = d0 - d0.mean()
    d1 = d1 - d1.mean()

    if d0.shape != d1.shape:
        raise ValueError("Input shapes do not match")

    if d0.ndim > 2:
        raise ValueError("Only 1-d or 2-d data supported")

    ## one dimensional case
    if d0.ndim == 1:
        cc = np.fft.fftshift(np.abs(np.fft.ifft(np.fft.fft(d0).conjugate() * np.fft.fft(d1))))
        cc_len = cc.shape[0]  # length of the cross-correlation vector.

        if tol is None:
            xmax = cc.argmax()
        else:
            tol = np.int(tol)
            xmax = cc[cc_len/2 - tol:cc_len/2 + tol].argmax() + cc_len/2 - tol

        #'''
        ## Polyfit of degree 2 for three points, extremum
        # ax**2 + bx + c = f(x)
        c = cc[xmax]  # constant
        b = (cc[xmax+1]-cc[xmax-1])/2.  # First deg coeff
        a = cc[xmax+1]-b-c      # Second deg coeff
        xmax = xmax - b/(2.*a)
        fxmax = c - b**2 / (4 * a)
        if maxval:
            return  xmax - d0.shape[0]/2, fxmax
        else:
            return xmax - d0.shape[0]/2
        #'''
        '''
        c1 = (cc[xmax+1] - cc[xmax-1])/2.0
        c2 = cc[xmax+1] - c1 - cc[xmax]
        xmax = xmax - c1/(2*c2)

        if maxval:
            return  xmax - d0.shape[0]/2, 10
        else:
            return xmax - d0.shape[0]/2
        '''

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


def calc_wln(wav):
    """
    Calculate the wavelength in air in AA when inputing wavenumbers in cm-1.
    Based on parameters from Ciddor 1996.
    See also note by http://www.as.utexas.edu/~hebe/apogee/docs/air_vacuum.pdf
    :param wav: Array of values in cm-1
    :return wln: Array of values in AA
    """

    b1 = 5.792105e-2 * 1.e16
    b2 = 1.679170e-3 * 1.e16
    c1 = 238.0185 * 1.e16
    c2 = 57.362 * 1.e16

    wln_air = 1.e8 / (wav * (1.0 + b1 / (c1 - wav**2) + b2 / (c2 - wav**2)))

    return wln_air


def calc_wln_energy(wav):
    """
    For a given wavenumber value calculate the corresponding eV value and associated wavelength.
    :param wav: Wavenumber value for the energy.
    :return energy_ev: Energy in eV
    :return lam: Wavelength of light in AA.
    """

    import scipy.constants as sc

    # One cm^-1 = 0.000123986 eV
    energy_ev = wav * 0.000123986

    # The wavelength associated with this energy is lam = hc/E
    lam = (sc.h * sc.c) / (energy_ev * sc.eV) / 1.e-10

    return energy_ev, lam


def parse_command_file(input_file, output_file):
    """
    Read the FORWARD input_file gui_command_log.txt from the current directory and
    write out the expanded version of the file to make identification easier.
    :return:
    """

    import os

    # ~~~ Load the file ~~~
    filename_in = os.getcwd() + '/' + input_file

    f_unit = open(filename_in, 'r')
    content = f_unit.readlines() # Read the file input.

    # Check if the file was created by FORWARD in which case it will consist of two lines.
    if len(content) == 1:
        content_split = content[0].split(',')
    elif len(content) == 2:
        content_split = content[1].split(',')

    f_unit.close()

    # ~~~ Create new file to hold the separated entries ~~~
    filename_out = os.getcwd() + '/' + output_file

    # Write the information in the new file.
    f_unit = open(filename_out,'w')

    # Check if the file was created by FORWARD in which case it will consist of two lines.
    if len(content) == 2:
        f_unit.write(content[0])

    # Write all the lines to the new file.
    for i in content_split:
        f_unit.write(i + '\n')

    f_unit.close()


def save_figure(image_display, plot_dir, plot_name, format_type=[], dpi=500, silent=True,
                record=None):
    """
    Save the information in the pointer image_display to the folder plot_dir.
    :param image_display: Pointer to plot for saving.
    :param plot_dir:  Directory for saving the plots.
    :param plot_name: Name string to save the figure as.
    :param out_type: List with the desired output formats. Default is pdf,png,eps,jpg
    :param dpi:  Desired resolution. Default is 500 since it's pretty good.
    :param silent: Determine if function informs the user about the data saved.
    :param record: Name of the script that produced the PDF or png recorded as Author in the metadata
    :return:
    """

    if len(format_type) == 0:
        # Default state.
        image_display.savefig(plot_dir + plot_name + '.pdf', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        image_display.savefig(plot_dir + plot_name + '.png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        image_display.savefig(plot_dir + plot_name + '.eps', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        image_display.savefig(plot_dir + plot_name + '.jpg', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        format_type = ['pdf', 'png', 'eps', 'jpg']
    else:
        for i, f_out in enumerate(format_type):
            if f_out == 'pdf' and record is not None:
                image_display.savefig(plot_dir + plot_name + '.' + f_out, dpi=dpi, bbox_inches='tight',
                                      pad_inches=0.1, metadata={'Author': record})
            elif f_out == 'png' and record is not None:
                image_display.savefig(plot_dir + plot_name + '.' + f_out, dpi=dpi, bbox_inches='tight',
                                      pad_inches=0.1, metadata={'Author': record})
            else:
                image_display.savefig(plot_dir + plot_name + '.' + f_out, dpi=dpi, bbox_inches='tight',
                                      pad_inches=0.1)


    if not silent:
        print 'Saved plot ' + plot_name + ' as: ', format_type


def print_out(prt_txt, out_file=None):
    """
    Output text to both stdout and a file.
    :param prt_txt: Array of elements to print
    :param out_file: Full path of the file to output to.
    :return:
    """

    import os

    # Default filename.
    if out_file is None:
        out_file = '/home/gabriel/Desktop/2015_SOLARC/output_test.log'

    # Stuff to print
    out_string = prt_txt[0]
    for i in prt_txt[1:]:
        out_string = out_string + ' ' + str(i)

    info_output_file = open(out_file, 'a')

    print out_string
    print >>info_output_file, out_string

    info_output_file.close()


def lam_vac_air(lam_vac):
    """
    Convert wavelengths from vacuum to air.
    Based on parameters from Ciddor 1996.
    See also note by http://www.as.utexas.edu/~hebe/apogee/docs/air_vacuum.pdf
    :param lam_vac:
    :return:
    """

    b1 = 5.792105e-2
    b2 = 1.679170e-3
    c1 = 238.0185
    c2 = 57.362

    lam_air = lam_vac / (1.0 + b1 / (c1 - 1 / lam_vac**2) + b2 / (c2 - 1 / lam_vac**2))

    return lam_air
