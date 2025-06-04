"""functions for signal analysis"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.interpolate import InterpolatedUnivariateSpline as uspline
from scipy.ndimage import gaussian_filter as gfilt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress, pearsonr, spearmanr
from .io import rtsv, read_roi, wtsv
from .paths import pjoin, precede
from .utils import btwn, strsplit

def qnorm(data):
    """
    returns array after normalizing in range [0,1]
    args:
        data : sequence of numeric values
    returns:
        array of the values linearly normalized to range [0,1]
    """
    return plt.Normalize()(data).data


def hnorm(x):
    """
    subtracts the smallest value from all values in the collection
    args:
        x : sequence of numeric values
    returns:
        np.array of values with minimal value subtracted
    """
    tmp = np.array(x) - np.nanmin(x)
    return tmp

def monotone(x, verb=0, _start=1):
    """
    checks if an ordered collection is monotonic
    args:
        x : data
        verb : whether to ouput where it is not monotonic (result signage could be weird)
        _start : not to be used by user, for the function to track whether it is the first round or not.
    returns:
        boolean, either is monotonic or isn't
    """
    if len(x) > 1:
        if x[-1] < x[-2] and _start:
            x = -np.array(x)
        if x[-1] > x[-2]:
            return monotone(x[:-1], verb, _start=0)
        if verb:
            print(x[-1], x[-2])
        return False
    return True

def gaussian(x: float, a: float, b: float, c: float):
    """
    gaussian function
    args:
        x : x value
        a : peak height
        b : peak location
        c : peak width
    returns:
        float value of the gaussian at value x
    """
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))




def fwhm(x: float):
    """
    converts gaussian parametric width to FWHM value
    args:
        x : gaussian parametric width
    returns:
        FWHM of the same guassin
    """
    return x * 2 * np.sqrt(2 * np.log(2))


es = np.arange(396, 405, 0.1)


LEFT_PEAK = [399.5, 400.75]
RIGHT_PEAK = [400.75, 402.5]
SIGMA_PEAK = [404.0, 408.0]

def guan_ratios(e, z, ints=0, peaks=0):
    """
    filter's out different types of noises, logs them all on a graph and returns result.
    can also return intensity of interesting peaks.
    args:
        param e : energy values
        param z : absorption values
        param ints : flag to return peak intensities
        param peaks : flag to print out energy value of peaks
    returns:
        list of energy and absorption levels at different processing steps, and optionally intensity levels
    """
    e, z = fix_ens(e, z)
    if flipped_signal(e, z):
        z = z * (-1)
    fig1 = z
    pree = np.where(btwn(e, 390, 397))[0]
    try:
        poly = np.polynomial.Polynomial(
            np.polynomial.chebyshev.chebfit(e[pree], fig1[pree], 1)
        )
    except Exception as exceptionE:
        print(e[pree])
        print(fig1[pree])
        raise exceptionE
    fig2 = poly(e)
    fig3 = fig1 - fig2

    # fit arctan to subtract hill
    exinds = np.where(
        np.logical_not(((e < 403) * (e > 399)) + ((e < 426) * (e > 406)))
    )[0]
    e1 = e[exinds]
    y1 = fig3[exinds]
    fig4 = [e1, y1]
    atanf = (
        lambda x, *b: b[0] * np.arctan(b[1] * (x - b[2])) - b[3]
    )  # function to fit to
    b1 = [1, 1, 405, 0.1]  # initial parameters
    try:
        b = scipy.optimize.curve_fit(atanf, e1, y1, b1, ftol=6e-7, maxfev=int(1e6))[
            0
        ]  # optimized parameters
    except:
        return [0]
    atan_fitted = (
        lambda x: b[0] * np.arctan(b[1] * (x - b[2])) - b[3]
    )  # fitted curve function
    fig5 = atan_fitted(e)
    fig6 = fig3 - fig5

    # find peak ratio
    left = find_peak(e, fig6, LEFT_PEAK[0], LEFT_PEAK[1])  # CHANGED TO FIND_PEAK
    right = find_peak(e, fig6, RIGHT_PEAK[0], RIGHT_PEAK[1])
    if peaks:
        leftpeak = (e[np.where(fig6 == left)[0]][0], left)
        rightpeak = (e[np.where(fig6 == right)[0]][0], right)
        print(leftpeak)
        print(rightpeak)
    ratio = right / left
    if ints:
        ints = np.mean([right, left])
        return [ratio, ints, fig1, fig2, fig3, fig4, fig5, fig6, e1, y1]
    return [ratio, fig1, fig2, fig3, fig4, fig5, fig6, e1, y1]

def bulk_guan_rats(
    paths: list, retval: int = 6, estart: float = 396.0, estop : float = 405, estep : float = 0.1, clean_matrix: int = 0, spline: bool = True, verb: bool = False
):
    """
    given roi*.txt paths, returns processed data of those files.
    args:
        paths : collection of paths containing the files
        retval : stage in asher_plot to return
        estart, estop : from where to where to return the spectra (in eV)
        estep : how much of a eV step to make along the x axis (there is interpolation)
    returns:
        either tuple (clean roi values, paths to rois) or a matrix containing interpolated clean values from the rois
    """
    # collect roi.txt files
    roi_paths = np.array(filtl(lambda x: headtail("roi", ".txt")(phead(x)), paths))
    # get corresponding energies
    roi_energies = [tomo_energy(x.split("proc")[0]) for x in roi_paths]
    # get corresponding z values
    z_vals = [trimlist(rtsv(x)[1:, 1]).astype(float) for x in roi_paths]
    # fix non matching energy and z values
    for i, x in enumerate(z_vals):
        roi_energies[i], z_vals[i] = fix_ens(roi_energies[i], z_vals[i])
    filler = [1 for _ in roi_energies]  # filler to force shape of energy_z_zip
    # zip energies and z values together
    energy_z_zip = np.array([roi_energies, z_vals, filler], dtype=object)[:-1].T
    # predefined final energy values
    energies = np.arange(start=estart, stop=estop, step=estep)
    results = []
    for i, energy_z in enumerate(energy_z_zip):
        # check there are any values in the roi before calculation
        if len(energy_z[1]):
            try:
                # get stages of preprocessing and ratio
                current_res = guan_ratios(energy_z[0], energy_z[1])
                # save current result in results list, based on retval
                results.append([current_res[0], current_res[retval], energy_z[0]])
                energy = results[-1][2]
                z = results[-1][1]
                if spline:
                    # sort and spline energy and z values to predefined energies
                    ezs = np.array(
                        sorted(np.array([energy, z]).T, key=lambda x: x[0])
                    ).T
                    splined_z = uspline(ezs[0], ezs[1], ext = 1)(energies)
                    # save splined values instead of original values
                    results[-1][1] = splined_z
                    results[-1][2] = energies
            except Exception as exception:
                if verb:
                    print(exception)
                results.append(
                    [0, 0, 0]
                )  # save bogus result to be easily found downstream without breaking analysis
        else:  # no z values
            results.append([0, 0, 0])
    if clean_matrix:
        mat = np.array(
            [
                list(x)
                for x in np.array(results, object)[:, 1:]
                if isinstance(x[0], np.ndarray)
            ]
        )
        return mat  # matrix of values only, vector per sample
    return (
        results,
        roi_paths,
    )  # tuple, like ([ratio, requested_stage_z_vals, energies], paths)

def fix_ens(e, z):
    """
    executes a dumb fix to energy scores when there is a mismatch in the data points of energy and intensity.
    simply finds the smaller of the two sets and trims the other so they fit.
    args:
        e : np.array of energy values
        z : np.array of absorption values
    returns:
        tuple of (e, z) after the lengths have been modified
        
    """
    flen = min(e.shape[0], z.shape[0])
    e, z = e[:flen], z[:flen]
    return e, z

def flipped_signal(e, z, ignore : bool = True):
    """
    checks whether a roi z plot is flipped.
    the logic is that if the part of the plot before the sigma peak is higher than the mean, it is probably flipped.
    args:
        e : np.array of energy values
        z : np.array of absorption values
        ignore : flag, whether to infer if the values are flipped or simply return args as is. defaults to ignore.
    returns:
        tuple of modified (e, z)
    """
    if ignore:
        return e, z
    try:
        if qnorm(z)[e < 397].mean() > 0.5:
            z *= -1
    except:
        z *= -1
    return e, z

def get_roi_ez(path, e = None):
    """
    Function that accepts roi file and return properly fixed e,z array
    args:
        path : path to roi file to infer
        e : optionally input known energy values
    returns:
        tuple of (e, z) after equalizing lengths and ensuring the values aren't flipped
    """
    path = str(path)
    z = read_roi(path)
    if e is None:
        e = tomo_energy(path)
    return flipped_signal(*fix_ens(e, z))

def remove_background_guan(e, z):
    """
    accepts e and raw z values, returns e, z after background removal
    args:
        e : energy values
        z : absorption values. the values should be of equal length
    returns:
        tuple of (e, z3) where z3 is the z values after removing background
    """
    pree = np.where(btwn(e, 390, 397))[0]
    try:
        poly = np.polynomial.Polynomial(
            np.polynomial.chebyshev.chebfit(e[pree], z[pree], 1)
        )
    except Exception as exceptionE:
        print(e[pree])
        print(z[pree])
        print(exceptionE)
        return e, None
    z2 = poly(e)
    z3 = z - z2
    return e, z3

def remove_arctan_guan(e, z3):
    """
    accepts e and backgrounded z values, returns e, z after arctan removal
    args:
        e : energy values
        z : absorption values post background removal
    returns:
        tuple of (e, z6) where z6 is z values after removing background and arctan
    """
    # fit arctan to subtract hill
    if z3 is None:
        return e, None
    exinds = np.where(
        np.logical_not(((e < 403) * (e > 399)) + ((e < 426) * (e > 406)))
    )[0]
    e1 = e[exinds]
    y1 = z3[exinds]
    z4 = [e1, y1]
    atanf = (
        lambda x, *b: b[0] * np.arctan(b[1] * (x - b[2])) - b[3]
    )  # function to fit to
    b1 = [1, 1, 405, 0.1]  # initial parameters
    try:
        b = scipy.optimize.curve_fit(atanf, e1, y1, b1, ftol=6e-7, maxfev=int(1e6))[
            0
        ]  # optimized parameters
    except:
        return e, None
    atan_fitted = (
        lambda x: b[0] * np.arctan(b[1] * (x - b[2])) - b[3]
    )  # fitted curve function
    z5 = atan_fitted(e)
    z6 = z3 - z5
    return e, z6

def find_peak(x, y, low : float, high : float, arg : bool=0):
    """
    finds peaks in a given x, y landscape between x values low and high and returns either height or location.
    args:
        x, y : data
        low, high : boundaries in x dimension
        arg : whether to return peak location instead of intensity
    returns:
        either the argument where y value is maximal, or the maximal value
    """
    if arg:
        return np.argmax(y[btwn(x, low, high)])
    return max(y[btwn(x, low, high)])

def fit_sppeaks(
    e,
    z,
    show="000",
    llrng=(398, 400.3),
    rrrng=(400.5, 402),
    lb=(0, 396, 0.1),
    ub=(0.8, 404, 1.5),
):
    """
    fit pi peaks to a gaussian each, and optionally plot them.
    args:
        e : x values
        z : y values
        llrng : left peak range
        rrrng: right peak range
        lb : lower bounds for estimator
        ub : upperbounds for estimator
        show : describes what to plot, each digit being either 0 or 1 like booleans. 
        first digit is for original data
        second is for left peak gaussian
        third is for right peak gaussian
    returns:
        tuple of peak parameters of left and right prepeaks.
    """
    #     fit seperate gaussian peaks to the pi* prepeaks
    lrng = btwn(e, *llrng)
    rrng = btwn(e, *rrrng)
    e, z = e, z
    try:
        lpopt, _ = curve_fit(
            gaussian, e[lrng], z[lrng], [z[lrng].mean(), np.mean(llrng), 1], bounds=(lb, ub)
        )
    except Exception as err:
        print(err, 'lpopt')
        lpopt = (np.nan,)*3
    try:
        rpopt, _ = curve_fit(
            gaussian, e[rrng], z[rrng], [z[rrng].mean(), np.mean(rrrng), 1], bounds=(lb, ub)
        )
    except Exception as err:
        print(err, 'rpopt')
        rpopt = (np.nan,)*3
    show = show + "0" * (4 - len(show))
    show = list(map(int, strsplit(show)))
    if show[0]:
        plt.plot(e, z)
    if show[1]:
        plt.plot(e, gaussian(e, *lpopt))
    if show[2]:
        plt.plot(e, gaussian(e, *rpopt))
    return lpopt, rpopt

