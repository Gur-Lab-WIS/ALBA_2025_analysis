#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### MODULE ###
import datetime
import os
import re
import subprocess
from collections import OrderedDict
from colorsys import hsv_to_rgb
from functools import reduce
from io import StringIO
from os.path import join as pjoin
from pathlib import Path
from pickle import dump, load
from time import sleep

import h5py as h5
import numpy as np
import pandas as pd
import plotly.express as px
import scipy
import scipy.io
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba as trgba
from more_itertools import collapse
from pandas import DataFrame as df
from PIL import Image
from pyperclip import copy as _copy
from scipy.interpolate import InterpolatedUnivariateSpline as uspline
from scipy.ndimage import gaussian_filter as gfilt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn import decomposition
from tqdm.auto import tqdm

palette = [trgba('blue'), trgba('orange'), trgba('red'), trgba('green'), trgba('purple')]


def LS(path, dirs=0, dpath=0):
    """returns np.array of items in path"""
    func = [lambda x: x.name, lambda x: Path(x.path)][dpath]
    if dirs:
        return np.array([func(x) for x in os.scandir(path) if x.is_dir()])
    return np.array([func(x) for x in os.scandir(path) if not x.is_dir()])


def l2r(ls):
    """utility to turn collections (such as array of objects) to arrays"""
    return np.array(list(ls))


def copy(x):
    """copies the objects string representation to the clipboard"""
    return _copy(str(x))


def concat_np(x, arr):
    """
    preforms elementwise addition of x to every element in arr.
    if rev, every element of arr to x.
    mainly good for string concatenation
    """
    rev = not isinstance(arr, np.ndarray)
    if not rev:
        kk = np.vectorize(lambda a: x + a)
    else:
        arr, x = x, arr
        kk = np.vectorize(lambda a: a + x)
    return kk(arr)


def list_concat(lst):
    """concatenates lists of lists to one list"""
    nlst = []
    lst = list(lst)
    for x in lst:
        for y in x:
            nlst.append(y)
    return nlst


def is_num(string):
    """checks if string is convertable to number"""
    try:
        float(string)
        return True
    except ValueError:
        return False


def sort_by_list(mov, fix):
    """sorts list mov according to list fix"""
    return [y for x, y in sorted(zip(fix, mov), key=lambda x: x[0])]


def precede(path, n=1):
    """returns the path after discarding the last n items (climbs up the tree n steps)."""
    if n <= 0:
        raise ValueError("non-positive precede")
    return pjoin(*Path(path).parts[:-n])


def recede(x, n=1, dpath=1):
    """removes first n parts of a path. if dpath is false, only returns the last component of the result."""
    if not n:
        return x
    spath = os.path.normpath(x).split(os.sep)
    if not dpath:
        return spath[n]
    return pjoin(*spath[n:])


def dblashes(x, rev=0):
    """doubles each backlash.
    if rev, halves each pair of backlashes"""
    x = str(x)
    if rev:
        return x.replace("\\\\", "\\")
    return x.replace("\\", "\\\\")


def istrue(x):
    """returns truthness of a value, or of a list or array of values. not sure how it will work on multidimensional (>2)"""
    x = np.array(x)
    shape = x.shape
    if len(shape) == 0:
        if x:
            if np.isnan(x):
                return False
            return True
        return False
    return np.array([istrue(a) for a in x.flatten()]).reshape(shape)


def qnorm(data):
    """returns array after normalizing in range [0,1]"""
    return plt.Normalize()(data).data


def hnorm(x):
    """subtracts the smallest value from all values in the collection"""
    tmp = x - np.nanmin(x)
    return tmp


def rtsv(path, f=0):
    """
    reads a  tsv formatted text file as a 1/2d np array.
    if f is truthy, reads the contents of f instead.
    """
    if f:
        return np.array(
            [x.split("\t") for x in f.split("\n") if len(x.split("\t")) > 1]
        )
    with open(path) as tmp:
        file = tmp.read()
    return rtsv(0, file)


def wtsv(y, path=0):
    """
    writes a 1/2d collection as a tsv formatted text file.
    """
    print(path)
    if path:
        with open(path, "w") as file:
            file.write(wtsv(y))
        return path
    return "\n".join(["\t".join(x) for x in y])


def btwn(x, a, b, include=0):
    """
    returns logical array such that a<x_ij<b.
    include : whether to preform <=, >= (True) or <, > (False)
    """
    if include:
        return np.logical_and(x >= a, x <= b)
    return np.logical_and(x > a, x < b)


def trimlist(x):
    """trims empty rows from a list"""
    if len(x):
        if len(str(x[-1])) == 0:
            return trimlist(x[:-1])
    return x


def trimlist_numer(ls):
    """
    trims non numerical rows from array
    """
    nums = []
    for x in ls:
        try:
            np.array(x).astype(float)
            nums.append(True)
        except ValueError:
            nums.append(False)
    return ls[nums].astype(float)


def monotone(x, verb=0, _start=1):
    """
    checks if an ordered collection is monotonic
    x : data
    verb : whether to ouput where it is not monotonic (result signage could be weird)
    _start : not to be used by user, for the function to track whether it is the first round or not.
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


def strsplit(x):
    """gets list of any iterable"""
    return list(x)


def flat_dir(path, dirs=0):
    """
    like LS but iteratively on all subdirectories
    dirs : whether to return file paths or directory paths.
    """
    return list_concat([[pjoin(x[0], y) for y in x[2 - dirs]] for x in os.walk(path)])


def phead(path):
    """return the name of the file, e.g. file.ext"""
    return Path(path).name


def headtail(head="", tail=""):
    """
    return a function that filters only text that begins with `head` and ends with `tail`
    leave either of them empty to search one side only.
    """
    lh = len(head)
    lt = len(tail)
    return lambda x: x[:lh] == head and x[len(x) - lt :] == tail


def filtl(*args, **kwargs):
    """
    preforms filter but returns as list instead of filter object
    """
    return list(filter(*args, **kwargs))


def mapl(*args, **kwargs):
    """
    preforms map but returns as list instead of map object
    """
    return list(map(*args, **kwargs))


def gaussian(x, a, b, c):
    """
    gaussian function
    x : x value
    a : peak height
    b : peak location
    c : peak width
    """
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))




def fwhm(x):
    """
    converts gaussian parametric width to FWHM value
    """
    return x * 2 * np.sqrt(2 * np.log(2))


es = np.arange(396, 405, 0.1)



def split_roiset(path):
    """
    splits a roiset.txt into roi#.txt files in the same directory.
    """
    y = rtsv(path)
    lists = [np.array([range(1, len(y) + 1), y[:, i]]).T for i in range(1, y.shape[1])]
    for i, z in enumerate(lists):
        wtsv(z, pjoin(precede(path), "roi" + str(i + 1) + ".txt"))
    return None


LEFT_PEAK = [399.5, 400.75]
RIGHT_PEAK = [400.75, 402.5]
SIGMA_PEAK = [404.0, 408.0]

def guan_ratios(e, z, ints=0, peaks=0):
    """
    replicates asher's script.
    filter's out different types of noises, logs them all on a graph and returns result.
    added to also return intensity of interesting peaks.
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


def save_filt(path):
    """given path with xanesmaps file in it, will save EJfilter and EJ images in same folder."""
    filt = path
    npath = [x for x in LS(path, dpath=1) if "XANES" in str(x)][0]
    mat = scipy.io.loadmat(npath)  # changed to pjoin
    filtim = (((mat["filter_edgeJump"] > 0) * 1)).astype(int)
    levelim = mat["EdgeJump"]
    im = Image.fromarray(filtim)
    im.save(pjoin(filt, "samplename_Edge_jump_filtery.tif"))  # changed to pjoin
    im = Image.fromarray(levelim)
    im.save(pjoin(filt, "samplename_Edge_jump_grays.tif"))  # changed to pjoin
    return None


def tif_stack(dirr):
    """given path, returns stack of all tifs in the folder as np stack."""
    paths = LS(dirr)
    #     files = [x for x if os.path.split(x)[1][-3:] == 'tif']
    paths = np.array(list(filter(lambda x: os.path.split(x)[1][-3:] == "tif", paths)))
    paths = concat_np(dirr + "\\", paths)
    imgs = []
    for path in paths:
        file = Image.open(path)
        imgs.append(np.array(file))
        file.close()
    imgs = np.array(imgs)
    return imgs


def show_stack(x):
    """show a stack inside python. stack should be in the format of a collection of images in np (np shape (x, y, z))"""
    fig = px.imshow(x, animation_frame=0, binary_string=True)
    return fig


def bulk_guan_rats(
    paths, retval=6, estart=396, estop=405, estep=0.1, clean_matrix=0, spline=True, verb = False
):
    """
    given roi*.txt paths, returns processed data of those files.
    paths : collection of paths containing the files
    retval : stage in asher_plot to return
    estart, estop : from where to where to return the spectra (in eV)
    estep : how much of a eV step to make along the x axis (there is interpolation)
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


def make_proc_macro(BC_cropped, save=1, close=1):
    """makes imageJ macro that performs basic transformations"""
    ejfilter = pjoin(precede(BC_cropped), "samplename_Edge_jump_filtery.tif")
    BCname = os.path.split(BC_cropped)[1]
    ejname = os.path.split(ejfilter)[1]
    npath = pjoin(precede(BC_cropped), "BC_cropped_filt")
    if save:
        print(npath)
        if not Path(npath).is_dir():
            os.mkdir(npath)
    macro = rf"""run("Image Sequence...", "open=[{dblashes(LS(BC_cropped, dpath = 1)[0])}] sort");
run("TIFF Virtual Stack...", "open='{dblashes(ejfilter)}'");
selectWindow("{ejname}");
selectWindow("{BCname}");
run("Log", "stack");
run("Multiply...", "value=-1 stack");
imageCalculator("Multiply create stack", "{BCname}", "{ejname}");
selectWindow("Result of {BCname}");"""
    if save:
        macro += "\n"
        macro += f"""run("Image Sequence... ", "format=TIFF name=BC_cropped_filt use save='{dblashes(npath)}'");"""
    if close:
        macro += "\n"
        macro += f"""close();
selectWindow("{BCname}");
close();
selectWindow("{ejname}");
close();"""
    mac_path = pjoin(precede(BC_cropped), "proc.ijm")
    with open(mac_path, "w") as mac:
        mac.write(macro)
    return Path(mac_path)


def make_proc_nofilt_macro(BC_cropped, save=0, projection="Median"):
    """same as make_proc_macro, but without multiplying by filter"""
    BCname = os.path.split(BC_cropped)[1]
    npath = pjoin(precede(BC_cropped), "BC_cropped_filt")
    if save:
        print(npath)
        if not Path(npath).is_dir():
            os.mkdir(npath)
    macro = rf"""run("Image Sequence...", "open=[{dblashes(LS(BC_cropped, dpath = 1)[0])}] sort");
selectWindow("{BCname}");
run("Log", "stack");
run("Multiply...", "value=-1 stack");"""
    if save:
        macro += "\n"
        macro += f"""run("Image Sequence... ", "format=TIFF name=BC_cropped_filt use save='{dblashes(npath)}'");"""

    if projection:
        macro += f"""\nrun("Z Project...", "projection={projection}");"""
    mac_path = pjoin(precede(BC_cropped), "proc.ijm")
    with open(mac_path, "w") as mac:
        mac.write(macro)
    return Path(mac_path)


# IMAGEJ_PATH = r'C:\Users\YONATABR\Downloads\ImageJ_Java16\ImageJ_Java16\ij.jar' #gurlab location
# IMAGEJ_PATH = r"C:\Users\yonatabr\Downloads\ImageJ\ij.jar" # dudi lab location
IMAGEJ_PATH = r"C:\TOOLS\ImageJ_Java16\ij.jar"  # gurlab server location


def run_ijm(mac_path, imagej_path=IMAGEJ_PATH):
    """runs imageJ macros"""
    return subprocess.Popen(
        ["Java", "-jar", imagej_path, "-macro", mac_path], start_new_session=True
    )


def get_roiz(roidir):
    """input is dir with RoiSet.zip in it, output is z values of all the rois."""
    bcfilt = os.path.split(roidir.split("proc")[0])[0]
    bcfilt = pjoin(bcfilt, "cropped", "BC_cropped_filt")
    roisetp = pjoin(roidir, "RoiSet")
    macro = f"""run("Image Sequence...", "open=[{dblashes(LS(bcfilt, dpath = 1)[0])}] sort");
    run("Set Measurements...", "mean redirect=None decimal=3");
    run("ROI Manager...");
    roiManager("Open", "{dblashes(roisetp)}.zip");
    roiManager("Deselect");
    roiManager("Multi Measur
    e");
    saveAs("Results", "{dblashes(roisetp)}.txt");
    run("Close");
    close();
    roiManager("Deselect");
    roiManager("Delete");"""
    mac_path = pjoin(roidir, "roi_extract_macro.ijm")
    with open(mac_path, "w") as file:
        file.write(macro)
    run_ijm(mac_path)
    sleep(20)
    file = rtsv(roisetp + ".txt")
    return file


def show_rois(dirr, roidir):
    """input is dir with RoiSet.zip in it, output is none, imagej opens with the rois."""
    filt = dblashes(pjoin(dirr, "cropped", "samplename_Edge_jump_grays.tif"))
    rois = filtl(headtail("", ".roi"), flat_dir(roidir))
    roiset = filtl(lambda x: headtail("RoiSet", ".zip")(phead(x)), flat_dir(roidir))
    roistr = "\n".join([f'roiManager("Open", "{dblashes(x)}");' for x in rois])
    roisetstr = "\n".join([f'roiManager("Open", "{dblashes(x)}");' for x in roiset])
    macro = f"""run("TIFF Virtual Stack...", "open=[{dblashes(filt)}]");
run("ROI Manager...");"""
    macro += roistr
    macro += roisetstr
    mac_path = pjoin(
        r"\\isi.storwis.weizmann.ac.il\Labs\dvirg\yonatabr\ijm", "see_rois.ijm"
    )
    with open(mac_path, "w") as file:
        file.write(macro)
    run_ijm(mac_path)
    return mac_path


tilts = r"""\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_0deg2\serie_0deg2
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_0deg2\serie_0deg2_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_0deg3\serie_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_0deg\serie_0deg
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_0deg\serie_0deg_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_10deg\serie_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_20deg\serie_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_30deg\serie_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_40deg\serie_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_50deg\serie_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230524\ALBA25_G4_72hpf\tilting\Escan_60deg\serie_OF""".split(
    "\n"
)

nrad = r"""\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230525\ALBA33_G4_72hpf\Escan\serieOF_tomo06
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230525\ALBA33_G4_72hpf\Escan\serieOF_tomo07
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230525\ALBA33_G4_72hpf\Escan\serieOF_tomo08
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230525\ALBA33_G4_72hpf\Escan\serieOF_tomo09
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230525\ALBA33_G4_72hpf\Escan\serieOF_tomo10""".split(
    "\n"
)


raddam = r"""\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie5
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie2_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie6_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie2
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie4
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie1_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie3
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie5_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie4_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie1
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie3_OF
\\isi.storwis.weizmann.ac.il\Labs\dvirg\zoharba\ALBA May 2023\20230523\ALBA25_G4_72hpf\rad_dam\serie6""".split(
    "\n"
)

def get_rois(path):
    return [str(x) for x in Path(path).rglob('**/roi*.txt') if 'RoiS' not in str(x)]

# Read z values per roi file (remove non numerical rows, empty rows, and only return value column)
def read_roi(path):
    """
    :param path: str, path to roi.txt file containins z values
    """
    tx = Path(path).read_text()
    first_row = pd.read_csv(StringIO(tx), sep='\t', nrows=1, header=None)
    if first_row.map(lambda x: isinstance(x, (str, bytes))).any().any():
        df = pd.read_csv(StringIO(tx), sep='\t', header=0)
    else:
        df = pd.read_csv(StringIO(tx), sep='\t', header=None)
    df = df.dropna(how = 'all', axis = 1)
    return df.iloc[:, -1].values

# Get e values per roi file
def tomo_energy(dirr, roi = True):
    """given folder with ali.hdf5 file, returns energies from scan"""
    if roi:
        dirr = precede(str(dirr).split("proc")[0])
    currfiles = LS(dirr, dpath=True)
    tiled_file = [x for x in currfiles if "ali" in str(x).rsplit("_", maxsplit = 1)[-1]][0]
    tmp = h5.File(tiled_file, libver="latest")
    e = np.array(tmp[list(tmp.keys())[0]]["energy"])
    tmp.close()
    return np.array(first_unique(e))


def first_unique(iterable):
    """Fix e values array (only keep first unique value, maintaining original order)"""
    return list(OrderedDict.fromkeys(iterable))

def fix_ens(e, z):
    """
    executes a dumb fix to energy scores when there is a mismatch in the data points of energy and intensity.
    simply finds the smaller of the two sets and trims the other so they fit.
    """
    flen = min(e.shape[0], z.shape[0])
    e, z = e[:flen], z[:flen]
    return e, z

def flipped_signal(e, z):
    """checks whether a roi z plot is flipped.
    the logic is that if the part of the plot before the sigma peak is higher than the mean, it is probably flipped.
    """
    return e, z
    try:
        if qnorm(z)[e < 397].mean() > 0.5:
            z *= -1
    except:
        z *= -1
    return e, z

def get_roi_ez(path, e = None):
    """Function that accepts roi file and return properly fixed e,z array"""
    path = str(path)
    z = read_roi(path)
    if e is None:
        e = tomo_energy(path)
    return flipped_signal(*fix_ens(e, z))

def remove_background_guan(e, z):
    """accepts e and raw z values, returns e, z after background removal"""
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
    """accepts e and backgrounded z values, returns e, z after arctan removal"""
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

def color_plots(x, y, c, cmap = cm.viridis, **kwargs):
    """Accept x,y values array and array, plot colored by array"""
    c = cmap(qnorm(c))
    for xx, yy, cc in zip(x, y, c):
        plt.plot(xx, yy, color = cc, **kwargs)

def find_peak(x, y, low, high, arg=0):
    """
    finds peaks in a given x, y landscape between x values low and high and returns either height or location.
    x, y : data
    low, high : boundaries
    arg : whether to return peak location instead of intensity"""
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
    e : x values
    z : y values
    llrng : left peak range
    rrrng: right peak range
    lb : lower bounds for estimator
    ub : upperbounds for estimator
    show : as fit_ppeaks
    estimates the peaks as seperate gaussians
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

def noerror(func, retval = None):
    """returns version of function that returns either the proper return value, or None if function fails"""
    def nfunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            return retval
    return nfunc
