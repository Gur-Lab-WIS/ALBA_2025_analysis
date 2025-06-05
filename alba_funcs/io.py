"""python functions for reading and writing files"""

import numpy as np
import os
import scipy
from PIL import Image
from os.path import join as pjoin
from pathlib import Path
import pandas as pd
import h5py as h5
from io import StringIO
from .paths import LS, precede
from .utils import concat_np, first_unique

def rtsv(path, f=0):
    """
    reads a  tsv formatted text file as a 1/2d np array.
    if f is truthy, reads the contents of f instead.
    args:
        path : path to file to read
        f : optionally input the contents as a string
    returns:
        tsv as list of lists of values
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
    args:
        y : list of lists (or similar) to write
        path : optional path to write to
    returns:
        either path where the data was written or the content as a string
    """
    print(path)
    if path:
        with open(path, "w") as file:
            file.write(wtsv(y))
        return path
    return "\n".join(["\t".join(x) for x in y])

def save_filt(path):
    """
    given path with xanesmaps file in it, will save EJfilter and EJ images in same folder.
    args:
        path : path to dir containing xanes map matlab file
    returns:
        None
    """
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
    """
    given path, returns stack of all tifs in the folder as np stack.
    args:
        dirr : path to directory containing tif files
    returns:
        np.array of all the tif files values together
    """
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

def get_rois(path):
    """
    finds all z value roi files below a directory
    args:
        path : path to directory under which to search
    returns:
        list of all the file paths
    """
    return [str(x) for x in Path(path).rglob('**/roi*.txt') if 'RoiS' not in str(x)]

# Read z values per roi file (remove non numerical rows, empty rows, and only return value column)
def read_roi(path):
    """
    reads the content of a roi value file
    args:
        path : str, path to roi.txt file containins z values
    returns:
        values as a np.array
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
def tomo_energy(dirr, roi: bool = True):
    """
    given folder with ali.hdf5 file, returns energies from scan
    args:
        dirr : path to folder containing hdf5 files from scan
        roi : whether the path is to the directory with the scans or the directory of the roi
    returns:
        np.array of energy values from the scan
    """
    if roi:
        dirr = precede(str(dirr).split("proc")[0])
    currfiles = LS(dirr, dpath=True)
    tiled_file = [x for x in currfiles if "ali" in str(x).rsplit("_", maxsplit = 1)[-1]][0]
    tmp = h5.File(tiled_file, libver="latest")
    e = np.array(tmp[list(tmp.keys())[0]]["energy"])
    tmp.close()
    return np.array(first_unique(e))

def split_roiset(path):
    """
    splits a roiset.txt into roi#.txt files in the same directory.
    args:
        path : path to roiset.txt file containing several roi values
    returns:
        None
    """
    y = rtsv(path)
    lists = [np.array([range(1, len(y) + 1), y[:, i]]).T for i in range(1, y.shape[1])]
    for i, z in enumerate(lists):
        wtsv(z, pjoin(precede(path), "roi" + str(i + 1) + ".txt"))
    return None