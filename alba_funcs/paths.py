"""python functions to query and manipulate paths"""

import os
from os.path import join as pjoin
import numpy as np
from .utils import list_concat

def LS(path, dirs : bool=False, dpath : bool=False):
    """
    returns np.array of items in path
    args:
        path : path under which to search
        dirs : whether to return only directories (True) or only files(False)
        dpath : whether to return relative path or just file name
    returns:
        np.array of either filenames or relative paths
    """
    func = [lambda x: x.name, lambda x: Path(x.path)][dpath]
    if dirs:
        return np.array([func(x) for x in os.scandir(path) if x.is_dir()])
    return np.array([func(x) for x in os.scandir(path) if not x.is_dir()])

def precede(path, n=1):
    """
    returns the path after discarding the last n items (climbs up the tree n steps).
    args:
        path : path to trim
        n : how many coponents to trim
    returns:
        path after discarding n items from the end
    """
    if n <= 0:
        raise ValueError("non-positive precede")
    return pjoin(*Path(path).parts[:-n])


def recede(x, n=1, dpath: bool=True):
    """
    removes first n parts of a path. if dpath is false, only returns the last component of the result.
    args:
        path : path to trim
        n : how many coponents to trim
        dpath : whether to return whole path after trimming (default) or just file name
    returns:
        path after discarding n items from the start
    """
    if not n:
        return x
    spath = os.path.normpath(x).split(os.sep)
    if not dpath:
        return spath[n]
    return pjoin(*spath[n:])


def dblashes(x, rev: bool=0):
    """
    doubles each backslash.
    if rev, halves each pair of backlashes
    args:
        x : string on which to double backslashes
        rev : whther to double or halve the number of backslashes
    returns:
        modified x
    """
    x = str(x)
    if rev:
        return x.replace("\\\\", "\\")
    return x.replace("\\", "\\\\")

def flat_dir(path, dirs=0):
    """
    like LS but iteratively on all subdirectories
    args:
        path : path under which to search
        dirs : whether to return file paths or directory paths.
    returns:
        all files underneath (not only directly) path in a list
    """
    return list_concat([[pjoin(x[0], y) for y in x[2 - dirs]] for x in os.walk(path)])


def phead(path):
    """
    return the name of the file, e.g. file.ext
    args:
        path : input path
    returns:
        filename (last component of the path)
    """
    return Path(path).name

