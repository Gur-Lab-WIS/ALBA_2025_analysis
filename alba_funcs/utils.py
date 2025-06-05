"""general utilities to be used within this analysis"""

import numpy as np
from collections import OrderedDict
from pyperclip import copy as _copy

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

def strsplit(x):
    """gets list of any iterable"""
    return list(x)

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

def first_unique(iterable):
    """Fix e values array (only keep first unique value, maintaining original order)"""
    return list(OrderedDict.fromkeys(iterable))

def noerror(func, retval = None):
    """returns version of function that returns either the proper return value, or None if function fails"""
    def nfunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            return retval
    return nfunc
