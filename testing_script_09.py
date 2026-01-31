# -*- coding: UTF-8 -*-
"""
:Script:   testing_script_09.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2018-10-13
:
:Purpose:
:
:Functions list .........
:...... np functions .....
:    num_121()  # 2D array to xy_val
:    num_122()  # Do an equation expansion
:    num_123()  # Reshape and transpose columns using einsum
:    num_124()  # Running count of values in a 1D array
:    num_125()  # combine_dicts(ds)
:    num_126()  # prime number calculator
:    num_127()  #
:    num_128()  #
:    num_129()  #
:    num_130()  #
:Notes:
:
:References
:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import os
import numpy as np
from textwrap import dedent, indent
import string
#import arraytools as art
# import datetime
# import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=100, precision=1,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]  # print this should you need to locate the script

pr_opt = np.get_printoptions()
df_opt = ", ".join(["{}={}".format(i, pr_opt[i]) for i in pr_opt])

# ---- functions ----

# ----------------------------------------------------------------------
# num_121 2D array to xy_val
def num_121():  # 2D array to xy_val
    """Convert an array to r, c, value format for 2D, 3D, n-D arrays

    `<https://stackoverflow.com/questions/46135070/generalise-slicing-
    operation-in-a-numpy-array>`_.
    """
    def xyz(arr):
        """array to xyz, for 2D arrays
        """
        r, c = arr.shape
        n = r * c
        x, y = np.meshgrid(np.arange(c), np.arange(r))
        dt = [('X', '<i8'), ('Y', '<i8'), ('Vals', a.dtype.str)]
        out = np.zeros((n,), dtype=dt)
        out['X'] = x.ravel()
        out['Y'] = y.ravel()
        out['Vals'] = a.ravel()
        return out
    r = 5  # Y
    c = 4  # X
    a = np.arange(r * c).reshape(r, c)
    b = xyz(a)
    frmt = """
    :---------------------------------------------------------------------:
    {}
    Input array...
    {!r:}\n
    Output array...
    {}
    :---------------------------------------------------------------------:
    """
    args = [dedent(num_121.__doc__), a, b]
    print(dedent(frmt).format(*args))
    return a, b, c

# ----------------------------------------------------------------------
# num_122()  # Do an equation expansion
def num_122():
    """ Do an equation expansion
    :
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :input arrays..
    m  {}
    x  {}
    y  {}
    result of ... out = np.array(m)*x[:, None] + y[:, None]
    {}

    :---------------------------------------------------------------------:
    """

    m = np.array([0.2, 0.4, 1.2])
    x, y = np.array([1,2,3,4]), np.array([.1,.2,.3,.4])
    out = np.array(m)*x[:, None] + y[:, None]
    args = [dedent(num_122.__doc__), m, x, y, out]
    print(dedent(frmt).format(*args))
    # return

# ----------------------------------------------------------------------
# num_123()  # Reshape and transpose columns using einsum
def num_123():
    """
    num_123() # Reshape and transpose columns using einsum
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    Input array... a
    {}\n
    Reformatted array...
    b = np.einsum('ijk->jik', a)
      = np.rollaxis(a, 1)
      = np.transpose(a, (1,0,2))\n
    {}
    :---------------------------------------------------------------------:
    """
    a = np.arange(2*3*4).reshape(2,3,4)
    b = np.einsum('ijk->jik', a)
    args = [dedent(num_123.__doc__), a, b]
    print(dedent(frmt).format(*args))
    # return

# ----------------------------------------------------------------------
# num_124()  # Running count of values in a 1D array
def num_124():
    """Running count of values in a 1D array
    can be used for labelling

    Reference:
    `<https://stackoverflow.com/questions/52754108/computing-a-moving-sum-of-
    counts-on-a-numpy-array>`_.
    """
# def running_count(arr):
    a = np.random.randint(1, 10, 20)
    a = np.array(list("xabaaybeeetz"))
    dt = [('Value', a.dtype.str), ('Count', '<i4')]
    z = np.zeros((a.shape[0],), dtype=dt)
    idx = a.argsort(kind='mergesort')
    s_a = a[idx]
    neq = np.where(s_a[1:] != s_a[:-1])[0] + 1
    run = np.ones(a.shape, int)
    run[neq[0]] -= neq[0]
    run[neq[1:]] -= np.diff(neq)
    out = np.empty_like(run)
    out[idx] = run.cumsum()
    z['Value'] = a
    z['Count'] = out
    frmt = """
    ---------------------------------------------------------------------:
    {}
    Input array... a
    {}
    Running count
    {}
    Output
    {}
    :---------------------------------------------------------------------:
    """
    args = [dedent(num_124.__doc__), a, out, z]
    print(dedent(frmt).format(*args))

# ----------------------------------------------------------------------
# num_125 combine dictionary keys and values
def num_125():
    """Combine dictionary values from multiple dictionaries and combine
    their keys if needed.
    Requires: import numpy as np
    Returns: a new dictionary
    """
    def combine_dicts(ds):
        """Combine dictionary values from multiple dictionaries and combine
        their keys if needed.
        Requires: import numpy as np
        Returns: a new dictionary
        """
        a = np.array([(k, v)                 # key, value pairs
                      for d in ds            # dict in dictionaries
                      for k, v in d.items()  # get the key, values from items
                      ])
        ks, idx = np.unique(a[:, 0], True)
        ks = ks[np.lexsort((ks, idx))]       # optional sort by appearance
        uniq = [np.unique(a[a[:, 0] == i][:, 1]) for i in ks]
        nd = [" ".join(u.tolist()) for u in uniq]
        new_d = dict(zip(ks, nd))
        return new_d

    ds = [
         {'Subdivision': 'NENW', 'Twp': '026S', 'Range': '030E', 'Sec': '14', 'Sur Type': 'B', 'Meridian': 'Numpy'},
         {'Subdivision': 'NWNE', 'Twp': '020S', 'Range': '033E', 'Sec': '13', 'Sur Type': 'A', 'Meridian': 'Is'},
         {'Subdivision': 'SENW', 'Twp': '021S', 'Range': '033E', 'Sec': '15', 'Sur Type': 'A', 'Meridian': 'The'},
         {'Subdivision': 'SWNE', 'Twp': '025S', 'Range': '033E', 'Sec': '13', 'Sur Type': 'A', 'Meridian': 'Best'}
         ]
    new_d = combine_dicts(ds)
    frmt = """
    :---------------------------------------------------------------------:
    {}
    see dictionaries in code...
    Result...
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_125.__doc__, new_d]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_126 calculate primes in a range
def num_126():
    """Calculate the primes within a range of values
    """
#    from itertools import permutations as permutations
#    from itertools import combinations
#    from itertools import chain
    def primes(start, end):
        """Primes within start and end
        """
        _primes = []
        for number in range(start, end + 1):
            is_prime = True
            for num in range(2, number):
                if number % num == 0:
                    is_prime = False
                    break
            if is_prime:
                _primes.append(number)
        return _primes
    #
    start = 2
    value = 100
    end = (value + 1)//start
    p = primes(start, end)
    p = np.array(p)
    vals = p[np.where(value % p == 0)[0]]
    return vals

#from itertools import combinations, permutations, chain
#def powercomb(iterable):
#    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
#
#    `<https://docs.python.org/3/library/itertools.html#itertools-recipes>`
#    """
#    s = list(iterable)
#    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def reshape_options(a):
    """Alternative shapes for a numpy array.

    Parameters:
    a : ndarray
        The ndarray with ndim >= 2

    Returns:
    --------
    The shapes of equal or lower dimension, excluding ndim=1

    >>> a.shape # => (3, 2, 4)
    array([(2, 12), (3, 8), (4, 6), (6, 4), (8, 3), (12, 2), (2, 3, 4),
           (2, 4, 3), (3, 2, 4), (3, 4, 2), (4, 2, 3), (4, 3, 2)],
          dtype=object)

    Notes:
    ------
    >>> s = list(a.shape)
    >>> case = np.array(list(chain.from_iterable(permutations(s, r)
                        for r in range(len(s)+1)))[1:]
    >>> prod = [np.prod(i) for i in case]
    >>> match = np.where(prod == size)[0]

    References:
    -----------
    `<https://docs.python.org/3/library/itertools.html#itertools-recipes>`
    """
    from itertools import permutations, chain
    s = list(a.shape)
    case0 = np.array(list(chain.from_iterable(permutations(s, r)
                    for r in range(len(s)+1)))[1:])
    case1 = [i + (-1,) for i in case0]
    new_shps = [a.reshape(i).shape for i in case1]
    z = [i[::-1] for i in new_shps]
    new_shps = new_shps + z
    new_shps = [i for i in np.unique(new_shps) if 1 not in i]
    new_shps = np.array(sorted(new_shps, key=len, reverse=False))
    return new_shps


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        out = np.empty_like(ix, dtype=dtype)

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def struct_deco(func):
    """Prints a structured array using `frmt_struct`
    Place this decorator over any function that returns a structured array so
    it can be easily read
    """
    from functools import wraps  # Uncomment, or move it inside the script.
    @wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper function"""
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        argf = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                argf.append("array (shape: {})".format(args[0].shape))
            else:
                argf.append(arg)
        d = {**dict(zip(arg_names, argf)), **kwargs}
        nf = len(max(arg_names, key=len))
        darg = "\n".join(["  {!s:<{}} : {}".format(i, nf, d[i]) for i in d])
        ar = [func.__name__, darg]
        print("\nFunction... {}\nInputs...\n{}\n".format(*ar))
        #
        result = func(*args, **kwargs)   # do the work
        prn_struct(result)              # format the result
        return result                    # for optional use outside.
    return wrapper


def prn_struct(a, edgeitems=3, max_lines=25, wdth=100, decimals=2, prn=True):
    """Format a structured array by reshaping and replacing characters from
    the string representation
    """
    a = np.asanyarray(a)
    nmes = a.dtype.names
    if nmes is not None:
        dtn = "Column names ...\n" + ", ".join(a.dtype.names)
    with np.printoptions(precision=decimals,
                         edgeitems=edgeitems,
                         threshold=max_lines,
                         linewidth=wdth):
        repl = ['[', ']', '(', ')', '"', "'", ',']
        s = str(a.reshape(a.shape[0], 1))
        for i in repl:
            s = np.char.replace(s, i, " ")
        print("{}\n\n{}".format(dtn, s))
     #


import arcpy
arcpy.env.workspace = r"C:\Data\rasters"
import numpy as np
r_max_mean = []
for raster in arcpy.ListRasters("*"):
    ras = arcpy.RasterToNumPyArray(raster)
    r_max_mean.append( np.nanmax(ras) )
print("max_list\n{}\nmean_max {}".format(r_max_mean, np.nanmean(r_max_mean)))



# -----------------------------
# an interestiing ditty on finding points with a minimum spacing
"""
coords
 
array([[278, 236],
       [ 98, 969],
       [807, 380],
       [924, 526],
       [232, 828],
       [816, 918],
       [797, 361],
       [167, 912],
       [112, 804],
       [396, 609]])

m = np.triu(e_dist(coords, coords))  # a function I can provide

w = np.where(m > 700)

f = coords[w[0]]; t= coords[w[1]]

ft = np.hstack((f, t))

m
Out[475]: 
array([[  0.  , 754.78, 548.25, 708.11, 593.78, 868.66, 533.84, 685.05, 591.76, 391.22],
       [  0.  ,   0.  , 921.74, 937.3 , 194.52, 719.81, 926.43,  89.5 , 165.59, 467.34],
       [  0.  ,   0.  ,   0.  , 187.1 , 728.92, 538.08,  21.47, 832.24, 814.13, 470.49],
       [  0.  ,   0.  ,   0.  ,   0.  , 755.03, 406.61, 208.22, 849.73, 858.27, 534.48],
       [  0.  ,   0.  ,   0.  ,   0.  ,   0.  , 590.89, 733.02, 106.21, 122.38, 273.6 ],
       [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  , 557.32, 649.03, 713.17, 521.42],
       [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  , 836.96, 815.77, 471.49],
       [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  , 121.2 , 379.8 ],
       [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  , 344.5 ],
       [  0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ,   0.  ]])

ft
Out[476]: 
array([[278, 236,  98, 969],
       [278, 236, 924, 526],
       [278, 236, 816, 918],
       [ 98, 969, 807, 380],
       [ 98, 969, 924, 526],
       [ 98, 969, 816, 918],
       [ 98, 969, 797, 361],
       [807, 380, 232, 828],
       [807, 380, 167, 912],
       [807, 380, 112, 804],
       [924, 526, 232, 828],
       [924, 526, 167, 912],
       [924, 526, 112, 804],
       [232, 828, 797, 361],
       [816, 918, 112, 804],
       [797, 361, 167, 912],
       [797, 361, 112, 804]])
"""
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    num_121()  # 2D array to xy_val
#    num_122()  # Do an equation expansion
#    num_123() # Reshape and transpose columns using einsum
#    num_124()  # Running count of values in a 1D array
#    num_125()  # combine_dicts(ds)
    p = num_126()  # calculate primes in a range

