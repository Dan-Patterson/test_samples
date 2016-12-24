# -*- coding: UTF-8 -*-
"""
:Script:   testing_script_04.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-12-17
:Purpose:  Demonstration functions for a variety of small examples.
:Functions:  help(<function name>) for help
:---------
:Functions list .........
:...... np functions .....
:    num_56()  Geometric mean calculation
:    num_57()  transposing and reshaping arrays
:    num_58()  arrays from uneven lists
:    num_59()  advanced slicing
:    num_60()  subtraction between arrays and transposed arrays
:    num_61()  Unravel indices
:    num_62()  Products of various kinds7
:    num_63()  quick length/distance demo using row norms from scipy
:    num_64()  mixing dtypes in arrays
:    num_65()  index array and booleans
:    num_66()  savetxt example...
:    num_67()  datetime operations
:    num_68()  local minima demo
:    num_69()  as_strided useage
:    num_70()  unique on recarrays
:
:Notes:
:  genfromtxt
:  data = "1, 1.23, "
:  s = BytesIO(data.encode())
:  s = StringIO("1,1.3,abcde")
:  data = np.genfromtxt(s, dtype=[('myint','i8'),('myfloat','f8'),
:    ... ('mystring','S5')], delimiter=",")
:
:References
:
:
:---------------------------------------------------------------------:
"""
#---- imports, formats, constants ----

import os
import sys
import numpy as np
import inspect
from textwrap import dedent, indent, wrap
from io import StringIO, BytesIO
from plot_arr import _f
import arr_tools as art


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=5, linewidth=80, precision=1,
                    suppress=True, threshold=100, 
                    formatter=ft)
np.ma.masked_print_option.set_display('-')
script = sys.argv[0]

# ---- functions ----
#----------------------------------------------------------------------
# num_56
def num_56():
    """(num_56)...Geometric mean calculation
    :Reference:
    :---------
    :  https://geonet.esri.com/thread/13811
    :
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array ... shape: {}
    {}
    :log10 array
    {}
    :Cumulative sum
    {}
    :Geometric mean = 10.0**((cum_sum[-1])/N)
    {}
    :------------------------------------------------------------------
    """
    import random
    r = 6
    c = 6
    a = (np.arange(1, r*c + 1)).reshape((r,c))
    loga = np.log10(a)
    cum_sum = np.cumsum(loga)
    N = np.size(a)
    gm = 10.0**((cum_sum[-1])/N)
    """
    big = [ random.randint(1,255) for i in range(1000)]
    b = np.array(big,dtype='int64')
    loga = np.log10(b)
    N = len(big)
    cum_sum = np.cumsum(loga)
    gm = 10.0**((cum_sum[-1])/N)
    """
    p = "   . "
    args = [num_56.__doc__, a.shape,
            indent(str(a), p),
            indent(str(loga), p),
            indent(str(cum_sum), p),
            indent(str(gm),p)]
    print(dedent(frmt).format(*args))


#----------------------------------------------------------------------
# num_57
def num_57():
    """(num_57)...transposing and reshaping arrays
    :
    :Reference:
    :---------
    :  http://stackoverflow.com/questions/40551022/reshape-numpy-
    :  array-to-contain-logical-blocks-of-values-from-original-array
    """
    frmt = """
    :------------------------------------------------------------------
    :Input array before transpose and reshape...
    {}
    :a.reshape(3, 3, 2, 2).transpose([0, 2, 1, 3]).reshape(6, 6) ...
    {}
    :a.reshape(2, 2, 3, 3).transpose([0, 2, 1, 3]).reshape(6, 6)
    {}
    :a.reshape(6, 2, 3).transpose([1, 2, 0]).reshape(6, 6)
    {}
    :a.reshape(2, 3, 6).transpose([2, 0, 1]).reshape(6, 6)
    {}
    :a.reshape(2, 6, 3).transpose([1, 0, 2]).reshape(6, 6)
    {}
    :------------------------------------------------------------------
    """
    a = np.arange(6*6).reshape(6, 6)
    b = a.reshape(3, 3, 2, 2).transpose([0, 2, 1, 3]).reshape(6, 6)
    c = a.reshape(2, 2, 3, 3).transpose([0, 2, 1, 3]).reshape(6, 6)
    d = a.reshape(6, 2, 3).transpose([1, 2, 0]).reshape(6, 6)
    e = a.reshape(2, 3, 6).transpose([2, 0, 1]).reshape(6, 6)
    f = a.reshape(2, 6, 3).transpose([1, 0, 2]).reshape(6, 6)
    print(dedent(frmt).format(a, b, c, d, e, f))
    #return a, b, c, d, e
    
#----------------------------------------------------------------------
# num_58
def num_58():
    """(num_58)... arrays from uneven lists
    :
    : http://stackoverflow.com/questions/40569220/efficiently-convert
    :      -uneven-list-of-lists-to-minimal-containing-array-padded-with
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :------------------------------------------------------------------
    """
    import itertools
    null = np.nan
    a = [[1, 10, 100, 1000], [2, 20], [3, 30, 300]]
    b = np.array(list(itertools.zip_longest(*a, fillvalue=null))).T
    args = [num_58.__doc__]    
    #return a, b

#----------------------------------------------------------------------
# num_59
def num_59():
    """(num_59)...advanced slicing
    :
    :Reference:
    :---------
    :  http://stackoverflow.com/questions/40598824/advanced-slicing-
    :       when-passed-list-instead-of-tuple-in-numpy#40599589
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array...
    {}\n
    :s_0 => {}
    {}\n
    :s_1 => {}
    {}
    :------------------------------------------------------------------
    """
    #a = np.arange(15)
    a = np.arange(2*3*4).reshape(2,3,4)
    s0 = slice(1, 2, 3)
    s1 = [1, 2, [3, 2, 1]]
    b = a[s0]
    c = a[s1]
    args = [num_59.__doc__, a, s0, b, s1, c]
    print(dedent(frmt).format(*args))
    #return a, b, c
#----------------------------------------------------------------------
# num_60
def num_60():
    """(num_60)... subtraction between arrays and transposed arrays
    :
    :  http://stackoverflow.com/questions/40601144/fastest-pairwise-
    :        difference-of-rows
    :
    :  http://stackoverflow.com/questions/40617212/extract-indices-of-
    :       minimum-in-numpy-array
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array (a)...
    {}\n
    :Reshape and subtract... a[:, np.newaxis] - a
    {}\n
    :Reshape, transpose and subtract transpose...
    : ...=> a[:, np.newaxis].T - a.T
    {}
    :------------------------------------------------------------------
    """
    a = np.arange(2*4).reshape(4,2)
    b = a[:, np.newaxis] - a
    c = a[:, np.newaxis].T - a.T
    args = [num_60.__doc__, a, b, c]
    print(dedent(frmt).format(*args))
    #return a, b, c


#----------------------------------------------------------------------
# num_61
def num_61():
    """(num_61)... Unravel indices
    :
    :  unravel_index(indices, dims, order='C')
    :  dims is essentially a shape you want
    :  You can produce an array of coordinates as shown in the last one.
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Find the indices of values in an array...
    :
    : np.unravel_index(condition, shape) for array...
    {}
    : a.argmin {}  {} 
    : a.argmax {}  {}
    : np.argwhere(a % 5 == 0)
    {}
    : np.array(np.unravel_index([7, 3, 1, 9], a.shape)).T
    : finds the indices where 7, 3, 1 and 9 are located
    {}
    :------------------------------------------------------------------
    """
    a = np.arange(4*6).reshape(6,4)
    #cool 
    #np.unravel_index([6], (3,4))
    tests = [np.argmin(a), np.argmax(a)]
    args = [num_61.__doc__, a]
    for i in tests:
        idx = np.unravel_index(i, a.shape)
        args.extend([i, idx])
    w = np.argwhere(a % 5 == 0)
    s = np.array(np.unravel_index([7, 3, 1, 9], a.shape)).T
    print(dedent(frmt).format(*args, w, s))
    #return args


#----------------------------------------------------------------------
# num_62
def num_62():
    """(num_62)... Products of various kinds
    : Syntax 
    :  np.prod(a, axis=None, dtype=None, out=None, keepdims=False)
    :  np.cumsum(a, axis=None, dtype=None, out=None)
    :  np.inner(a, b) => out.shape = a.shape[:-1] + b.shape[:-1]
    :  np.outer(a, b) => out[i, j] = a[i] * b[j]
    :  np.dot(a, b)   => matrix multiplication or
    :                    dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])
    :  np.cumprod
    :  np.tensordot
    """
    frmt = """
    :------------------------------------------------------------------
    :Products using numpy...
    : ....
    : .... 1D arrays ......................................:
    : ....
    :array 'a1' {}
    :array 'b1' {}
    :
    :np.prod(a1) .... {}
    :
    :np.prod(b1, keepdims=True) .... {}
    :
    :np.inner...  {}
    :
    :np.outer...
    {}
    :
    :np.cumprod(a1, axis=None) ...... {}
    :
    :np.einsum('i,i->i', a1, b1) .... {}
    :
    :np.dot(a1, b1) ... {}  (1*4 +2*5 + 3*6)
    :
    :np.kron(a1, b1) ... 
    {}
    :
    :np.kron(b1, a1) ...
    {}
    :
    : ....
    : .... 2D arrays ......................................:
    : ....
    :array 'a2'...
    {}
    :array 'b2'
    {}
    :np.prod(a2) = {}
    :
    :np.prod(a2, axis=0) = {}
    :
    :np.prod(a2, axis=1) = {}
    :
    :np.prod((a2, b2)) = {}
    :
    :np.prod((a2, b2), axis=0)
    {}
    :
    :np.dot(a2, b2)
    {}
    :
    :np.cumprod(a2) .... {}
    :
    :np.cumprod(a2, axis=0)
    {}
    :
    :np.cumprod(a2, axis=1)
    {}
    : ....
    : .... mixed dimensions ................................:
    : ....
    :np.outer(a1, b2) shape {}, product...
    {}
    : 1D with 2D ....
    :np.kron(a1,b2)
    {}
    :
    :np.tensordot(a1, a2, axes=0)
    {}
    :
    : 2D with 2D ....
    :np.kron(a2,b2)
    {}
    :
    :np.tensordot(a2, a2, axes=0)
    {}
    :
    :np.tensordot(a2, a2, axes=1)
    {}
    :
    :np.tensordot(a2, a2, axes=2)
    {}
    : ....
    : .... fun stuff ........................................:
    : ....
    : z = np.arange(9).reshape(3,3)
    : z1 = np.ones(2*2, dtype='int').reshape(2,2)
    :
    :np.kron(z,z1)
    {}
    :------------------------------------------------------------------
    """
    args = [num_62.__doc__]
    p = "    "
    a1 = np.array([1, 2, 3])
    b1 = np.array([4, 5, 6])
    a2= np.array([[1, 2], [3, 4]])
    b2 = np.array([[3, 4], [5, 6]])
    z = np.arange(9).reshape(3,3)
    z1= np.ones(2*2, dtype='int').reshape(2,2)
    ar11 = [a1, b1, np.prod(a1), np.prod(b1, keepdims=True),
            np.inner(a1, b1),
            indent(repr(np.outer(a1, b1)), p),
            np.cumprod(a1),
            np.einsum('i,i->i', a1, b1),
            np.dot(a1, b1),
            np.kron(a1, b1),
            np.kron(b1, a1)
            ]
    ar12 = [indent(repr(a2), p), indent(repr(b2), p),
            np.prod(a2),
            np.prod(a2, axis=0),
            np.prod(a2, axis=1),
            np.prod((a2, b2)),
            indent(repr(np.prod((a2, b2), axis=0)), p),
            indent(repr(np.dot(a2, b2)), p),
            np.cumprod(a2),
            indent(repr(np.cumprod(a2, axis=0)), p),
            indent(repr(np.cumprod(a2, axis=1)), p),
            np.outer(a1, b2).shape,
            indent(repr(np.outer(a1, b2)), p),
            np.kron(a1,b2),
            indent(repr(np.tensordot(a1, a2, axes=0)), p)
            ]
    ar22 = [np.kron(a2,b2),
            indent(repr(np.tensordot(a2, a2, axes=0)), p),
            indent(repr(np.tensordot(a2, a2, axes=1)), p),
            indent(repr(np.tensordot(a2, a2, axes=2)), p)
            ]
           
    arxt = [indent(repr(np.kron(z, z1)), p)
           ]
    args = ar11 + ar12 + ar22 + arxt
    print(dedent(frmt).format(*args))
    return a1, b1, a2, b2


#----------------------------------------------------------------------
# num_63
def num_63():
    """(num_63)... quick length/distance demo using row norms from scipy
    a = np.array([0, 0]) 
    b = np.array([1, 1])
    dist = norm(a - b)
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Length/distance demo using einsum
    :from scipy.distance _row_norms
    :Input point pairs....
    {}
    :Distance between pairs...
    {}
    :Cumulative distance......
    {}
    :------------------------------------------------------------------
    """
    from numpy.linalg import norm
    #
    def _row_dist(a):
        diff = a[1:] - a[:-1]
        norms = np.einsum('ij,ij->i', diff, diff, dtype=np.double)
        return np.sqrt(norms, out=norms)
    #
    wonz = np.ones(7, dtype='int')
    steps = np.array([0, 1, 2, 5, 10, 20, 50])
    a = np.array(list(zip(steps, wonz)))
    dists = _row_dist(a)
    c_dist = np.cumsum(dists)
    args = [num_63.__doc__, a, dists, c_dist ]
    print(dedent(frmt).format(*args))
    return a, dists


#----------------------------------------------------------------------
# num_64
def num_64():
    """(num_64)... mixing dtypes in arrays
    :A couple of options to combine arrays of mixed dtypes.  This 
    :includes generating the dtype from the inputs and using the 
    :recfunctions module.
    :
    :Reference:
    :---------
    :  http://stackoverflow.com/questions/40697241/is-it-possible-to-
    :        retain-the-datatype-of-individual-numpy-arrays-with-
    :        concatenat/40705832#40705832
    """
    from numpy.lib import recfunctions as rfn
    frmt = """
    :------------------------------------------------------------------
    {}
    :Array 'a'...
    {}
    :Array 'b'...
    {}
    :Combined dtype ... {}
    :Resultant structured array...
    {!r:}
    :Column-wise view
    {!r:}
    :
    :Alternate using recfunctions (rfn.merge_arrays)
    :
    {!r:}
    :
    : np.alltrue(c==d)? {}
    :
    :------------------------------------------------------------------
    """
    p = "    "
    a = np.array([[1, 2], [3, 4], [5, 6]])
    b = np.array([[1, 2.1], [3.5, 4], [5, 6.8]])
    a0 = a.flatten()
    b0 = b.flatten()
    dt = a0.dtype.descr + b0.dtype.descr
    c = np.array(list(zip(a0, b0)), dtype=dt)
    d = rfn.merge_arrays((a, b),
                         flatten=True,
                         usemask=False,
                         asrecarray=False)
    args = [num_64.__doc__, indent(str(a), p),
            indent(str(b), p), dt, c, 
            c.reshape(c.shape[0], 1), 
            d, np.alltrue(c==d)
            ]
    print(dedent(frmt).format(*args))
    return a, b, c, d

#----------------------------------------------------------------------
# num_65
def num_65():
    """(num_65)... index array and booleans
    :Produce an index array emulating rows and columns
    :Show setting it to a value
    :Reference
    :---------
    :  http://stackoverflow.com/questions/40729927/2-d-arrays-with-
    :          numpy-arange
    :Notes
    :-----
    :  np.where(m[0]==m[1],1, np.nan)
    :  np.where(m[0]>=m[1], m[0], m[1])
    :  np.where(m[0]==m[1], 1, np.where(m[0]>=m[1], m[0]+1, m[1]+1))
    """
    frmt = """
    :------------------------------------------------------------------
    :{}
    :Basic array...{}
    :Index array...
    {}
    :Results...
    {}
    :
    :------------------------------------------------------------------
    """
    p = "    "
    a = np.arange(0, 10)
    idx = a[:,None]+a
    out = (np.mod(a,a[:,None])==0) & (a[:,None]!=0)
    args = [num_65.__doc__, a, indent(str(idx), p), indent(str(out), p)]
    print(dedent(frmt).format(*args))
    return a, idx, out

#----------------------------------------------------------------------
# num_66
def num_66():
    """(num_66)... savetxt example...
    :
    :References:
    :----------
    :  http://stackoverflow.com/questions/40735584/numpy-savetxt-error
    :  http://stackoverflow.com/questions/36507283/shape-of-a-
    :       structured-array-in-numpy/36509122#36509122
    :  http://stackoverflow.com/questions/16621351/how-to-use-python-
    :       numpy-savetxt-to-write-strings-and-float-number-to
    :       -an-ascii-fi/35209070#35209070
    """
    frmt = """
    :------------------------------------------------------------------
    
    :------------------------------------------------------------------
    """
    names  = np.array(['NAME_1', 'NAME_2', 'NAME_3'])
    floats = np.array([ 0.1234 ,  0.5678 ,  0.9123 ])
    ab = np.zeros(names.size, dtype=[('var1', 'S6'), ('var2', float)])
    ab['var1'] = names
    ab['var2'] = floats
    #np.savetxt('test.txt', ab, fmt="%10s %10.3f")
    args = [num_66.__doc__]
    return None

#----------------------------------------------------------------------
# num_67
def num_67():
    """(num_67)... datetime operations
    :
    :Reference:
    :---------
    :  http://stackoverflow.com/questions/40751055/generate-1-1-days-
    :       around-a-given-date-in-numpy
    :  or (t[:, None] + [-1,0,1]).ravel()
    :Notes:
    :-----
    :  d0 = np.datetime64('2016-09')
    :  d0.dtype  =>  dtype('<M8[M]')
    :  d1 = np.datetime64('2016-09-01')
    :  d1.dtype  =>  dtype('<M8[D]')
    :
    :  d0 + np.arange(-1,2)
    :    array(['2016-08', '2016-09', '2016-10'], dtype='datetime64[M]')
    :  d1 + np.arange(-1,2)
    :  array(['2016-08-31', '2016-09-01', '2016-09-02'],
    :        dtype='datetime64[D]')
    :
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input times...
    {}
    :Bracketed by a day...
    {}
    :    
    :------------------------------------------------------------------
    """
    t = np.array(['2016-04-30', '2016-06-30', '2016-09-30', '2016-12-31'],
                  dtype='datetime64[D]')
    a = (t[:,None] + np.arange(-1,2)).ravel()
    args = [num_67.__doc__, t, a]
    print(dedent(frmt).format(*args))
    return t

#----------------------------------------------------------------------
# num_68
def num_68():
    """(num_68)... local minima demo
    """
    frmt = """
    :------------------------------------------------------------------
    :input array 'a'
    {}
    :local minima of 'a'
    {}
    :------------------------------------------------------------------
    """
    args = [num_68.__doc__]
    #a = np.random.randint(1,4, size=(10,10))
    a = np.zeros((10,10), dtype='int')
    a[2,2]=3
    a[6,7]=2    
    #np.random.shuffle(a)
    #a = a.reshape(6,6)    
    def local_minima(a):
        m =((a <= np.roll(a,  1, 0)) &
            (a <= np.roll(a, -1, 0)) &
            (a <= np.roll(a,  1, 1)) &
            (a <= np.roll(a, -1, 1)))
        return m
    m = local_minima(a)
    print(dedent(frmt).format(a, m))


#----------------------------------------------------------------------
# num_69
def num_69():
    """(num_69)... as_strided useage
    :Reference:
    :---------
    :  http://stackoverflow.com/questions/40773275/sliding-standard-
    :         deviation-on-a-1d-numpy-array
    :  https://github.com/numpy/numpy/blob/master/
    :         numpy/lib/stride_tricks.py
    :  np.lib.stride_tricks.as_strided
    :
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :input array
    {}
    :Sample calculation using window of {}
    :strided array
    {}
    :result...
    {}
    :------------------------------------------------------------------
    """
    from numpy.lib.stride_tricks import as_strided
    a = np.random.randint(0,5, size=10)
    W = 3 # Window size
    nrows = a.size - W + 1
    n = a.strides[0]
    a2D = as_strided(a, shape=(nrows, W), strides=(n, n))
    out = np.sum(a2D, axis=1)
    args = [num_69.__doc__, a, W, a2D, out]
    print(dedent(frmt).format(*args))
    return a, a2D


#----------------------------------------------------------------------
# num_70
def num_70():
    """(num_70)... unique on recarrays
    """
    a = [[r'c:\folder1\fcs1', 'fcs1', 100, 200],
     [r'c:\folder2\fcs1', 'fcs1', 100, 200],
     [r'c:\folder3\fcs2', 'fcs2', 100, 200],
     [r'c:\folder4\fcs1', 'fcs1', 100, 200],
     [r'c:\folder5\fcs1', 'fcs1', 100, 999]]
    a = [tuple(i) for i in a]
    dt = [("A", "U50"), ("B", "U10"), ("C", '<i4'), ("D", '<i4')]
    a = np.array(a, dtype=dt)
    uni, idx = np.unique(a[['B', 'C', 'D']], return_index=True)
    idx.sort()
    a_rows = a[idx]
    frmt = """
    input array
    {!r:}
    Unique from cols B,C,D
      {}
    indices {}
    Unique rows
    {}
    """
    print(frmt.format(a, uni, idx, a_rows))
    
    args = [num_70.__doc__]
    return None



#----------------------
if __name__ == "__main__":
    """Main section...   """
    #print("Script... {}".format(script))
    script = sys.argv[0]
#    in_file = os.path.dirname(script) + '/data/csv.txt'
#    num_56()  # Geometric mean calculation
#    num_57()  # transposing and reshaping arrays
#    num_58()  # arrays from uneven lists
#    num_59()  # advanced slicing
#    num_60()  # subtraction between arrays and transposed arrays
#    num_61()  # Unravel indices
#    num_62()  # Products of various kinds
#    num_63()  # quick length/distance demo using row norms from scipy
#    num_64()  # mixing dtypes in arrays
#    num_65()  # index array and booleans
#    num_66()  # savetxt example...
#    num_67()  # datetime operations
#    num_68()  # local minima demo
#    num_69()  # as_strided useage
#    num_70()  # unique on recarrays
#    num_71()
#    a = np.array([2, 0, -1, -5, 3, 4])  # format a list with extra spaces
#    print(("a : [" + " {:>3}"*len(a) + "]").format(*a))


