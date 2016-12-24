#coding: utf-8
"""
:Script:   testing_script_01.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-10-27
:Purpose:  
:  The following functions, from num_1 to num_N perform several simple tasks.
:  No error checking provided.  Uncomment the ones you want to run in
:  the __main__ section.
:  The data for each function is included in each def so that it is local.
:
:Table of contents
:
:Arrays.........
:......  construction .....
:    num_01()  Array creation using vstack, zip and array filling
:    num_02()  Using np.linspace with floats instead of np.arange
:    num_03()  sparse array
:    num_04()  Sub-dtypes in numpy and array formulation
:    num_05()  recarray access in numpy
:......  altering ......
:    num_06()  Changing array types via rounding etc
:    num_07()  Flatten an array
:    num_08()  Array size information
:    num_09()  Transposing 3D arrays
:    num_10()  Array slicing using the ellipse
:    num_11()  Slicing arrays
:    num_12()  Sorting an array
:    num_13()  Sorting an array revisited (see # 12)
:    num_14()  Array padding example
:......  working with array data ......
:    num_15()  Subtracting an array mean and the array
:    num_16()  Unique values for 1D and 2D arrays
:    num_17()  Striding arrays demo
:    num_18()  Condition checking and useage now in numpy.
:    num_19()  Using linalg, einsum, distance and timing
:    num_20()  Using fromiter, unique and histo all at once.
:    num_21()  Rearranging, deleting rows and columns using slicing
:    num_22()  Reclass arrays
:    num_23()  Block statistics
:    num_24()  Concatenate arrays
:
:...... python ......
:    py_01()   List comprehension formats
:    py_02()   List comprehensions alternate outputs
:    py_03()   formatting output with textwrap
:
...... matplotlib ......
:    mpl_01()  Plotting and interpolating
:    mpl_02()  Construct rectangular geometries
:
:    to do np.polynomial
"""

import sys
from textwrap import dedent
import numpy as np

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, 
                    formatter=ft)
script = sys.argv[0]


def OD():
    """Run this to save to one drive"""
    import console, editor 
    console.open_in(editor.get_path())


#----------------------------------------------------------------------
# num_01  Array creation using vstack, zip and array filling
def num_01():
    """(num_01)... Array creation using vstack, zip and array filling 
    """
    dt = [('X', '<f8'),('Y', '<f8'),('Z', '<f8')]
    dt1 = [('X', '<f8'), ('Y', '<f8'), ('Count', 'int32')]  # uniform dtype
    X = [9, 2, 6, 1, 0, 7, 2, 2, 6, 0, 9]
    Y = [0, 5, 1, 1, 9, 7, 5, 3, 8, 0, 9]
    Z = [9, 1, 2, 3, 9, 5, 6, 7, 8, 9, 9]
    a = np.vstack((X, Y, Z)).T        # this has a uniform dtype 'float64'
    a2 = np.array(list(zip(X, Y, Z)), dtype=dt)
    a3 = np.ones(shape=(11,), dtype=dt1)  # 1's array to fill with the dtype 
    a3['X'] = a[:, 0]                   # in the above, note shape=(11,0)
    a3['Y'] = a[:, 1]                   # since we are passing mixed types
    a3['Count'] = a[:, 2]
    frmt = """
    :------------------------------------------------------------------
    {}
    Array with uniform dtype...
    Construct using vstack and transform, no named fields...
    ...np.vstack((X, Y, Z)).T ...
    {!r}\n
    Construct using zip, named fields...
    ...np.array(list(zip(X, Y, Z)),dtype=dt)...\n
    {!r}\n
    Array with mixed dtype and named fields...
    Construct from a ones array and assignment from the vstack array...
    ... array=np.ones(shape=(11,),dtype=dt), array['X'] = vstack[:,0] etc\n
    {!r}
    :------------------------------------------------------------------
    """
    frmt = dedent(frmt)
    print(frmt.format(num_01.__doc__, a, a2, a3))
    return a3

#-----------------------------------------------------------------------
# num_02  http://stackoverflow.com/questions/32513424/
#          find-intersection-of-numpy-float-arrays/32513892#32513892   
def num_02():
    """(num_02)... 
    Using np.linspace with floats instead of np.arange
    ... arr = np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    If np.arange, with a step is used, the np.intersect1d will fail when comparing floats
    to floats.  Workarounds are found in the code, but np.linspace seems to work
    ... results ...
    """
    import numpy as np
    #option 1
    a = np.arange(2, 3, 0.1)
    b = np.array([2.3, 2.4, 2.5])
    a1 = ((a*10)).astype(np.int)
    a2 = a1/10.
    np.intersect1d(a2, b)  # result ...array([ 2.3,  2.4,  2.5])
    # option 2   result [2.3000000000000003, 2.4000000000000004, 2.5000000000000004]
    result = [i for i in a for j in b if np.isclose(i, j)]
    # option 3  sequence and step returned as setup
    seq, stp = np.linspace(2, 3, num=11, endpoint=True, retstep=True) #, dtype='float64')
    r = np.intersect1d(seq, b)
    frmt = """
    :------------------------------------------------------------------
    {}
    :linspace test: intersect sequence
    {} 
    :with   {} using
    :yields {}\n
    :Kludgy alternates are listed in the code
    :------------------------------------------------------------------
    """
    frmt = dedent(frmt)
    print(frmt.format(num_02.__doc__, seq, b, r,))
    return seq, b, r

#-----------------------------------------------------------------------
# num_03 coordinate sparse array
def num_03():
    """(num_03)...
    - Coordinate sparse array
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array........
    - shape: {}  ndim: {}
    {}
    :z values: {}  using a[::, 2] or a[...,2]
    :------------------------------------------------------------------
    """
    a = np.array([[ 0,  0, 10],[ 0,  1,  5], [ 1,  0,  3],[ 1,  1,  4]])
    z = a[::, 2]  # or   a[...,2]
    args = [num_03.__doc__, a.shape, a.ndim, a, z]
    print(dedent(frmt).format(*args))
    return a

#----------------------------------------------------------------------
# num_04
def num_04(prn=True):
    """(num_04)... Sub-dtypes in numpy and array formulation
    You can embed a dtype within a dtype to accommodate a field that contains
    two or more fields.  In this example, the dtype is constructed in this
    fashion, allowing: a['id'],  a['xy'],  a['xy']['x']  and a['xy']['y']
    """       
    N=10
    x = np.random.random_integers(0, 10, N)
    y = np.random.random_integers(10, 20, N)
    id = np.arange(10)
    dt_sub = np.dtype([('x', '<f8'), ('y', '<f8')])
    dt = np.dtype([('id', '<i4'), ('xy', dt_sub)])
    extra = np.ones(len(x), dt)   #just for demo and printing purposes
    a = np.ones(len(x), dt)
    a['id'] = id
    a['xy']['x'] = x
    a['xy']['y'] = y
    frmt = """
    :------------------------------------------------------------------
    {}
    : - sub-dtype: {}
    : - dtype as:  np.dtype([('id', '<i4'),('xy', dt_sub)])
    : - yields:  {}
    : - unshaped array...
    {}
    : - reshaped & filled array...
    {}
    : id ....{}
    : xy ....{}
    : x  ....{}
    : y  ....{}
    :
    :field access...
    : - ndarray:   a['id']  a['xy']  a['xy']['x']  a['xy']['y']
    : - recarray:  a.id     a.xy     a.xy.x        a.xy.y   via...
    : a.view(np.recarray)
    : -    plus ndarray access      
    : -reshaped...
    {}
    :------------------------------------------------------------------
    """
    frmt = dedent(frmt)
    if prn:  # optional printing
        args = [num_04.__doc__, dt_sub, dt, extra,
                a, id, a['xy'], x, y, a.reshape(-1, 1)]
        print(frmt.format(*args))
    return id, x, y, a
#-----------------------------------------------------------------------
# num_05
def num_05(): # needs to call 4
    """(num_05)... ndarray and recarray access in numpy, continued...
    The array has a shape=(10,) it can be viewed by reshaping to a.reshape(-1, 1)
    ndarray supports   a['field'] access
    recarray supports  a['field'] and... a.field ... access
    """
    id, x, y, a = num_04(prn=False)
    frmt = """
    :------------------------------------------------------------------
    {}
    :Array basics....
    :Input ndarray...
    {!r}
    :...reshaped... 
    {!r}
    :Viewed as recarray... 
    {!r}
    :...reshaped... 
    {!r}
    :------------------------------------------------------------------
    """
    a_rec = a.view(np.recarray)
    frmt = dedent(frmt)
    args = [num_05.__doc__, a, a.reshape(-1, 1), a_rec, a_rec.reshape(-1, 1)]
    print(frmt.format(*args))
    frmt = """
    :------------------------------------------------------------------
    :ndarray and recarray access...
    : - both...
    : -   a['id']      = {}
    : -   a['xy']      = {}
    : -   a['xy']['x'] = {}
    : - recarray only...
    : -   a_rec.id     = {}
    : -   a_rec.xy     = {}
    : -   a_rec.xy.x   = {}
    :------------------------------------------------------------------  
    """
    args = [a['id'], a['xy'], a['xy']['x'], a_rec.id, a_rec.xy, a_rec.xy.x]
    print(dedent(frmt).format(*args))
    return a

#------------------------------------------------------------------
# num_06 Changing array types via rounding etc
def num_06():
    """(num_06)... Changing array types via rounding etc"""
    np.set_printoptions(edgeitems=3,linewidth=80,precision=1, threshold=20)
    a = np.array([-2.0, -1.7, -1.5, -0.2, -0.1,
                  0.0, 0.1, 0.2, 1.5, 1.7, 2.0])
    frmt = """
    :------------------------------------------------------------------
    {}
    :Changing array types...
    : -  a .....{!r:}
    : -  ceil...{!r:}
    : -  floor..{!r:}
    : -  trunc..{!r:}
    : -  round..{!r:}
    :------------------------------------------------------------------
    """
    args = [num_06.__doc__, a, np.ceil(a), np.floor(a),
            np.trunc(a), np.round(a)]
    print(dedent(frmt).format(*args))

#-----------------------------------------------------------------------
# num_07  flatten array
#     http://stackoverflow.com/questions/32743414/
#     how-to-partially-flatten-a-cube-or-higger-dimensional-ndarray-in-numpy
"""
#I want to find the operation that turns this:
X = array([[[ 0, 1, 2, 3],
            [ 4, 5, 6, 7],
            [ 8, 9,10,11]],
           [[12,13,14,15],
            [16,17,18,19],
            [20,21,22,23]]]) into this:
array([[ 0, 1, 2, 3,12,13,14,15],
       [ 4, 5, 6, 7,16,17,18,19],
       [ 8, 9,10,11,20,21,22,23]])
"""
def num_07():
    """(num_07)... Flatten an array by one dimension or reshape
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array........
    : - shape: {}  ndim: {}
    {}\n
    :Transposed array... swapping -> transpose(1, 0, 2)
    : - shape: {}  ndim: {}
    {}\n
    :Reshaped array.....
    : - (a.shape[1], a.shape[2]*a.shape[0])
    : - (3, (4*2)) = (3, 8)
    : - shape: {}  ndim: {}
    {}
    :------------------------------------------------------------------
    """
    np.set_printoptions(edgeitems=4,linewidth=80,precision=2,suppress=True,threshold=10)
    a = np.array([[[ 0, 1, 2, 3],
                   [ 4, 5, 6, 7],
                   [ 8, 9,10,11]],
                  [[12, 13, 14, 15], 
                   [16, 17, 18, 19],
                   [20, 21, 22, 23]]])
    b0 = a.transpose(1, 0, 2)  # step 1
    b = b0.reshape(a.shape[1], a.shape[2]*a.shape[0])
    args = [num_07.__doc__, a.shape, a.ndim, a, b0.shape,
            b0.ndim, b0, b.shape, b.ndim, b]
    print(dedent(frmt).format(*args))
    out = []
    for i in range(len(b0)):
        delta = [["\n ", "__"],
                 ["[[", "_["],
                 ["]]", "]_"]]
        line = (np.array_str(b0[i]))
        for j in delta:
            line=line.replace(j[0], j[1])
        out.append(line)
        z = "".join([i+"\n" for i in out])    
    return a, b0, b, out

#-----------------------------------------------------------------------
# num_08 2D array size information...
def num_08():
    """(num_08)... 2D array size information
    http://stackoverflow.com/questions/9395758/how-much-memory-is-
         used-by-a-numpy-ndarray
          x = 128, 256, 512, 1024, 2048, 4096 float64/double
       size = 1024, 2048, 4096, 8192, 16384, 32768 bytes
    """
    print("{}".format(num_08.__doc__)) 
    frmt = """
    :------------------------------------------------------------------
    :Array type: {} dtype: {}
    :shape: {}  size: {}  ndims: {}  Mb: {}
    :------------------------------------------------------------------
    """
    for dt in [np.int32, np.float64]:
        for j in [128, 256, 512, 1024]: #,2048,4096]:
            a = np.ones((j, j), dtype=dt) *1024
            a.nbytes       #8192
            args = [type(a).__name__, a.dtype, 
                    a.shape, a.size, a.ndim, a.nbytes/(1024.**2)]
            print(dedent(frmt).format(*args))
    del a

#-----------------------------------------------------------------------
# num_09
#   http://stackoverflow.com/questions/31917061/merge-axis-in-numpy-array
#   You can use np.transpose to swap rows with columns and then reshape -
#   Or use np.swapaxes to do the swapping of rows and columns and then reshape -
#
def num_09():
    """(num_09)... Transposing 3D arrays"""
    X = np.arange(9).reshape(3, 3)
    X1 = X.transpose(1, 0).reshape(-1, X.shape[1])
    X2 = X.swapaxes(0, 1).reshape(-1, X.shape[1])
    frmt = """
    :------------------------------------------------------------------
    {}
    :arr... X = np.arange(9).reshape(3, 3)
    {}\n
    :transpose... X.transpose(1, 0).reshape(-1, X.shape[1])
    {}\n
    :transpose alternate... X.swapaxes(0, 1).reshape(-1, X.shape[1])
    {}
    :------------------------------------------------------------------
    """
    print(dedent(frmt).format(num_09.__doc__, X, X1, X2))

#-----------------------------------------------------------------------
# num_10 array slicing using the ellipse
#   http://stackoverflow.com/questions/118370/how-do-you-use-the-
#     ellipsis-slicing-syntax-in-python
def num_10():
    """(num_10)... Some demos of array slicing using the ellipse. 
    Nothing to pass to the function, but it can be altered to do so.
    """
    a = np.arange(25).reshape(5, 5)
    frmt = """
    :------------------------------------------------------------------
    {}\n
    :Input array ........  a = np.arange(25).reshape(5,5)
    {}\n
    :Column slices ......
    : -  along column 0 ...  a[..., 0] = {}
    : -  along column 1 ...  a[..., 1] = {}
    : -  from 2nd on ......  a[..., 2:]
    {}\n
    :Row slices .........
    : - along row 1 ......  a[0,...]  = {}
    : - first two rows ...  a[:2,...]
    {}\n
    :Fancy slices .....  a[[0, 4, 2],...]   they need not be in order either
    {}
    :------------------------------------------------------------------
    """
    funcs = [num_10.__doc__, a, a[..., 0], a[..., 1],
             a[..., 2:], a[0,...], a[:2,...], a[[0, 4, 2],...]]
    print(dedent(frmt).format(*funcs))
    return a

#-----------------------------------------------------------------------
# num_11 Slicing arrays
#   http://stackoverflow.com/questions/4257394/slicing-of-a-numpy-2d-array
#     -or-how-do-i-extract-an-mxm-submatrix-from-an-nxn-ar
def num_11():
    """(num_11)... Slicing arrays...
    """
    a = np.arange(25).reshape((5, 5))
    r0,r1,r2 = np.split(a, [1, 3], axis=0)
    c0,c1,c2 = np.split(a, [1, 3], axis=1)
    r3 = a[[0, 3, 4], :]
    r4 = a[0:5:2, :]
    c3 = a[:, [0, 3, 4]]
    c4 = a[:, 0:5:2]
    block =  a[0:3, 2:4]
    rc = a[ [0, 2], [3, 4]]
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array... a = np.arange(25).reshape((5, 5))
    {}\n
    :Row split using rows >>> np.split(a, [1, 3], axis=0)
    :notice that it splits <1, 1:<3 and 3:>
    {!r:}
    {!r:}
    {!r:}\n
    :Column split using   >>> np.split(a, [1, 3], axis=1)
    {!r:}
    {!r:}\n
    {!r:}\n
    :Select specific rows and columns 
    : - rows = a[[0, 3, 4], :]
    {!r:}\n
    : - cols = [:, [0, 3, 4]]
    {!r:}\n
    :Skipping rows and columns
    : - rows = a[0:5:2, :]
    {!r:}
    : - cols = a[:, 0:5:2]
    {!r:}\n
    :Specific blocks 
    : >>>  a[0:3, 2:4]
    {!r:}\n    
    :Specific cells  
    : >>>  a[ [0, 2], [2, 4]]
    {!r:}
    :------------------------------------------------------------------                   
    """
    args = [num_11.__doc__, a, r0, r1, r2,
            c0, c1, c2, r3, c3, r4, c4, block, rc]
    print(dedent(frmt).format(*args))
    return a

#-----------------------------------------------------------------------
# num_12  Sorting an array
def num_12(prn=True):
    """(num_12)... Sorting an array
    : - arr ... input array
    : - new_arr = arr[np.argsort(arr[:, 0])]      # sorts by X not y
    : - idx =  np.lexsort((arr[:, 1], arr[:, 0])) # works
    : - new_arr = arr[idx]
    """       
    X = [9, 2, 6, 1, 0, 7, 2, 2, 6, 0, 9]
    Y = [0, 5, 1, 1, 9, 7, 5, 3, 8, 0, 9]
    Z = [9, 1, 2, 3, 9, 5, 6, 7, 8, 9, 9]
    dt = [('X', '<f8'), ('Y', '<f8'), ('Z', 'int32')]
    arr = np.array(list(zip(X, Y, Z)), dtype=dt)
    idx = np.lexsort((Y, X))         # sort... note the order of columns
    a_s = arr[idx]                   # get the values in sorted order
    if prn:
        print("{}".format(num_12.__doc__))
        print("XYZ array......\n{}".format(arr.reshape((-1, 1))))  # fancy
        print("Sorted array...\n{}".format(a_s.reshape(-1, 1)))
    return a_s
#-----------------------------------------------------------------------
# num_13 sorting arrays again (see num_12)
def num_13():
    """(num_13)... Sorting sparse array"""
    frmt = """
    :------------------------------------------------------------------
    {}\n
    :Input array........
    : - shape: {}  ndim: {}\n
    {}\n
    :Sorting indices:... {}  using np.lexsort((a[:, 1], a[:, 0]))
    : ....a[:, 1], a[:, 0] means sort the rows by column 0, then 1
    :Output array.......
    {}\n 
    :Sorting indices:... {}  using np.lexsort((a[:, 4],a[:, 3]))
    :Output array.......
    {}\n
    :------------------------------------------------------------------
    """
    #a = np.arange(25, 0, -1).reshape((5, 5))
    a = np.array([[25, 23, 21, 23, 25],
                  [20, 18, 16, 18, 20],
                  [ 5,  4,  3,  2,  1],
                  [ 6,  8, 10,  8,  6], 
                  [11, 13, 15, 13, 11] ])
    idx0 = np.lexsort((a[:, 1], a[:, 0]))
    b0 = a[idx0]
    idx1 = np.lexsort((a[:, 4], a[:, 3]))
    b1 = a[idx1]
    args = [num_13.__doc__, a.shape, a.ndim, a, idx0, b0, idx1, b1]
    print(dedent(frmt).format(*args))
    return a

#-----------------------------------------------------------------------
# num_14
def num_14():
    """(num_14)... Array padding example"""
    print("{}".format(num_14.__doc__))
    X = np.arange(16).reshape(4, 4)
    Y = np.pad(X,pad_width=(1, 1),mode='constant', constant_values=0)
    frmt = "\nArray padding example...\narray\n{}\npadded array\n{}"
    print(frmt.format(X, Y))

#-----------------------------------------------------------------------
# num_15
def num_15():
    """num_(15)... Subtracting array values and the array mean
    :
    : x = np.random.rand(3, 5)  create an array using random values
    :
    """
    X = np.random.rand(3, 5)
    frmt = """
    :------------------------------------------------------------------
    {}
    :numpy version: {}
    :array... shape(3, 5)
    {}\n
    :row mean.... X.mean(axis=1, keepdims=True)
    {}\n
    : col mean... X.mean(axis=0, keepdims=True)
    {}\n
    :arr - row mean ... 
    {}
    :------------------------------------------------------------------
    """
    frmt = dedent(frmt)
    try:
        Xm = X.mean(axis=1, keepdims=True)
        Xc = X.mean(axis=0, keepdims=True)
        ans = X - Xm
        print(frmt.format(num_15.__doc__, np.__version__, X, Xm, Xc, ans))
    except:
        ans = X - X.mean(axis=1).reshape(-1, 1)
        print("previous method doesn't work...\n{}".format(ans))

#-----------------------------------------------------------------------
# num_16
def num_16():
    """\n(num_16)... Unique values for 1D and 2D arrays"""
    print("{}".format(num_16.__doc__))
    X = [0, 2, 6, 0, 7, 2, 2, 6, 0, 0]
    Y = [0, 5, 1, 9, 7, 5, 3, 1, 0, 9]
    Z = [9, 1, 2, 9, 5, 6, 7, 8, 9, 9]
    dt = [('X', '<f8'), ('Y', '<f8'), ('Z', 'int32')]
    arr = np.array(list(zip(X, Y, Z)), dtype=dt)
    idx = np.lexsort((Y, X))         # sort... note the order of columns
    a_s = arr[idx]                   # get the values in sorted order
    a_x = np.unique(a_s['X'])        # unique values, no need to sort
    a_y = np.unique(a_s['Y'])
    # 2 or more, use a list in the slice
    a_xy,idx = np.unique(a_s[['X', 'Y']], return_index=True)
    frmt = """
    :------------------------------------------------------------------
    :Unique array...
    : - input array ...
    {}\n
    : X... {}  ... np.unique(a_s['X'])  sorted array X values
    : Y... {}\n
    : Return unique values using X and Y and the indices
    :   XY.. np.unique(a_s[['X', 'Y']], return_index=True)
    {}\n
    : Indices {}... idx
    :------------------------------------------------------------------
    """
    print(dedent(frmt).format(a_s, a_x, a_y, a_xy.reshape(-1, 1), idx))
    #print("Unique values from indices\n{}".format(a_s[idx]))
    return a_s, a_xy  # return just the unique values
    
#-----------------------------------------------------------------------
# num_17 Strided array examples
# 
def rolling_window_lastaxis(a, window):
    """Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>"""
#    if window < 1:
#       raise ValueError, "window must be at least 1."
#    if window > a.shape[-1]:
#       raise ValueError, "`window` is too long."
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def rolling_window(a, window):
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)
    return a

#----------------  above are required ---------------------
def num_17():
    """\n(num_17)... Striding arrays demo...
    """
    import numpy as np
    from numpy.lib.stride_tricks import as_strided as as_strided
    np.set_printoptions(edgeitems=3, linewidth=80, precision=2, suppress=True, threshold=5)    
    a = np.arange(12).reshape(3, 4)
    a = np.pad(a, (1,), 'reflect', reflect_type='odd')  # pad demo
    shape = a.shape
    strides = a.strides
    #b = as_strided(a,shape=(25,2,2))
    filtsize = (3, 3)
    b = rolling_window(a, filtsize)
    b_m = b.mean(axis=-1).mean(axis=-1)
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array ...
    {}\n
    :shape = {}  strides = {}\n
    :Reshaped array ...
    {}\n
    :shape = {}  strides = {}\n
    :Mean of array b...
    {}
    :------------------------------------------------------------------
    """
    args = [a, shape, strides, b, b.shape, b.strides, b_m]
    print(dedent(frmt).format(num_17.__doc__,*args))
    return a, b
    
#-----------------------------------------------------------------------
# num_18 http://stackoverflow.com/questions/32290060/list-comprehensions-tip/32291666#32291666
def num_18():
    """(num_18)...
    :Condition checking and useage in numpy...
    :  see py_02() for list comprehension version
    """
    a = np.arange(10)
    q1 = (a > 2)       # Note the enclosing brackets
    q2 = (a < 8)       # Use them for all sub-queries
    two_bool = (q1, q2)                  # two boolean arrays
    all_bool = np.all((q1, q2), axis=0)  # or:   np.all(two_bool,axis=0)
    b = a[np.logical_and(q1,q2)]         # logical functions and slice
    c = a[q1 & q2]                       # simplified version 
    d = np.where(q1 & q2)[0]             # equivalent logical and slicing 
    e = a[np.all((q1, q2), axis=0)]
    sqr_result = np.square(a[(a > 2) & (a < 8)])
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array:  {}
    : q1 = (a > 2)  {}     booleans cast as int32
    : q2 = (a < 8)  {}     for easier viewing and printing
    :
    :Or one of these forms...
    : -  two_bool = (q1, q2)
    : -  np.all((q1, q2), axis=0)       
    : -  np.all(two_bool, axis=0)
    : -  (q1) & (q2)
    : -  np.logical_and(q1, q2)\n 
    :Results... 
    : - a[np.logical_and(q1, q2)]     {}
    : - a[q1 & q2]                    {}
    :     a[q1[q2]] slice of slice
    : - np.where((q1) & (q2))[0]      {}
    : - a[np.all((q1, q2), axis=0)    {}\n
    :Finally... square the selection 
    :    np.square(a[(a > 2) & (a < 8)]) {}
    :------------------------------------------------------------------   
    """
    q1 = q1.astype('int32')
    q2 = q2.astype('int32')
    args = [num_18.__doc__, a, q1, q2, b, c, d, e, sqr_result]
    print(dedent(frmt).format(*args))

#-----------------------------------------------------------------------
# num_19
#   L2 norm calculations
#   http://stackoverflow.com/questions/32083997/huge-speed-difference-in-numpy-between-similar-code
#   a = np.arange(1200.0).reshape((-1,3))
#   %timeit [np.sqrt((a*a).sum(axis=1))]
#   100000 loops, best of 3: 12 µs per loop
#   %timeit [np.sqrt(np.dot(x,x)) for x in a]
#   1000 loops, best of 3: 814 µs per loop
#   %timeit [np.linalg.norm(x) for x in a]
#   100 loops, best of 3: 2 ms per loop

def num_19():
    """(num_19)... Using linalg, einsum, distance and timing
    Data consists of 1000 random points from a uniform distribution.
    The distances are then calculated between all points and themselves
    which can be cast in either 2D, 3D or 4D"""
    print("{}".format(num_19.__doc__))
    import numpy as np
    import timeit
    a = np.random.uniform(0, 1000, 2000).reshape(1000, 2)
    a = np.sort(a, axis=1)
    stuff = "import numpy as np\na = np.random.uniform(0,1000,2000).reshape(1000,2)"
    # uncomment what you want to test
    to_test = ["np.linalg.norm(a,axis=1)",
               "np.sqrt(np.einsum('ij,ij->i',a,a))"
               ]
    for test in to_test:
        print("\nTesting... {}".format(test))
        for loop in [100,1000,10000]:
            t = timeit.timeit(test, setup=stuff, number=loop)
            print("N: {:>8.0f}, sec.: {:>12.4e} sec/N {:>12.4e}".format(loop,t,t/loop))
    return a
"""my results
testing... np.linalg.norm(a,axis=1)
N:        100, t:   0.007859 per loop 7.85899162292e-05
N:      10000, t:   0.711287 per loop 7.11287021637e-05
N:     100000, t:   6.584570 per loop 6.58457016945e-05
testing... np.sqrt(np.einsum('ij,ij->i',a,a))
N:        100, t:   0.003134 per loop 3.13401222229e-05
N:      10000, t:   0.300021 per loop 3.00020933151e-05
N:     100000, t:   2.745381 per loop 2.74538087845e-05
"""

#-----------------------------------------------------------------------
# num_20 http://stackoverflow.com/questions/32090058/
#    testing-whether-a-string-has-repeated-characters/32090862#32090862
def num_20():
    """(num_20)... Using fromiter, unique and histo all at once...
    The np.unique(b,return_counts=True) is only available in versions >= 1.9
    """
    import numpy as np
    a = "12348546478"                 # a potential iterable
    b = np.fromiter(a, dtype='int')   # convert it to an iterable
    if (np.version.version > '1.8.1'):
        uniq,cnt = np.unique(b, return_counts=True)
        # equivalent of set, and returns counts
        histo = np.array([uniq, cnt]).T  
        gram = histo[np.where(histo[:,1] > 1)]
        # extract the values and count where count > 1   
    else:
        uniq = np.unique(b)           # the first returned from unique
        cnt,bin = np.histogram(b, bins=uniq)
        histo = [bin, cnt]  # produce histogram, transpose it to look nice 
        gram = "  need numpy > 1.8, sorry..."  
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input iteratable {}
    :Unique values and their count
    {}
    :those where count > 1
    {}
    :------------------------------------------------------------------
    """
    print(dedent(frmt).format(num_20.__doc__, a, histo, gram)) # get it? I kill me sometimes

#-----------------------------------------------------------------------
# num_21 http://stackoverflow.com/questions/34007632/how-to-remove-a-column-in-a-numpy-array
#   reordering an array, deleting rows or columns
def num_21():
    """(num_21)... 
    Reordering an array or deleting rows and/or columns using slicing.
    """
    import numpy as np
    a = np.arange(20).reshape((5, 4))
    b = a[:, [2, 1, 0, 3]]
    c = a[:, [0, 2, 3]]
    d = a[[0, 1, 3, 4],:]
    frmt = """
    :------------------------------------------------------------------
    {}
    :The following are some tips.  If you assign to a new variable
    : it makes a copy, otherwise they are views and the original array
    : is not changed.\n
    :Input array ...  b = a[:] = a[:, :] copy syntax
    {}\n
    :Arrange cols...  a[:, [2, 1, 0, 3]] ...just a view
    {}\n
    :Delete col 1...  a[:, [0, 2, 3]]   ...keep all but column 1
    {}\n
    :Delete row 2...  a[[0, 1, 3, 4],:] ...keep all but 2nd row
    {}
    :------------------------------------------------------------------
    """
    print(dedent(frmt).format(num_21.__doc__, a, b, c, d))

#-----------------------------------------------------------------------
# num_22 reclass an array to integer classes based upon unique values
#    unique values are easily determined by sorting, finding successive 
#    differences as a reverse boolean, then producing and assigning a 
#    cumulative sum
def num_22():
    """
    :(num_22)... Reclass Arrays
    :Reclass an array to integer classes based upon unique values.
    :Unique values are easily determined by sorting, finding successive 
    :  differences as a reverse boolean, then producing and assigning a 
    :  cumulative sum.
    """    
    np.set_printoptions(edgeitems=5, linewidth=80, precision=2,
                        suppress=True, threshold=10)
    # construct the array
    #
    vals = np.array(['a', 'b', 'c', 'a', 'a', 'a', 'b', 'b', 'c', 'a'])
    idx = np.arange(len(vals), dtype="int32")
    ord = np.zeros(len(vals), dtype="int32")
    dt = [("ID", "int32"), ("Values", "U5"), ("Order", "int32")]
    a = np.array(list(zip(idx, vals, ord)), dtype=dt)
    #
    # sort the array, determine where consecutive values are equal
    # reclass them using an inverse boolean and produce a cumulative sum
    s_idx = np.argsort(a, order=['Values','ID'])
    final = a[s_idx]
    bool = final['Values'][:-1] == final['Values'][1:] # read very carefully again
    w = np.where([bool], 0, 1)   # find indices where they are the same
    csum = np.zeros(len(final), dtype="int32")
    #
    # sort back
    final["Order"][1:] = np.cumsum(w) #csum[1:] = np.cumsum(w)
    final = final[np.argsort(final, order=['ID'])]
    frmt = "{}\nInput array...\n{!r:}\n\nFinal array\n{!r:}"
    args = [dedent(num_22.__doc__), a.reshape(-1, 1), final.reshape(-1, 1)]
    print(dedent(frmt).format(*args))
    #return a, final

def num_23():
    """(num_23)... Block statistics
    :Block statistics uses a predefined block size, specified by
    : 'n' in this example. 
    :The array is created as a square using 'rc' rows and columns.
    :In this simplified example, the array is reshaped and the properties
    :  determined for the sub-blocks.
    :
    :References:
    : http://stackoverflow.com/questions/4624112/grouping-2d-numpy-array-in-average
    : http://stackoverflow.com/questions/16713991/indexes-of-fixed-size-sub-matrices-of-numpy-array
    :
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array...
    {}\n
    :Reshaped array...{}
    {}\n
    :Averages...
    : by column, per block... b1 = np.mean(b,axis=2, keepdims=True)
    {}\n
    : by row, per block...... b2 = np.mean(b,axis=3, keepdims=True)
    {}\n
    : by block, overall...... b3 =np.mean(b,axis=(2,3), keepdims=True)
    {}\n
    :Note: arrays b2 and b3 should be 'squeezed' to remove one or more
    :  axes that are 1
    : a.shape(6, 6),  b.shape(2, 2, 3, 3)
    : b1.shape(2, 2, 1, 3)
    : b2.shape(2, 2, 3, 1)
    : b3.shape(2, 2, 1, 1)
    :------------------------------------------------------------------
    """
    np.set_printoptions(edgeitems=5, linewidth=80,
                        precision=2, suppress=True, threshold=10)
    #rc = 6; n = 3;  num = rc*rc*n
    #a = np.arange(num,dtype="float64").reshape((rc*n,rc))
    #
    a = np.arange(36, dtype="float64").reshape((6, 6))
    r = 3
    c = 3
    lenr = a.shape[0]/r
    lenc = a.shape[1]/c
    b = np.array([a[i*r:(i+1)*r, j*c:(j+1)*c] 
                  for (i, j) in np.ndindex(lenr, lenc)])
    b = b.reshape(lenr, lenc, r, c) 
    b1 = np.mean(b, axis=2, keepdims=True)
    b2 = np.mean(b, axis=3, keepdims=True)
    b3 = np.mean(b, axis=(2, 3), keepdims=True)
    frmt = dedent(frmt)
    doc_string = dedent(num_23.__doc__)
    print(dedent(frmt).format(doc_string, a, b.shape, b, b1, b2, b3))
    return a, b, b1, b2, b3
    
def num_24():
    """(num_24)... Concatenate arrays
    :Concatenate by ensuring the shape of the two arrays agrees on
    :   the first dimension if concatenating columns.
    :Reference: hjpaul comment
    :  http://stackoverflow.com/questions/36878089/python-add-a-column
    :         -to-numpy-2d-array
    :
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array...
    {}\n
    :Add to columns... {} ... yields
    :Reshape: np.concatenate([a, b.reshape(a.shape[0], 1)], axis=1)
    {}\n
    :Add to rows... {} ... yields
    :Reshape: np.concatenate([a, b.reshape(1, a.shape[1])], axis=0) 
    {}
    :------------------------------------------------------------------
    """
    a = np.arange(12).reshape(3, 4)
    b0 = np.array([1, 2, 3, 4])
    c = np.concatenate([a, b0[:3].reshape(a.shape[0], 1)], axis=1)
    d = np.concatenate([a, b0.reshape(1, a.shape[1])], axis=0)                        
    doc_string = dedent(num_24.__doc__)
    print(dedent(frmt).format(dedent(doc_string), a, b0[:3], c, b0, d))

#-----------------------------------------------------------------------
# 
def py_01():
    """(py_01)... List comprehension demonstration with conditions...
    : http://stackoverflow.com/questions/32290060/list-comprehensions-tip/
    :       32291666#32291666\n
    :Conventional code....\n
    :  c = []  
    :  for i in range(10):  
    :      if (i > 2) and (i < 8):  
    :          c.append(i**2)
    :
    :List comprehensions.....\n
    :  a = [i**2 for i in range(10) if i > 2 and i < 8]\n
    :  b = [i**2               # do this  
    :       for i in range(10) # using these  
    :       if (i > 2) and     # where this and  
    :          (i < 8)         # this is good
    :      ]                   # that is all
    """
    # conventional LC 
    a = [i**2 for i in range(10) if i > 2 and i < 8]
    # alternate with multiline presentation
    b = [i**2               # do this  
         for i in range(10) # using these  
         if (i > 2) and     # where this and  
            (i < 8)         # this is good
        ]                   # that is all
    frmt = """
    :------------------------------------------------------------------
    {}
    :List comprehension formats...\n
    :  (a) typical list. comp...{}\n
    :  (b) multi-line version...{}\n 
    :      does a == b == c???  {}
    :------------------------------------------------------------------
    """
    c = []  
    for i in range(10):  
        if (i > 2) and (i < 8):  
            c.append(i**2)  
    # are they all good?      
    print(dedent(frmt).format(py_01.__doc__, a, b, a==b))

#-----------------------------------------------------------------------
def py_02():
    """(py_02)... List comprehensions alternate outputs
    from
    - http://stackoverflow.com/questions/35215024/attempting-python-
       list-comprehension-with-two-variable-of-different-ranges
    """
    a = [0, 1, 2, 3, 4, 5]
    b = [0, 1, 2]
    out1 = [(a[x], b[x % len(b)]) for x in range(len(a))]
    out2 = [(a[x % len(a)], b[x % len(b)])
            for x in range(max(len(a), len(b)))]
    #print("out1 {}\nout2 {}".format(out1,out2))
    """ 
      ET < 28): return  1
      ET >= 28 and ET < 91: return 2
      ET >= 91 and ET < 182: return 3
      ET >= 182 and ET < 365: return 4
      ET >= 365): 5
      another
      return ( (a + abs(a-b)) if ((a-b) <= 1) else min(a, (a-b)) ) 
    """
    dts = [1, 28, 29, 91, 92, 182, 183, 365, 366]
    bins = [28, 91, 182, 365, 400]
    #bins = np.array(b_ins)
   
    c = np.digitize(dts, bins, right=True) + 1
    #
    import bisect    
    d = [bisect.bisect_left(bins, i) + 1 for i in dts]
    return a, b, c, d
#-----------------------------------------------------------------------
def py_03():
    """(py_03)... Formatting output values using the textwrap module from
    - https://docs.python.org/3.4/library/textwrap.html#module-textwrap
    - inputs
    - outputs
    Read the output for more tips....
    """
    from textwrap import dedent
    a = np.arange(25, 0, -1).reshape((5, 5))
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array........
    : - shape: {}  ndim: {}
    {}   
    :The key is to keep the leading spaces on each line equal
    : and put curly brackets on separate lines rather
    : than using a backslash n ... but if you have to, put the specified
    : number of spaces before the newline character.
    : - Notice that I also dedented the docstring and I didn't start on a
    : - new line   and so on... then I used dedent.
    : ....
    :------------------------------------------------------------------
    """
    doc_string = dedent(py_03.__doc__)
    #frmt = frmt.replace("\n    ","\n") 
    print(dedent(frmt).format(doc_string, a.shape, a.ndim, a))
    return frmt


#-----------------------------------------------------------------------
# num_XX see below
"""
http://stackoverflow.com/questions/32661348/numpy-selecting-every-other-n-entries
Question:
  Does anyone perhaps know a clever way to select every other group of
  n entries in a numpy array? For example here Id like to select all "unique" 
  entries 0,1,4,5,8,9 etc. (n=2) without resorting to a sort:
      
final result see printout for steps
[[ 0  2]   0
 [ 1  3]   1
 [ 2  0]
 [ 3  1]
 [ 4  6]   4
 [ 5  7]   4
 [ 6  4]
 [ 7  5]
 [ 8 10]   8
 [ 9 11]   9
 [10  8]
 [11  9]
 [12 14]
 [13 15]
 [14 12]
 [15 13]]
In [12]: x.reshape(-1, 2, 2)[::2].reshape(-1, 2)
Out[12]: 
array([[ 0,  2],
       [ 1,  3],
       [ 4,  6],
       [ 5,  7],
       [ 8, 10],
       [ 9, 11],
       [12, 14],
       [13, 15]])
"""
def num_xx():
    """Use even numbers to see the affect of reshaping arrays.
    """
    np.set_printoptions(edgeitems=10, linewidth=80, threshold=20)
    N = 12
    idx = np.arange(N)
    vals = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]
    b = np.array(vals[:N])
    a = np.vstack((idx, b))
    x = a.reshape(2, 2, -1)
    y = x[::2]     # or a.reshape(2, 2, -1)[::2]
    z = y.reshape(2, -1)  # or a.reshape(-1, 2, 2)[::2]).reshape(-1, 2)
    print("Input array...arr: shape={}\n{}\n".format(a.shape,a))
    print("x = arr.reshape(2,2,-1) = {}\n{}\n".format(x.shape,x))
    print("slice some x...\n...y = x[::2]  y.shape={}\n{}\n".format(y.shape, y))
    print("reshape the slice...\n...z = y.reshape(2, -1) = {}\n{}".format(z.shape, z))
    #
    return a

#-------------------------------------------------------------------
# mpl_01
# http://matplotlib.org/mpl_examples/pylab_examples/griddata_demo.py
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np

def mpl_01():
    """(mpl_01) Plotting and interpolating"""
    print("\n{}".format(mpl_01.__doc__))

    a_s = np.array([(0.1, 0.1, 8), (0.1, 8.9, 8), 
                    (1.0, 1.0, 6.5), (2.0, 3.0, 5.2), 
                    (2., 7, 7.8), (6.5, 2.0, 2.3), 
                    (6.0, 8.0, 8), (7.0, 6.0, 5.5), 
                    (8.9, 0.1, 1), (8.9, 8.9, 8)],
                    dtype=[('X','<f8'),('Y','<f8'),('Z','<f8')])
    print(a_s)
    x = a_s['X']
    y = a_s['Y']
    z = a_s['Z']  # from above
    # define grid.
    L = np.min(x)
    R = np.max(x)
    B = np.max(y)
    T = np.min(y)
    L, B = np.round([L, B])
    R, T = np.round([R, T])
    xi = np.linspace(L, R, 100)
    yi = np.linspace(B, T, 100)
    # grid the data.
    fig, ax = plt.subplots(figsize=(5, 5))  # set the figure siz in inches 
    ax.set_xlim(L, R)
    ax.set_ylim(B, T)
    ax.tick_params(direction='out', length=6, width=1, colors='k')
    zi = griddata(x, y, z, xi, yi, interp='linear') #natural neigh or linear')
    print("grid min = {} grid max = {}".format(np.min(zi),np.max(zi)))
    # contour the gridded data, plotting dots at the nonuniform data points.
    ax.autoscale('True')
    #CS = plt.contour(xi, yi, zi, 15, linewidths=0.5, colors='k') # black and white
    vmin = 0
    vmax = 10
    #vmax=abs(zi).max(); vmin=-abs(zi).max()
    CS = plt.contourf(xi, yi, zi, 15, cmap=plt.cm.rainbow, vmax=vmax , vmin=vmin)
    plt.contour(xi, yi, zi)
    CS.levels = np.arange(vmax).tolist()
    u = [str(i) for i in CS.levels]
    CS.levels = [str(i) for i in u]
    plt.clabel(CS, CS.levels, inline=True, fontsize=10)

    #plt.colorbar()  # draw colorbar
    # plot data points.
    plt.scatter(x, y, marker='s', c='b', s=20, zorder=10)
    #plt.subplot.set_aspect(1.)
    plt.axis([L, R, B, T])
    #plt.axis('equal')
    plt.xlim(L, R)
    plt.ylim(B, T)
    plt.title("griddata test (%d points)" % len(z))
    plt.show()
    #del zi,griddata,CS,plt

#-----------------------------------------------------------------------
# mpl_02 construct geometries demo

def plot_3d(x, y, z, title="Title", x_lbl="X", y_lbl="Y", z_lbl="Z"):
    """3D plot
    plot_scores_3d(x, y, z, 'Test')
    """
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(y_lbl)
    ax.set_zlabel(z_lbl)
    plt.show()
    plt.close()

def mpl_02():
    """(mpl_02)... Construct geometries demo
    - http://stackoverflow.com/questions/18107608/
      what-is-the-pythonic-way-of-generating-this-type-of-
      list-faces-of-an-n-cube
    array([[ 1,  0,  0],
           [-1,  0,  0],
           [ 0,  1,  0],
           [ 0, -1,  0],
           [ 0,  0,  1],
           [ 0,  0, -1]])    
    """
    from textwrap import dedent
    frmt = """
    :------------------------------------------------------------------
    {}
    :Input array........
    : - shape: {: <10}  ndim: {}
    {}
    :Output array.......
    {}
    {}
    :------------------------------------------------------------------
    """
    cube_nodes = [(x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)]
    a = np.array(cube_nodes)
    x = a[:,0]
    y = a[:,1]
    z = a[:,2]
    frmt = dedent(frmt)
    doc_string = dedent(mpl_02.__doc__)
    #print(frmt.format(doc_string, a.shape, a.ndim, cube_nodes, a))
    plot_3d(x, y, z, 'Test')
    return a


#-----------------------------------------------------------------------
if __name__=="__main__":
    """uncomment the one you want to test"""
    # ......  construction .....
    #num_01() # Array creation using vstack, zip and array filling
    #num_02() # Using np.linspace with floats instead of np.arange
    #num_03() # sparse array
    #num_04() # Sub-dtypes in numpy and array formulation
    #num_05() # recarray access in numpy
    # ......  altering ......
    #num_06() # Changing array types via rounding etc
    #num_07() # Flatten an array
    #num_08() # Array size information
    #num_09() # Transposing 3D arrays
    #num_10() # Array slicing using the ellipse
    #num_11() # Slicing arrays
    #num_12() # Sorting an array
    #num_13() # Sorting an array revisited (see # 12)
    #num_14() # Array padding example
    # ......  working with array data ......
    #num_15() # Subtracting an array mean and the array
    #num_16() # Unique values for 1D and 2D arrays
    #num_17() # Striding arrays demo
    #num_18() # Condition checking and useage now in numpy.
    #num_19() # Using linalg, einsum, distance and timing
    #num_20() # Using fromiter, unique and histo all at once.
    #num_21() # Reorder, delete rows/columns using slicing
    #num_22() # Reclass arrays
    #num_23() # Block statistics
    #num_24() # Concatenate arrays
    #a = num_xx() # reshaping stuff
    # ...... python ......
    #py_01()  # List comprehension formats
    #py_02()  # List comprehension alternate outputs
    #py_03()  # formatting output with textwrap
    #
    # ....... matplotlib ......
    #mpl_01() # Plotting and interpolating
    #mpl_02() # Construct rectangular geometries
    #
    # to do np.polynomial
    
    """a = np.array([[-2, 1], [2, 1], [1, -2], [1, 2], [1,1]])
    b = np.array([[3, 3], [5, 5]])
    c = np.vstack((a + b[:,None,:]))
    d = a + b[:,None,:]  # produces a 3D array
    e = (a + b[:,None,:]).reshape(-1, a.shape[1])
    f = np.vstack((a*b[:, None,:]))
    g = a[:,None,:] + b
    a = np.array([1, 2, 1, 2, 2, 2, 1, 1, 1, 2, 3, 4, 5, 1, 1])
    """

