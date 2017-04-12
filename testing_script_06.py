# -*- coding: UTF-8 -*-
"""
:Script:   my_new_scripts.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-03-12
:
:Purpose:
:
:Functions list .........
:...... np functions .....
:    num_86()  # standardize by rows or columns
:    num_87()  # Unique in 3d array
:    num_88()  # nested recarrays
:    num_89()  # reshape array to row format
:    num_90()  #
:    num_91()  #
:    num_92()  #
:    num_93()  #
:    num_94()  #
:    num_95()  #
:    num_96()  #
:    num_97()  #
:    num_98()  #
:    num_99()
:Notes:
:
:References
:
:
:---------------------------------------------------------------------:
:Notes:
:
:References:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
import datetime
import numpy as np
from textwrap import dedent, indent
import arraytools as art

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

# ---- functions ----


# ----------------------------------------------------------------------
# num_86
def num_86():
    """(num_86)... standardize by rows or columns
    :Divide rows by row total or column by column totals
    :Requires:
    :--------
    :
    :Returns:
    :-------
    :
    """
    frmt = """
    {}\n
    :input array 'a'...
    {}\n
    :row totals 'b' ... a.sum(axis=2, keepdims=True)
    {}\n
    :col totals 'c' ... a.sum(axis=1, keepdims=True)
    {}\n
    :normalized by row... 'a/d'
    {}\n
    :normalized by col... 'a/c'
    {}\n
    """
    a = np.arange(2*3*3).reshape(2, 3, 3)
    b = a.sum(axis=2, keepdims=True)    # row totals, keeping dimensions
    c = a.sum(axis=1, keepdims=True)    # col totals, keeping dimensions
    d = a/b  # or a/a.sum(axis=2, keepdims=1).astype(np.float)
    e = a/c
    ar = [art.in_by(i) for i in [a, b, c, d, e]]
    args = [num_86.__doc__, *ar]  # b, c, d, e]
    print(dedent(frmt).format(*args))
#    return a, b, c, d, e


# ----------------------------------------------------------------------
# num_87
def num_87():
    """(num_87)... Unique in 3d array
    :  Reshape the 3d array to a series of rows, convert each row to a
    :  tuple and process.  May not be the quickest, but it works
    :Requires:
    :--------
    :
    :Returns:
    :-------
    :Reference:
    :---------
    :  http://stackoverflow.com/questions/31097247/remove-duplicate
    :       -rows-of-a-numpy-array *** best, ref from
    :  http://stackoverflow.com/questions/41071116/how-to-remove-
    :       duplicates-from-a-3d-array-in-python
    :
    :  sample data
    :  a = [[[1, 2], [1, 2]], [[1, 2], [4, 5]], [[1, 2], [1, 2]]]
    :  b = numpy.array(a)
    :
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}\n
    :Input array... shape:{}
    {}\n
    :Reshaped for analysis... shape:{}
    {}
    :uniques in the dimensions... shape:{}
    {}
    :---------------------------------------------------------------------:
    """
    data = [[[1, 2], [1, 2]], [[1, 2], [4, 5]], [[1, 2], [1, 2]]]
    a = np.array(data)
    if a.ndim == 3:
        b = a.reshape(a.shape[0], -1)
    out = [tuple(row) for row in b]
    c = np.unique(out)
    a0, b0, c0 = [[i.shape, art.in_by(i)] for i in [a, b, c]]
    args = [num_87.__doc__, *a0, *b0, *c0]
    print(dedent(frmt).format(*args))
#    return a, b, c


# ----------------------------------------------------------------------
# num_88 ... nested recarrays
def num_88():
    """(num_88)... nested recarrays
    :Form structured/recarrays with nested dtypes
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}\n
    :Input data...
    {}\n
    :recarray...
    {!r:}\n
    :from to_struct...
    {}
    :zeros array...
    {!r:}\n
    :---------------------------------------------------------------------:
    """

    def flatten(container):
        """recursively flatten containers"""
        for i in container:
            if isinstance(i, (list, tuple, np.ndarray)):
                for j in flatten(i):
                    yield j
            else:
                yield i
    # print list(flatten(nests))

    def to_struct(a, data):
        """ """
        data = np.atleast_1d(data)
        nms = a.dtype.names
        for i in range(len(data)):
            for j in range(i):
                val = data[i, j]
                if isinstance(val, (list, tuple)):
                    a[nms[i]] = tuple(val)
                elif isinstance(val, (np.ndarray)):
                    a[nms[i]] = tuple(val.tolist())[0]
                else:
                    print("not good value {}".format(val))
        return a
    data = [[(1, 2), 3, 4], [(5, 6), 7, 8]]
    a0 = np.recarray((2,), dtype=[('shp', float, (2,)),
                                  ('y', 'float64'), ('z', 'int32')])
    ar = to_struct(a0, data)
    a1 = np.zeros((2,), dtype=[('shp', [('X', float), ('Y', float)]),
                               ('a', float), ('b', int)])
    args = [num_88.__doc__, data, a0, ar, a1]
    print(dedent(frmt).format(*args))
    # return data, a0, a1


# ----------------------------------------------------------------------
# num_89... reshaping from column to row
def num_89():
    """Reshape an array from column to row format collapsing the number
    :  of dimensions by one
    :References:
    :----------
    :  http://stackoverflow.com/questions/41904598/how-to-rearrange-an-
    :         array-by-subarray-elegantly-in-numpy
    :
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}\n    :input array...\n    {}\n\n    :Output array...\n    {}
    :---------------------------------------------------------------------:
    """
    a = np.arange(3*4*5).reshape(3, 4, 5)
    b = a.swapaxes(0, 1).reshape(a.shape[1], -1)
    args = [num_89.__doc__, a, b]
    print(dedent(frmt).format(*args))
    # return a, b


# ----------------------------------------------------------------------
# num_90 ... sorting two dimensional arrays, lexsort etc
def num_90():
    """num_90() sorting two dimensional arrays, lexsort etc
    :Requires:
    :--------
    :
    :Returns:
    :-------
    :References:
    :-----------
    :  http://stackoverflow.com/questions/41903502/
    :         sort-two-dimensional-list-python
    :Notes:
    :-----
    :  sorting - np.lexsort((a[:,1], a[:,0])) sort by x, then y
    :            np.lexsort(a.T) >= np.lexsort((a[:,0], a[:,1])) sort y, x
    :Distances
    : unsorted
    :   np.linalg.norm(a[1:] - a[:-1], axis=1)
    :   array([ 166.027,  165.000,  164.003,  164.000,  165.003,  162.000,
    :           162.003, 162.000,  164.003])
    :   np.sum(np.linalg.norm(a[1:] - a[:-1], axis=1)) => 1474.0393...
    : sorted
    :   a_srt = a[np.lexsort(a.T),:]
    :   np.linalg.norm(a_srt[1:] - a_srt[:-1], axis=1)
    :   array([ 4.472,  2.000,  1.000,  2.236,  161.003,  1.000,  4.243,
    :           1.000,  4.123])
    :   np.sum(np.linalg.norm(a_srt[1:] - a_srt[:-1], axis=1)) => 181.0770...
    : OP's sorting process
    :   s_a = [[45,205], [42,206], [46,205], [47,202], [48,202],
    :               [45,40], [46,41], [47,40], [48,40], [49,38]]
    :  s_a = np.array(sorted_a)
    :  np.linalg.norm(s_a[1:] - s_a[:-1], axis=1)
    :  array([ 3.162,  4.123,  3.162,  1.000,  162.028,  1.414,  1.414,
    :         1.000,  2.236])
    :  np.sum(np.linalg.norm(s_a[1:] - s_a[:-1], axis=1)) => 179.5399
    :
    :Near results...
    :------------
    :coords, dist, n_array = n_near(s, N=2)
    :  n_array
    :  array(
    : [(0, 42.0, 206.0, 45.0, 205.0, 46.0, 205.0, 3.1622, 4.1231),
    :  (1, 45.0, 205.0, 46.0, 205.0, 42.0, 206.0, 1.0,    3.1622),
    :  (2, 45.0,  40.0, 46.0,  41.0, 47.0,  40.0, 1.4142, 2.0),
    :  (3, 46.0, 205.0, 45.0, 205.0, 47.0, 202.0, 1.0,    3.16227),
    :  (4, 46.0,  41.0, 45.0,  40.0, 47.0,  40.0, 1.4142, 1.4142),
    :  (5, 47.0, 202.0, 48.0, 202.0, 46.0, 205.0, 1.0,    3.1622),
    :  (6, 47.0,  40.0, 48.0,  40.0, 46.0,  41.0, 1.0,    1.41421),
    :  (7, 48.0, 202.0, 47.0, 202.0, 46.0, 205.0, 1.0,    3.60555),
    :  (8, 48.0,  40.0, 47.0,  40.0, 46.0,  41.0, 1.0,    2.23606),
    :  (9, 49.0,  38.0, 48.0,  40.0, 47.0,  40.0, 2.2360, 2.8284)],
    :  dtype=[('ID', '<i4'), ('Xo', '<f8'), ('Yo', '<f8'), ('C0_X', '<f8'),
    :         ('C0_Y', '<f8'), ('C1_X', '<f8'), ('C1_Y', '<f8'),
    :         ('Dist0', '<f8'), ('Dist1', '<f8')])

    :  0 - 2 3.2   5 - 7 1   2 - 5  2   connects all
       1 - 3 1.4   6 - 8 1
       2 - 4 1.0   7 - 9 2.2  sum([3.2, 1.4, 1, 1.4, 3.2,
       3 - 5 1.4   8 - 4 3.6       1, 1, 2.2, 3.6, 2.8]) => 20.8
       4 - 6 3.2   9 - 5 2.8
        Distance sorted... _d_dist(d2), _e_dist(d3) np.allclose = True
d0.astype('float')
      0     1      2      3       4    5       6      7      8     9
 0 [ 0.0, 166.0,  3.2, 165.0,    4.1, 166.1,   6.4, 166.1,   7.2, 168.1],
 1 [ 0.0,   0.0, 165.0,   1.4, 165.0,   2.0, 162.0,   3.0, 162.0,   4.5],
 2 [ 0.0,   0.0,   0.0, 164.0,   1.0, 165.0,   3.6, 165.0,   4.2, 167.0],
 3 [ 0.0,   0.0,   0.0,   0.0, 164.0,   1.4, 161.0,   2.2,  161.0,  4.2],
 4 [ 0.0,   0.0,   0.0,   0.0,   0.0, 165.0,   3.2, 165.0,   3.6, 167.0],
 5 [ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 162.0,   1.0, 162.0,   2.8],
 6 [ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 162.0,   1.0, 164.0],
 7 [ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 162.0,   2.2],
 8 [ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0, 164.0],
 9 [ 0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0]])
    :  np.sum(n_array['Dist0']) => 14.22677276241436
    :  np.sum(n_array['Dist1']) => 27.10841210857896
    :
    :  x0 = n_array['Xo']
    :  y0 = n_array['Yo']
    :  s_c = np.array(list(zip(x0,y0)))
    :  array([[ 42.0,  206.0],
    :         [ 45.0,  205.0],
    :         [ 45.0,   40.0],
    :         [ 46.0,  205.0],
    :         [ 46.0,   41.0],
    :         [ 47.0,  202.0],
    :         [ 47.0,   40.0],
    :         [ 48.0,  202.0],
    :         [ 48.0,   40.0],
    :         [ 49.0,   38.0]])
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :Input array...
    {}
    :Distance array... _d_dist(d0), _e_dist(d1) np.allclose = {}
    {}
    :Sorted array...
    {}
    :Distance sorted... _d_dist(d2), _e_dist(d3) np.allclose = {}
    {}
    :---------------------------------------------------------------------:
    """
    def _d_dist(a):
        """2D array pairwise distance.  no error checking"""
        r, c = a.shape
        d = np.zeros((r, r))
        for i in range(r):
            for j in range(i, r):
                d[i, j] = np.sqrt(sum((a[i, :] - a[j, :])**2))
                # d[i,j] = np.linalg.norm(a[i, :] - a[j, :])
        return d
    #
    def _e_dist(a):
        """see e_dist in ein_geom.py"""
        b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
        diff = a - b
        d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff)).squeeze()
        d = np.triu(d)
        return d
    #
    """ sorted_a = [[45, 205], [42,206], [46,205], [47,202], [48,202],
                    [45,40], [46,41], [47,40], [48,40], [49,38]]
    """
    a = np.array([[42, 206], [45, 40], [45, 205], [46, 41], [46, 205],
                  [47, 40], [47, 202], [48, 40], [48, 202], [49, 38]])
    """
    d = np.zeros((10,10))
    for i in range(10):
        for j in range(i,10):
            d[i,j] = np.sqrt(sum((a[i,:]-a[j,:])**2))
    """
    d0 = _d_dist(a)
    d1 = _e_dist(a)
    idx = np.lexsort(a.T)  # = np.lexsort((a[:,0], a[:,1])) sort y, then x
    a_srt = a[idx, :]
    d0a = _d_dist(a_srt)
    d1a = _e_dist(a_srt)
    args = [num_90.__doc__,
            a, np.allclose(d0, d1), d0.astype('int'),
            a_srt, np.allclose(d0a, d1a), d1a.astype('int')]
    print(dedent(frmt).format(*args))
    return a, d0, d1a


# ----------------------------------------------------------------------
# num_91 ... smallest 'x' values in 2D numpy array
def num_91():
    """Return the 'x' number of values from a 2D array using triu_indices
    :  np.argsort and slicing.
    :
    """
    a = np.array([[6, 0, 3, 9, 8],
                  [8, 7, 7, 4, 8],
                  [5, 3, 7, 9, 1],
                  [9, 1, 2, 9, 3],
                 [4, 5, 1, 4, 9]])
    # a = np.random.randint(0, 10, size=(5,5))
    rows, cols = np.triu_indices(a.shape[1], 1)
    idx = a[rows, cols].argsort()[:4]
    r, c = rows[idx], cols[idx]
    out = list(zip(r, c, a[r, c]))
    idx2 = a[rows, cols].argsort()[4:]
    r, c = rows[idx2], cols[idx2]
    out2 = list(zip(r, c, a[r, c]))
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :Input 5x5 array with random values in the range 0-10
    {}\n
    :Smallest 5 values and their Y, X (row, col) location...
    {}\n
    :Largest 3 values...
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_91.__doc__, a, np.array(out), np.array(out2)]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_92 gray scale image from rgb
def num_92():
    """num_92... gray-scale image from rgb
    :Essentially gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    : np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    : http://stackoverflow.com/questions/12201577/how-can-i-convert
    :       -an-rgb-image-into-grayscale-in-python
    : https://en.m.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    : see https://en.m.wikipedia.org/wiki/HSL_and_HSV
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """
    import matplotlib.pyplot as plt
    a = np.arange(256).reshape(16, 16)
    b = a[::-1]
    c = np.ones_like(a)*128
    rgb = np.dstack((a, b, c))
    gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
    plt.imshow(gray, cmap=plt.get_cmap('gray'))
    plt.show()
    args = [_demo.__doc__]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_93
def num_93():
    """
    convolve 2d array a with kernel

    1/8 1/8 1/8
    1/8  0  1/8
    1/8 1/8 1/8
    """
    a = np.arange(36).reshape(6, 6)
    tmp = a.copy()
    tmp[:, 1:] += a[:, :-1]
    tmp[:, :-1] += a[:, 1:]
    out = tmp.copy()
    out[1:, :] += tmp[:-1, :]
    out[:-1, :] += tmp[1:, :]
    result = (out - a) / 8.
    frmt = """
    :---------------------------------------------------------------------:
    {}
    : input
    {}\n
    : output
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_93.__doc__, a, result]
    print(dedent(frmt).format(*args))
    # return result


# ----------------------------------------------------------------------
# num_94 many plots on screen
def num_94():
    """
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(figsize=(10, 10), sharex=True, sharey=True,
                             ncols=3, nrows=3)
    x = np.linspace(0, 10, 100)
    for i in range(3):
        for j in range(3):
            if i < j:
                axes[i, j].axis('off')
            else:
                axes[i, j].plot(x, np.sin((i + j) * x))
    plt.show()
    plt.close()

    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_94.__doc__]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_95 ... working with dates
def num_95():
    """
    """
    import datetime
    from calendar import monthrange
    yr = 2017
    cal = []
    for m in range(1, 2):  # by month
        base = datetime.datetime(yr, m, 1)
        d_max = monthrange(yr, m)[1]
        for d in range(1, d_max + 1):  # by day
            ymd = datetime.datetime(yr, m, d)
            cal.append(ymd)
        # print("date {} days in month {}".format(base, max_days))
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_95.__doc__]
    # print(dedent(frmt).format(*args))
    return cal


# ----------------------------------------------------------------------
# num_96  n smallest in column in sorted order
def num_96():
    """Find the n smallest values in an array by column.
    :
    :Reference
    :---------
    :  http://stackoverflow.com/questions/42874944/geting-the-k-smallest
    :       -values-of-each-column-in-sorted-order-using-numpy-argparti
    """
    a = np.array([[1, 3, 2, 5, 7, 0],
                  [14, 15, 6, 5, 7, 0],
                  [17, 8, 9, 5, 7, 0]])
    num = 3
    idx = np.argpartition(a, range(num), axis=1)[:, :num]
    out = a[np.arange(idx.shape[0])[:,None], idx]
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :Input array...
    : {} smallest by column for array
    {}
    :Indices......
    {}
    :Result
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_96.__doc__, num, a, idx, out]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_97
def num_97():
    """
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_97.__doc__]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_98
def num_98():
    """
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_98.__doc__]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_99
def num_99():
    """
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_99.__doc__]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_100
def num_100():
    """
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_100.__doc__]
    print(dedent(frmt).format(*args))


# ---------------------------------------------------------------------
if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#    num_86()  # standardize by rows or columns
#    num_87()  # Unique in 3d array
#    a, a0, a1 = num_88()  # nested recarrays
#    a, b = num_89()  # reshape arrays to row format
#    a, d0, d1 = num_90()
#    num_91()
#    num_92()
#    num_93()
#    num_94()
#    cal = num_95()
    num_96()  # n smallest in column in sorted order
"""
a = list('!!!feirg ...uoy taeb redyR DNA ardnaiD !!!yllanif')
arr = np.array(a).reshape(7, 7)
answer = arr.flatten()[::-1]
aa = "".join([i for i in answer])
print(aa)
#print("{}".format('uoy knahT'[::-1]))
"""
