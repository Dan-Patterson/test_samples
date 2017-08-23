# -*- coding: UTF-8 -*-
"""
:Script:   testing_script_07.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-08-23
:
:Purpose:
:
:Functions list .........
:...... np functions .....
:    num_101()  # sum product, einsum by axis
:    num_102()  # Structured to ndarray demo
:    num_103()  # sample plot
:    num_104()  # fancy slicing
:    num_105()  # distance calculation, with large arrays
:    num_106()  # reset counter demo
:    num_107()  # 2D argmax use
:    num_108()  # voxel in matplot lib... not available yet
:    num_109()  # raster band math demo
:    num_110()  # Where or where is....
:
:Notes:
:
:References
:
:
:---------------------------------------------------------------------:
"""
# ---- imports, formats, constants ----

import sys
# import datetime
import numpy as np
from textwrap import dedent  # , indent
import arraytools as art

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]  # print this should you need to locate the script

# ---- functions ----


# ----------------------------------------------------------------------
# num_101 sum product, einsum by axis
def num_101():
    """sum product, einsum by axis """
    a = np.arange(1, 11).reshape(5, 2)
    b = np.array([[2, 3]])
    c = np.sum(a*b, axis=1)
    d = np.sum(a*b, axis=0)
    e = np.einsum('ij, ij->i', a, b)
    f = np.einsum('ij, ij->j', a, b)
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :input arrays...
    :a...
    {}
    :b...
    {}
    :sum product by row and col... c, d
    {} ... {}
    :einsum by row and col... e, f
    {} ... {}
    :---------------------------------------------------------------------:
    """
    args = [num_101.__doc__, a, b, c, d, e, f]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_102 structured to ndarry
def num_102():
    """Structured to ndarray demo
    : Notes: to produce a structured array, make sure you are using a
    :    list of tuples.  To reshape back to an ndarray, IF! the dtype is the
    : same, you need to snag the first dtype, then reshape it using the
    : number of records, then -1 to get the number of columns.
    :
    : base : ndarray
    :  If the array is a view into another array, that array is its `base`
    :  (unless that array is also a view).  The `base` array is where the
    :  array data is actually stored.
    """
    dt = [('A', '<i4'), ('B', '<i4'), ('C', '<i4')]
    a = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)], dtype=dt)
    b = a.view(a.dtype[0]).reshape(a.shape[0], -1)
    c = a == b.base
    frmt = """
    {}\n
    :Input structured array (a)...
    {!r}\n
    :Array viewed as ndarray (b)...
    {!r}\n
    :Base of array (b) == (a) all true? {}
    {!r}
    """
    args = [num_102.__doc__, a, b, np.alltrue(c), c]
    print(dedent(frmt).format(*args))
    return a, b, c


# ----------------------------------------------------------------------
# num_103  sample plot
def num_103():
    """Sample plot from art.plot_pnts_ in arraytools
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """
    xy = np.random.randint(1, 100, size=100).reshape(50, 2)
    art.plot_pnts_(xy, title='Sample points')
    args = [num_103.__doc__]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_104  fancy slicing
def num_104():
    """
    :Fancy slicing keeping dimensions.  First method is to use a scalar
    : and the second is to use a list.  From the docs...
    : 'An integer, i, returns the same values as i:i+1 except the
    : dimensionality of the returned object is reduced by 1.'
    : https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :Input array... z = np.arange(2*3*4).reshape(2, 3, 4)
    {}\n
    :Slice with scalar...   z[:, 0, :]
    {}\n
    :Slice with list...     z[:, [0], :]
    {}\n
    :Slice more than 1...   z[:, [0, 2], :]
    {}\n
    :Slice and rearrange... z[:, [2, 0], :]
    {}\n
    :Slice and repeat... z[:, [2, 2, 1], :]
    {}
    :---------------------------------------------------------------------:
    """
    z = np.arange(2*3*4).reshape(2, 3, 4)
    z0 = z[:, 0, :]
    z1 = z[:, [0], :]
    z2 = z[:, [0, 2], :]
    z3 = z[:, [2, 0], :]
    z4 = z[:, [2, 2, 1], :]
    args = [dedent(num_104.__doc__), z, z0, z1, z2, z3, z4]
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_105 distance calculation, with large arrays
# from scipy.spatial.distance import cdist
def num_105():
    """Large file data test for spatial distance calculation
    :  Using abbreviated e_dist calculation methods
    """
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :origin {}
    :destinations (first 5)
    {}
    :distances (first 5)
    {}
    :cdist and e_dist all close? {}
    :---------------------------------------------------------------------:
    """
    from scipy.spatial.distance import cdist
    N = 5  # 50000000
    a = np.random.mtrand.RandomState(2).randint(0, 10, size=(1, 2))
    b = np.random.mtrand.RandomState(1).randint(0, 10, size=(N, 2))
    d = cdist(a, b)
    c = b.reshape(np.prod(b.shape[:-1]), 1, b.shape[-1])
    diff = c - a
    dist_arr = np.einsum('ijk,ijk->ij', diff, diff)
    dist_arr = np.sqrt(dist_arr).squeeze()
    all_ = np.allclose(d.squeeze(), dist_arr)
    args = [num_105.__doc__, a, b[:5], d[:5], all_]
    print(dedent(frmt).format(*args))
    return a, b, c


# ----------------------------------------------------------------------
# num_106 reset counter demo
def num_106():
    """a demo of resetting a counter given an input list/array/field"""
    vals = ['a', 'a', 'a', 'b', 'b', 'c', 'd', 'd', 'd', 'e']
#    cnt = 0

    global old
    old = ""
    cnt = 0

    def func(val, old, cnt):
        # print("is first {}".format(is_first))
        if old == val:
            cnt += 1
            ret = "{} {: 05.0f}".format(val, cnt)
        else:
            cnt = 0
            ret = "{} {: 05.0f}".format(val, cnt)
        old = val
        return ret, old, cnt
    for val in vals:
        ret, old, cnt = func(val, old, cnt)
        print(ret, old)
    del old
#    frmt = """
#    :---------------------------------------------------------------------:
#    {}
#    :---------------------------------------------------------------------:
#    """
#    args = [num_106.__doc__]
#    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_107 2D argmax use
def num_107():
    """
    2D argmax to return the row, col of the maximum in a 2D array.
    """
    a = np.random.randint(1, 10, size=(10, 10))
    m0 = a.argmax()
    m1 = np.argmax(a, axis=0)
    m2 = np.argmax(a, axis=1)
    m3 = np.unravel_index(a.argmax(), a.shape)

    frmt = """
    {}
    :Input array...
    {}
    :First occurance of the max is in the ... {} element unravelled
    :
    :maximum for the col {}
    :maximum for the row {}
    :unravel the index for the row and column ... {}
    """
    args = [dedent(num_107.__doc__), a, m0, m1, m2, m3]
    print(dedent(frmt).format(*args))
    return a


# ----------------------------------------------------------------------
# num_108 3D raster plot demo
def num_108():
    """a demo of plotting a 3D raster
    :https://stackoverflow.com/questions/44825276/
    :      visualizing-a-3d-numpy-array-of-1s-and-0s
    :https://github.com/matplotlib/matplotlib/pull/6404
    :https://github.com/matplotlib/matplotlib/pull/6404/files
    :
    : *** doesn't work yet, but keep an eye on voxels
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # your real data here - some 3d boolean array
    x, y, z = np.indices((10, 10, 10))
    voxels = (x == y) | (y == z)

    ax.voxels(voxels)

    plt.show()


# ----------------------------------------------------------------------
# num_109 raster band math
def num_109():
    """raster band math... just a quick demo
    : note: Bands are number from 0 to n,
    : references
    : http://conteudo.icmc.usp.br/pessoas/moacir/papers/Ponti_GRSL2013.pdf
    : http://www.sciencedirect.com/science/article/pii/S2214317315000347
    :
    :a = np.zeros(shape=(5, 5, 6), dtype='int32')
    a[..., 0] = vals[0]  # 1
    a[..., 1] = vals[1]  # 2
    a[..., 5] = vals[2]  # 6
    """
    import arcpy
    import matplotlib.pyplot as plt
    f = r'C:\Book_Materials\images\rgb.png'  # an image
    arcpy.overwriteOutput = True
    bands = arcpy.RasterToNumPyArray(f)
    a = bands[:3, :]
    r = bands[0, :]
    g = bands[1, :]
    b = bands[2, :]
    k = bands[3, :]
    # normalize
    denom = (r + g + b)
    denom = denom.astype('float')
    r1 = r / (denom)
    g1 = g / (denom)
    b1 = b / (denom)
    args = [r, g, b]
    print("Band 1\n{}\nBand 2\n{}\nBand 6\n{}".format(*args))
    cive = (0.441 * r) - (0.881 * g) + (0.385 * b) + 18.78745
    cive_2 = np.zeros(cive.shape, dtype='int32')
    cive_2 = np.zeros(cive.shape, dtype='int32')
    # exg = 2*g1 - r1 - b1
    # rgb = np.dstack((r, g, b))
    print("CIVE result...\n{}\n ExG result...\n{}".format(cive, cive_2))
    #out = arcpy.NumPyArrayToRaster(cive_2)  # uncomment to save
    #out.save(f.replace('.png', '1.tif'))
    #gray = np.dot(a[..., :3], [0.2989, 0.5870, 0.1140])
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.astype('uint8')
    plt.imshow(gray, cmap=plt.get_cmap('gray'))
    return a, r, g, b, k, gray


# ----------------------------------------------------------------------
# num_110 Where or where is....
def num_110():
    """Where or where is... a demonstration of finding Arc* products
    :
    :winreg.QueryValueEx options
    :  ProductName, PythonCondaEnv, PythonCondaRoot
    :- Computer\HKEY_CURRENT_USER\Software\ESRI\Desktop10.5\ArcToolbox\Settings
    :   contains ScriptDebugger, ScriptEditor
    :- Computer\HKEY_CURRENT_USER\Software\ESRI\Desktop10.5\ArcMap
    :- WOW6432Node info contains ESRI
    :- Computer\HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\ESRI\ArcGIS
    """
    import arcpy
    import inspect
    import os
    import sys
    import winreg

    a0 = inspect.getabsfile(arcpy)  # __init__ location for arcpy
    a1 = sys.prefix
    a2 = os.path.abspath(os.path.join(a1, '..', '..', '..'))  # ArcGIS Pro pth
    where_Pro = r"SOFTWARE\ESRI\ArcGISPro"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, where_Pro) as key:
        a3 = winreg.QueryValueEx(key, "InstallDir")[0]
    where_Map = r"SOFTWARE\WOW6432Node\ESRI\ArcGIS"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, where_Map) as key:
        a4 = winreg.QueryValueEx(key, "InstallDir")[0]
    frmt = """
    :ArcPy path ....... {}
    :sys.prefix ....... {}
    :ArcGIS Pro path .. {}
    :Pro winreg path .. {}
    :Map winreg path .. {}
    """
    print(dedent(frmt).format(a0, a1, a2, a3, a4))


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    num_101()  # sum product, einsum by axis
#    num_102()  # Structured to ndarray demo
#    num_103()  # sample plot
#    num_104()  # fancy slicing
#    num_105()  # distance calculation, with large arrays
#    num_106()  # reset counter demo
#    num_107()  # 2D argmax use
#    num_108()  # voxel code... not ready yet
#    a, r, g, b, k, gray = num_109()  # raster band math demo
#    num_110()  # Where or where is....
