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
:    num_111()  # create distance matrix as feature class
:    num_112()  # os.path information
:    num_113()  # sequential counts for attributes
:    num_114()  # heat map by sampling and bucketing
:    num_115()  #
:    num_116()  # form array patterns
:    num_117()  # Using a searchcursor in the field calculator
:    num_118()  # using random.mrand.RandomState
:    num_119()  # Equation of a plane through 3 points
:    num_120()  #
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
import arraytools as art
# import datetime
# import arcpy

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.1f}'.format}
np.set_printoptions(edgeitems=3, linewidth=80, precision=2,
                    suppress=True, threshold=20, formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]  # print this should you need to locate the script

# ---- functions ----


# ----------------------------------------------------------------------
# num_111 create distance matrix as feature class
def num_111():
    """Another distance example, but creating a FC as output
    :  uncomment the line that creates the featureclass
    """
    # import arcpy  # needed if producing a featureclass
    a = np.random.randint(0, 10, size=(5, 2))
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    c = np.einsum('ijk,ijk->ij', a - b, a - b)
    c = np.sqrt(c).squeeze()
    dt0 = [('X', '<f8'), ('Y', '<f8')]
    dt1 = [('p{}'.format(i), '<f8') for i in range(len(a))]
    dt = dt0 + dt1
    d = np.c_[a, c]
    e = np.copy(d)
    e.dtype = dt
    # ---- uncomment below to produce the file ----
    # arcpy.da.NumPyArrayToFeatureClass(out, r'c:\temp\test2.shp', ['X', 'Y'])
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :input arrays...
    :a...
    {}\n
    :b...
    {}\n
    :c... distance matrix...
    {}\n
    :d... full array with uniform dtype
    {!r:} ...\n
    :e... array with specified dtype
    {!r:}\n
    :---------------------------------------------------------------------:
    """
    args = [num_111.__doc__, a, b, c, d, e]
    print(dedent(frmt).format(*args))
    return a, b, c, d, e


# ----------------------------------------------------------------------
# num_112 os.path information
def num_112():
    """os.path examples to find arcgis pro bin
    :Coool os.path and sys.prefix stuff....
    :In order... syss.float_info, sys.prefix with its devolution
    :
    """
    pt1 = os.path.abspath(os.path.join(sys.prefix, "..", ".."))
    pt2 = os.path.abspath(os.path.join(sys.prefix, "..", "..", ".."))
    pt3 = os.path.abspath(os.path.join(sys.prefix, "..", "..", "..", ".."))
    args = [["num_112 docs", num_112.__doc__], ["sys.prefix", sys.prefix],
            ["sys.float_info", sys.float_info], ["sys.int_info", sys.int_info],
            ["sys.path", sys.path], ["sys.platform", sys.platform],
            ["sys.ps1", sys.ps1], ["sys.ps2", sys.ps2], ["sys.ps3", sys.ps3],
            ["sys.version", sys.version], ["path step1", pt1],
            ["path step2", pt2], ["path step 3", pt3]]
    print("\n-----".join(["-----\n{}\n{}".format(i, j) for i, j in args]))

# ----------------------------------------------------------------------
# num_113 sequential counts for attributes
f0 = 0
n0 = 0
def num_113():
    """
    sequentially number an attributes base on its occurrance.
    """
    def hdr(a):
        global f0
        global n0
        frst = a[0]
        if frst == 'F':
            rtrn = "F_{}".format(f0)
            f0 += 1
        else:
            rtrn = "N_{}".format(n0)
            n0 += 1
        return rtrn
    # ----
    a = ['F', 'F', 'N', 'N', 'F', 'N', 'F', 'N', 'F', 'F']

    for i in a:
        print(hdr(i))
#        frmt = """
#        :---------------------------------------------------------------------:
#        {}
#        :---------------------------------------------------------------------:
#        """
#        args = [num_113.__doc__]
#        print(dedent(frmt).format(*args))

# ----------------------------------------------------------------------
# num_114 heat map by sampling and bucketing
def num_114():
    """
    :https://stackoverflow.com/questions/45777934/
    :       creating-a-heatmap-by-sampling-and-bucketing-from-a-3d-array
    """
    def app1(x,y,z):
        """Make so x,y,z data with masks
        """
        X = np.arange(min(x), max(x), 0.1)
        Y = np.arange(min(y), max(y), 0.1)
        x_mask = ((x >= X[:-1, None]) & (x < X[1:, None]))
        y_mask = ((y >= Y[:-1, None]) & (y < Y[1:, None]))
        z_g_out = np.dot(y_mask*z[None].astype(np.float32), x_mask.T)
        # If needed to fill invalid places with NaNs
        z_g_out[y_mask.dot(x_mask.T.astype(np.float32)) == 0] = np.nan
        return z_g_out

    x = np.array([1, 1.12, 1.109, 2.1, 3, 4.104, 3.1])
    y = np.array([-9, -0.1, -9.2, -8.7, -5, -4, -8.75])
    z = np.array([10, 4, 1, 4, 5, 0, 1])

    z_g_out = app1(x, y, z)
    frmt = """
    :---------------------------------------------------------------------:
    {}
    {}
    :---------------------------------------------------------------------:
    """
    args = [num_114.__doc__, z_g_out]
    print(dedent(frmt).format(*args))
    return z_g_out


# ----------------------------------------------------------------------
# num_115 angle between 2, 3d points vectors

def vector_angles(u, v, in_degrees=True):
    """Angle between 2 vectors
    :Point lists, a and b must be at least 2D and the vectors a and b
    :must be unit vectors
    :------
    """
    dot = np.einsum('ijk, ijk -> ij', [u, u, v], [v, u, v])
    ang = np.arccos(dot[0, :] / (np.sqrt(dot[1, :])*np.sqrt(dot[2, :])))
    if in_degrees:
        ang = np.rad2deg(ang).squeeze()
    return ang

# Realize that your arrays `x` and `y` are already normalized, meaning you can
# optimize method1 even more
def vect_angle(u, v, in_degrees=True):
    """Same as vector_angles but for two vectors
    : unit vectors required
    """
    costheta = np.einsum('ij, kj-> ik', u, v)  # Directly gives costheta, since
    ang = np.arccos(costheta)                # ||x|| = ||y|| = 1
    if in_degrees:
        ang = np.rad2deg(costheta)
    return ang


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if np.all([i == 0.0 for i in vector]):
        return vector
    else:
        return vector / np.linalg.norm(vector)


def angle_between(v0, v1, in_degrees=True):
    """ Returns the angle between vectors 'v0' and 'v1 in radians'
    """
    u = unit_vector(v0)
    v = unit_vector(v1)
    ang = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    if in_degrees:
        ang = np.rad2deg(ang)
    return ang


def num_115():
    """Angle between 2 vectors using 3d points vectors
    :
    :Notes:
    :-----
    : np.angle([1.0, 1.0j, 1+1j])               # in radians
    :   array([ 0.000000,  1.570796,  0.785398])
    : np.angle([1.0, 1.0j, 1+1j], deg=True)     # in degrees
    :   array([ 0.000000,  90.000000,  45.000000])
    :
    : angle_between((1, 0, 0), (0, 1, 0))  =>  1.570796326794896
    : angle_between((1, 0, 0), (1, 0, 0))  =>  0.0
    : angle_between((1, 0, 0), (-1, 0, 0)) =>  3.141592653589793
    :
    :Reference:
    :---------
    :  https://stackoverflow.com/questions/2827393/
    :           angles-between-two-n-dimensional-vectors-in-python
    :  https://stackoverflow.com/questions/34738076/
    :          compute-matrix-of-pairwise-angles-between-two-arrays-of-points
    :          ?noredirect=1&lq=1
    :  https://brilliant.org/wiki/3d-coordinate-geometry-equation-of-a-plane/
    :         see the plane passing through 3 points example... used below
    :  https://en.wikipedia.org/wiki/Plane_(geometry)
    : u = np.atleast_2d(u)
    : v = np.atleast_2d(v)
    :np.rad2deg(method3(u, v)) = ([[ 60.]])
    :
    :Notes:
    :-----  To produce the vectors, take an array of 3 points, like a0, a3
    :  below.  subtract the first point from the other 2, which gives
    :  vector u, v

    : P = (1, 1, 1), Q = (1, 2, 0), R = (-1, 2, 1).
    : ----------------------------------------------------------------:
    """
    # points (X, Y, Z)
    a0 = np.array([[ 0.0, 0.0,  0.0], [ 4.0, 3.0,  5.0], [4.0, 0.0, 0.0]])
    a1 = np.array([[ 0.0, 0.0,  0.0], [ 4.0, 3.0,  5.0], [4.0, 0.0, 5.0]])
    a2 = np.array([[ 0.0, 0.0,  0.0], [ 3.0, 4.0,  5.0], [0.0, 4.0, 0.0]])
    a3 = np.array([[ 0.0, 0.0,  0.0], [ 3.0, 4.0,  5.0], [0.0, 0.0, 5.0]])
    a4 = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 0.0], [-1.0, 2.0, 1.0]])
    aa = np.asarray([a0, a1, a2, a3])
    a = a1
    #a = np.array([[ 1.0, 0.0,  2.0], [ 2.0, 1.0,  1.0], [-1.0, 2.0,  1.0]])
    p0, p1, p2 = a
    u0, v0 = a[1:] - a[0] # produce the vectors from p0, p1, p2 = a
    u = unit_vector(u0)  #(p1 - p0) # v0
    v = unit_vector(v0)  #(p2 - p0)  # v1
    w = unit_vector(p2 - p1)
    p0_p1 = np.rad2deg(np.arctan2(u[1], u[0]))
    p0_p2 = np.rad2deg(np.arctan2(v[1], v[0]))
    p1_p2 = np.rad2deg(np.arctan2(w[1], w[0]))
    a_0 = angle_between(u, v)  # np.rad2deg(angle_between(a_av, b_av))
    a_1 = vector_angles(np.atleast_2d(u), np.atleast_2d(v))

    frmt = """
    {}
    :Points p0, p1, p2...
    {}, {}, {}
    :Unit Vectors u, v, w
    {}, {}, {}
    :Angle p0-p1, p0-p2, p1-p2
    {}, {}, {}
    :Angle between p1 p0 p2
    :{}, {}
    :-------------------------------------------------------------------
    """
    args = ['see num_115.__doc__',
            p0, p1, p2, u, v, w, p0_p1, p0_p2, p1_p2, a_0, a_1]
    print(dedent(frmt).format(*args))
    return p0, p1, p2, u, v, w, p0_p1, p0_p2, p1_p2, a_0, a_1, aa


# ----------------------------------------------------------------------
# num_116 form array patterns
def num_116():
    """A simple demo of forming array patterns
    :  using array broadcasting and multiplication
    """
    #
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :Input 1d array ... {}
    : a*a[:,np.newaxis]  results in...
    {}
    : (a+1)*a[:,np.newaxis] yields ...
    {}
    : (a-1)*a[:,np.newaxis]
    {}
    :---------------------------------------------------------------------:
    """
    a = np.array([1,2,3])
    b = a*a[:,np.newaxis]
    c = (a+1)*a[:,np.newaxis]
    d = (a-1)*a[:,np.newaxis]
    inby = [art.in_by(i, prefix='. . ') for i in [b, c, d]]
    args = [num_116.__doc__, a] + inby
    print(dedent(frmt).format(*args))
    return a

# ----------------------------------------------------------------------
# num_117 Using a searchcursor in the field calculator
def num_117():
    """Using a searchcursor in the field calculator"""
    #
    lst =[]
    in_tbl = r'C:\GIS\Tools_scripts\Table_tools\Table_tools.gdb\f1'
    fld_lst = ['sum_Pnts']
    # a = arcpy.da.SearchCursor(in_tbl, fld_lst)._as_narray()
    #lst = [i[0] for i in a]
    def fld_calc(in_fld):
        import arcpy
        global lst
        if len(lst) == 0:
            with arcpy.da.SearchCursor(in_tbl, fld_lst) as cursor:
                for row in cursor:
                    lst.append(row[0])
        del cursor, row
        # Now do the work
        m = min(lst)
        ret = in_fld - m
        return ret

    in_tbl = r'C:\GIS\Tools_scripts\Table_tools\Table_tools.gdb\f1'
    fld_lst = ['sum_Pnts']
    a = arcpy.da.SearchCursor(in_tbl, fld_lst)._as_narray()
    lst = [i[0] for i in a]
    m = min(lst)  # This is function line for the input *******
    def flc_cals2(in_fld):
        return in_fld - m

    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """

# ----------------------------------------------------------------------
# num_118 using random.mrand.RandomState
def num_118():
    """random.mrand.RandomState produces repeatable random numbers
    :
    """
    import arraytools as art
    from scipy.spatial.distance import cdist
    N = 10
    a = np.random.mtrand.RandomState(1).randint(0, 10, size=(N,2))
    b = np.random.mtrand.RandomState(2).randint(0, 10, size=(1,2))
    c = art.e_dist(a, b)  # e_dist calculation to compare to scipy
    d = cdist(a, b)
    d.shape, d.size

    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """

# ----------------------------------------------------------------------
# num_119 Equation of a plane through 3 points
def num_119():
    """Equation of a plane through 3 points in 3D
    :
    :Reference:
    :---------
    :https://sites.math.washington.edu/~king/coursedir/m445w04/notes/
    :        vector/normals-planes.html#cross
    : a0 point set used in the example
    : equation of that line is aX +bY + cZ = d => 1X + 2Y + 2Z = 5
    : for any of the points in a0
    """
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        if np.all([i == 0.0 for i in vector]):
            return vector
        else:
            return vector / np.linalg.norm(vector)

    def _cross_3pnts(a):
        """Requires 3 points on a plane:
        """
        p0, p1, p2 = a
        u, v = a[1:] - a[0]  # p1 - p0, p2 - p0
        #u = unit_vector(u)
        #v = unit_vector(v)
        eq= np.cross(u, v)  # Cross product times one of the points
        d = sum(eq * p0)
        return eq, d

    a0 = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 0.0], [-1.0, 2.0, 1.0]])
    #a0 = np.array([[1.0, 1.0, 0.0], [1.0, 0, 1.0],  [0, 1, 2]])
    eq, d = _cross_3pnts(a0)
    eq /= d
    d /= d
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :Input points .... 3 points on a plane
    {}\n
    :Equation of the line...\n
        {:0.3f}X + {:0.3f}Y + {:0.3f}Z = {:0.3f}\n
    :---------------------------------------------------------------------:
    """
    ar = art.in_by(a0, '  columns: X, Y, Z ', False, prefix='....')
    args = [num_119.__doc__, ar, eq[0], eq[1], eq[2], d]
    print(dedent(frmt).format(*args))
    return a0, eq, d
# ----------------------------------------------------------------------
# num_120
def num_120():
    """    """
    #
    frmt = """
    :---------------------------------------------------------------------:
    {}
    :---------------------------------------------------------------------:
    """
# ----------------------------------------------------------------------
# __main__ .... code section
if __name__ == "__main__":
    """Optionally...
    : - print the script source name.
    : - run the _demo
    """
#    print("Script... {}".format(script))
#    a, b, c, d, e = num_111()  # create distance matrix as feature class
#    num_112()  # os.path information
#    num_113()  # sequential counts for attributes
#    a = num_114()  # heat map by sampling and bucketing
#    p0, p1, p2, u, v, w, p0_p1, p0_p2, p1_p2, a_0, a_1, aa = num_115()  #
#    a = num_116()  # form array patterns
#    num_117()  # Using a searchcursor in the field calculator
#    num_118()  # using random.mrand.RandomState
    a0, eq, d = num_119()  # Equation of a plane through 3 points
#    num_120()  #

# ---------------------------------------------------------------------
