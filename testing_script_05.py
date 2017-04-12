# -*- coding: UTF-8 -*-
"""
:Script:   testing_script_05.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2017-01
:
:Purpose:
:
:Functions list .........
:...... np functions .....
:    num_71()  # slicing and remainders
:    num_72()  # logical_or and range checking
:    num_73()  # slicing in structured arrays by condition
:    num_74()  # Load *.npy files
:    num_75()  # make random3darrays wth a predetermined shape
:    num_76()  # produce row/column indices from triu
:    num_77()  # 2.7 indent function
:    num_78()  # logical_or, condition checking
:    num_79()  # bad floating point comparisons
:    num_80()  # using r_ and c_ for rapid indexing and array construction
:    num_81()  # sorting arrays by column, revisited
:    num_82()  # line indentation options
:    num_83()  # PIL testing
:    num_84()  # mandelbrot demo
:    num_85()  # circle search
:    num_86()  # standardize by rows or columns
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

ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100,
                    formatter=ft)
np.ma.masked_print_option.set_display('-')

script = sys.argv[0]

# ---- functions ----


# ----------------------------------------------------------------------
# num_71
def num_71():
    """(num_71)... slicing and remainders
    :
    :References:
    :----------
    :  http://stackoverflow.com/questions/41024037/get-whats-remaining
    :       -after-a-slice-using-numpy
    :
    """
    frmt = """
    :------------------------------------------------------------------
    :{}
    :Input array
    {}
    with slice r = {}  yields {}
    :Alternatives
    :  a[~np.in1d(np.arange(a.size), r)]  # Without r
    :  a[np.setdiff1d(np.arange(a.size), r)]
    :  np.concatenate((a[:r[0]], a[r[-1]+1:])) **** nice
    :  u, v, w = a[:r[0]], a[r], a[r[-1]+1:]
    :  u = array([10, 11, 12])
    :  v = array([13, 14, 15])
    :  w = array([16, 17, 18, 19, 20])
    :  {}
    :
    :------------------------------------------------------------------
    """
    a = np.arange(10, 21)
    r = [3, 4, 5]
    b = a[r]
    c = a[~np.in1d(np.arange(a.size), r)]  # Without r
    args = [num_71.__doc__, a, r, b, c]
    print(dedent(frmt).format(*args))
    return None


# ----------------------------------------------------------------------
# num_72
def num_72():
    """(num_72)... logical_or and range checking
    :logical_or demo
    : Assume these values represent the np.std(a) values for +/-1.25 std
    : about a mean
    """
    frmt = """
    :------------------------------------------------------------------
    :{}\n     :standard devs{}
    :Input Array\n
    {}\n\n
    :   np.where((a < stds[0]) | (a > stds[1]), a, -1) ... or ....
    :   np.where(np.logical_or(a < stds[0], a > stds[1]), a, -1)\n
    {}
    {}
    """
    a = np.arange(100).reshape(10, 10)
    # xbar = np.mean(a)
    stds = [20, 80]
    m = np.where((a < stds[0]) | (a > stds[1]), a, -1)
    m2 = np.where(np.logical_or(a < stds[0], a > stds[1]), a, -1)
    args = [num_72.__doc__, stds, a, m, m2]
    print(dedent(frmt).format(*args))
    return None


# ----------------------------------------------------------------------
# num_73
def num_73():
    """(num_73)... slicing in structured arrays by condition
    :
    """
    frmt = """
    :------------------------------------------------------------------
    :{}
    :Input array ...
    {}\n
    : a[a['x'] > 250]
    {!r:}\n
    : query parts  a['x'] > 250 ... a['y'] > 30
    : becomes     (a['x'] > 250) & (a['y'] > 30) ...with & or |
    : then   q1 = a[(a['x'] > 250) & (a['y'] > 30)] ... slice from 'a'
    : yields....
    {!r:}
    :------------------------------------------------------------------
    """
    a = np.array([(100., 50.), (200., 60.), (300., 70.),
                  (400., 40.), (500., 25.)],
                 dtype=[('x', '<f8'), ('y', '<f8')])
    q0 = a[a['x'] > 250]
    q1 = a[(a['x'] > 250) & (a['y'] > 30)]
    args = [num_73.__doc__, a, q0, q1]
    print(dedent(frmt).format(*args))
    # return a


# ----------------------------------------------------------------------
# num_74
def num_74():
    """(num_74)... Load *.npy files
    :Requires:
    :--------  The file name, the source folder, which can be "" if
    :  included n the file name (use "/" as a path separator)
    """
    frmt = """
    :------------------------------------------------------------------
    :{}
    :Source *.npy
    :{}
    """
    sep = os.path.sep
    src_npy = "z_8x10.npy"
    src_folder = "data"
    script_path = os.path.split(sys.argv[0])[0]  # for 2.7 compat.
    data_file = sep.join([script_path, src_folder, src_npy])
    #
    a = np.load(data_file)
    args = [num_74.__doc__, data_file]
    print(dedent(frmt).format(*args))
    return a


# ----------------------------------------------------------------------
# num_75
def num_75():
    """(num_75)... synthetic 3d arrays then cumulative sum
    :
    """
    frmt = """
    :------------------------------------------------------------------
    :{}
    {}
    {}
    """
    num = 5
    z = []
    r, c = (3, 4)
    for i in range(num):
        z.append(np.arange(r*c).reshape(r, c))
    z = np.array(z)
    zz = np.cumsum(z, axis=0)
    args = [num_75.__doc__, z, zz]
    print(dedent(frmt).format(*args))
    # return z, zz


# ----------------------------------------------------------------------
# num_76
def num_76():
    """(num_76)... produce row/column indices from triu
    :
    """
    frmt = """
    :------------------------------------------------------------------
    :{}
    : rows - {}
    : cols - {}
    : output... r + c first...
    {}
    :output... r - c  then fill in the remainder
    :------------------------------------------------------------------
    """
    N = 5
    r, c = np.triu_indices(N)  # r => y, c => x
    out = np.zeros((1, N, N), dtype=int)
    vals = r - c  # try a function
    out[:, r, c] = vals
    out[:, c, r] = vals
    args = [num_76.__doc__, r, c, out]
    print(dedent(frmt).format(*args))
    # return r,c, out


# ----------------------------------------------------------------------
# num_77
def num_77():
    """(num_77)... 2.7 indent function
    :See the documentaton in the enclosed function
    :
    """
    frmt = """
    :------------------------------------------------------------------
    :{}
    """

    def in_by(obj, hdr="", nums=False, prefix="  "):
        """
        textwrap.indent variant for python 2.7 or a substitute for
        :any version of python.  The function stands for 'indent by'
        :Requires:
        :--------
        :  obj - obj to indent, List, tuple, ndarray converted to strings
        :    first. You can use repr representation before using if needed.
        :  hdr - optional header
        :  nums - boolean, add line numbers
        :  prefix - text to use for indent ie '  ' for 2 spaces or '....'
        :Reference:
        :---------
        :  https://docs.python.org/3.7/library/textwrap.html for python >3.3
        :Notes:
        :-----
        :  Header and line numbers options added.
        """
        if hdr != "":
            hdr = "{}\n".format(hdr)
        if isinstance(obj, (list, tuple, np.ndarray)):
            obj = str(obj)

        def prefixed_lines():
            c = 0
            for line in obj.splitlines(True):
                if nums:
                    frmt = "{:>02}{}{}".format(c, prefix, line)
                else:
                    frmt = "{}{}".format(prefix, line)
                yield frmt
                c += 1
        out = hdr + "".join(prefixed_lines())
        return out
    frmt = """
    {}\n    {}\n    :A line and an array...\n    :... str(a)\n    {}
    \n    :... repr(a)\n    {}\n    :The end
    """
    a = np.arange(2*3*4).reshape(2, 3, 4)
    b = a.view(type=np.recarray)
    args = [num_77.__doc__, dedent(in_by.__doc__),
            in_by(a), in_by(repr(a))]
    print(dedent(frmt).format(*args))
    return a


# ----------------------------------------------------------------------
# num_78
def num_78():
    """(num_78)... logical_or, condition checking
    : split into where with condition, or using logical_or directly
    : The options chosen and syntax depends on whether you want to
    : retain the array shape or just get a result.  Obviously,
    : simplifying the result to slicing is visually shorter.
    """
    frmt = """
    :------------------------------------------------------------------
    :{}
    :Input array
    {}
    :Where result...
    {}\n
    :logical_or result...
    {}\n
    :nansum differences for...
    : b0-b1 => {}
    : b1-b2 => {}
    :NaNs out of N=> {}/{}
    :------------------------------------------------------------------
    """
    a = np.random.rand(5, 5)
    msk = (a < 0.5) | (a > 0.8)
    b0 = np.where((a < 0.5) | (a > 0.8), a, np.NaN)
    b1 = a[np.logical_or(a < 0.5, a > 0.8)]
    b2 = np.nansum(a[msk])
    a[msk] = np.NaN
    nan_s = np.isnan(a[msk]).sum()
    good = np.diff([np.nansum(b0), np.nansum(b1), b2])
    args = [num_78.__doc__, a, b0, b1, good[0], good[1], nan_s, a.size]
    print(dedent(frmt).format(*args))
    # return a, b0, b1, b2, msk


# ----------------------------------------------------------------------
# num_79
def num_79():
    """(num_79)... bad floating point comparisons
    :Reference:
    :  http://stackoverflow.com/questions/41233152/
    :         scaling-image-held-in-numpy-array-cannot-be-undone
    :
    """
    frmt = """
    {0}
    :array 'a' np.arange({1}*{2}*{3}).reshape({1},{2},{3})
    :array 'b' => a/{4}.
    :array 'c' => b*{4}
    :array sizes (a, b, c) N = {5}
    :array 'd' => c.astype(int)
    :comparison sums and inequalities
    :np.sum(a==c) => {6}   a[a!=c] => {7}
    :np.sum(a==d) => {8}   a[a!=d] => {9}
    :
    :------------------------------------------------------------------
    """
    shp = (5, 5, 3)
    N = np.prod(shp)
    prod = np.array((shp[0]*shp[1]), dtype=np.int64)
    a = np.arange(N, dtype=np.int64).reshape(shp)
    b = a/prod
    c = b*prod
    d = c.astype(np.int64)
    e = np.round(c)
    args = [num_79.__doc__, shp[0], shp[1], shp[2], prod, N,
            np.sum(a == c), a[a != c], np.sum(a == d), a[a != d]]
    print(dedent(frmt).format(*args))
    # return a, b, c, d, e


# ----------------------------------------------------------------------
# num_80
def num_80():
    """(num_80)... r_ and c_ for rapid array construction
    : Note:  the syntax for both is np.r_[] and np.c_[]
    : x = np.arange(10)
    : y = np.arange(10,0,-1)
    : construct x,y data
    :  xy_r = np.r_[x, y]  makes one big row out of the x,y values
    : these work...
    : (1) xy = np.array(list(zip(x, y)))
    : (2) xy_c = np.c_[x, y]
    :   xy_c
    :    array([[ 0, 10],
    :           [ 1,  9],
    :           [ 2,  8],
    :           [ 3,  7],
    :           [ 4,  6],
    :           [ 5,  5],
    :           [ 6,  4],
    :           [ 7,  3],
    :           [ 8,  2],
    :           [ 9,  1]])
    : - rapid math... add 2, to each of the coordinates
    :   np.c_[xy_c, xy_c+[2,2]]
    :    array([[ 0, 10,  2, 12],
    :           [ 1,  9,  3, 11],
    :           [ 2,  8,  4, 10],
    :           [ 3,  7,  5,  9],
    :           [ 4,  6,  6,  8],
    :           [ 5,  5,  7,  7],
    :           [ 6,  4,  8,  6],
    :           [ 7,  3,  9,  5],
    :           [ 8,  2, 10,  4],
    :           [ 9,  1, 11,  3]])
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Arrays for ndim=1
    : a => {}  b => {}
    :r_[a, b] => {}
    :c_[a, b]
    {}\n
    :Arrays for ndim=2
    : array c
    {}
    : array d
    {}\n
    :r_[c, d]
    {}\n
    :c_[c, d]
    {}
    :------------------------------------------------------------------
    """
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    r0 = np.r_[a, b]  # notice the square brackets
    c0 = np.c_[a, b]  # ditto
    c = np.arange(7, 11).reshape(2, 2)
    d = np.arange(12, 16).reshape(2, 2)
    r1 = np.r_[c, d]
    c1 = np.c_[c, d]
    strs = [indent(str(i), "    ") for i in [c0, c, d, r1, c1]]
    args = [num_80.__doc__, a, b, r0]
    args.extend(strs)
    print(dedent(frmt).format(*args))
    # return a, b, c, d


# ----------------------------------------------------------------------
# num_81
def num_81():
    """(num_81)... sorting arrays by column, revisited
    :
    :Reference:
    :---------
    :  http://stackoverflow.com/questions/2828059/sorting-arrays-in-
    :       numpy-by-column
    :Notes:
    :-----
    :From ndarray.sort and np.argsort
    :  ndarray.sort(axis=-1, kind='quicksort', order=None)
    :  np.argsort(axis=-1, kind='quicksort', order=None)
    :  - order : str or list of str, optional
    :When a is an array with fields defined, this argument specifies
    :  which fields to compare first, second, etc. A single field can be
    :  specified as a string, and not all fields need be specified, but
    :  unspecified fields will still be used, in the order in which they
    :  come up in the dtype, to break ties.
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Array 'a'  ... sort by the 2nd column
    {}
    : method 1 np.sort(a.view('i8,i8,i8'), order=['f1'], axis=0).view(np.int)
    {}
    : method 2 a[a[:,1].argsort()]
    {}
    : method 3 a[np.lexsort(a[:,::-1].T)]
    {}
    :------------------------------------------------------------------
    """
    a = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [0, 0, 1]])
    a0 = np.sort(a.view('i8,i8,i8'), order=['f1'], axis=0).view(np.int)
    a1 = a[a[:, 1].argsort()]  # column 2
    a2 = a[np.lexsort(a[:, ::-1].T)]
    args = [num_81.__doc__, a, a0, a1, a2]
    print(dedent(frmt).format(*args))
    return a, a0, a1


# ----------------------------------------------------------------------
# num_82
def num_82():
    """(num_82)... line indentation options
    :Demonstrating 'redent' and 'art.in_by'.  The latter has variants
    :  that proide for, a header, line numbers and a space or alternate
    :  leaders.
    """

    def redent(lines, spaces=4):
        """Strip and reindent by num_spaces, a sequence of lines
        :  lines - text or what can be made text
        :  Use str() or repr() on the inputs if you want control on form
        """
        lines = str(lines).splitlines()
        sp = [len(ln) - len(ln.lstrip()) for ln in lines]
        spn = " "*spaces
        out = list(zip(lines, sp))
        ret = "\n".join(["{0}{1!s:>{2}}".format(spn, *ln) for ln in out])
        return ret
    #
    frmt = """
    :------------------------------------------------------------------
    {}
    :Documentation indented by 4 using redent(lines, spaces=4)\n
    : .... redent(a, 4) ....
    {}\n
    : .... art.in_by(a, hdr="Option 1", nums=False, prefix = "    ")
    {}\n
    : .... art.in_by(a, hdr="Option 2", nums=False, prefix = "....")
    {}\n
    : .... art.in_by(a, hdr="Option 3", nums=True, prefix = "  ")
    {}
    :------------------------------------------------------------------
    """
    import arraytools as art
    a = np.arange(30).reshape(5, 6)
    # xtra =[redent(i, 4) for i in [a, a.T]] # to do a bunch
    args = [num_82.__doc__, redent(a, 4),
            art.in_by(a, hdr="Option 1", nums=False, prefix="    "),
            art.in_by(a, hdr="Option 2", nums=False, prefix="...."),
            art.in_by(a, hdr="Option 3", nums=True, prefix="  ")]
    # args.extend(xtra)
    print(dedent(frmt).format(*args))


# ----------------------------------------------------------------------
# num_83
def num_83():
    """(num_83)... PIL testing
    :
    """
    from PIL import Image
    frmt = """
    :------------------------------------------------------------------
    {}
    :Array a
    {}
    : a0....
    {}
    : a1....
    {}
    :------------------------------------------------------------------
    """

    def array2PIL(arr, size):
        mode = 'RGBA'
        arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
        if len(arr[0]) == 3:
            arr = np.c_[arr, 255*np.ones((len(arr), 1), np.uint8)]
        return Image.frombuffer(mode, size, arr.tostring(), 'raw', mode, 0, 1)
    a = np.arange(0, 256, 4).reshape(8, 8)
    a = art.scale(a, x=32, y=32)
    a0 = np.rot90(a, 2)
    a1 = np.abs(((a0-a)/2)).astype(int)
    arr = np.dstack((a, a0, a1))
    a2 = np.rollaxis(arr, 2, 0)
    i = Image.fromarray(a.astype(np.uint8))
    #i.show()
    # img2 = array2PIL(arr, a2.size)
    # img2.save('out.jpg')
    # im = Image.fromarray(cm.gist_earth(a, bytes=True))
    args = [num_83.__doc__, a, a0, a1]
    # print(dedent(frmt).format(*args))
    return a, i


# ----------------------------------------------------------------------
# num_84
def num_84():
    """(num_84)... mandelbrot demo
    : Python with PIL and python with numpy examples
    :  http://codegolf.stackexchange.com/questions/23423/
    :       mandelbrot-image-in-every-language/23558#23558
    """
    import numpy as np
    import pylab
    """
    i = 49 #99
    x, y = np.mgrid[-2:2:999j,-2:2:999j]
    c = r = x*1j + y
    x -= x
    while i:
        x[(abs(r) >2 ) & (x==0)] = i
        r = r*r + c
        i -= 1
    pylab.imshow(x)
    pylab.show()
    """
    from PIL import Image
    d = 600
    i = Image.new('RGB', (d, d))
    for x in range(d*d):
        z = o = x/9e4 - 2 - x % d/150.j - 2j
        c = 99
        while(abs(z) < 2)*c:
            z = z*z + o
            c -= 1
        i.putpixel((int(x/d), x % d), 5**8*c)
    i.show()
#    frmt = """
#    :------------------------------------------------------------------
#    {}
#    :------------------------------------------------------------------
#    """
#    return x, y


# ----------------------------------------------------------------------
# num_85
def num_85():
    """circle search"""

    def circle(radius=15, inner=0, border=0):
        rad = radius+border
        xx, yy = np.mgrid[:rad, :rad]
        cent = int(rad/2.0)
        circ = (xx - cent) ** 2 + (yy - cent) ** 2
        donut = np.logical_and(circ > (25), circ < 40)
        # or more simply donut =
        #                (circle < (6400 + 60)) & (circle > (6400 - 60))
        return circ, donut
    return circle()

if __name__ == "__main__":
    """Main section...   """
#    print("Script... {}".format(script))
#
#    num_71()  # slicing and remainders
#    num_72()  # logical_or and range checking
#    num_73()  # slicing in structured arrays by condition
#    num_74()  # Load *.npy files
#    num_75()  # make random3darrays wth a predetermined shape
#    num_76()  # produce row/column indices from triu
#    num_77()  # 2.7 indent function
#    num_78()  # logical_or, condition checking
#    num_79()  # bad floating point comparisons
#    num_80()  # using r_ and c_ for rapid indexing and array construction
#    a, a0, a1 = num_81()  # sorting arrays by column, revisited
#    num_82()  # line indentation options
#    num_83()  # PIL testing
#    num_84()  # mandelbrot demo
#    num_85()  # circle search
#    a, b, c, d, e = num_86()  # standardize by rows or columns
frmt = """
    :------------------------------------------------------------------
    :{}
    :Input array
    {!s:}
"""


def indent_size(lines):
    """Find the minimum number of spaces a line in lines is indented by.
    :  The minimum indent size is returned, then returned to the calling
    :  script dedent_by.
    :  sp_max - maximum spaces found
    """
    sp_max = 80
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            sze = len(line) - len(stripped)
            ind_min = min(sp_max, sze)
    if ind_min == sp_max:
        return 0
    return ind_min, ind_max


def dedent_by(in_lines):
    """Set docstring to minimum indent for all lines, including first
    >>> unindent_string(' two')
    'two'
    >>> unindent_string('  two\\n   three')
    'two\\n three'
    """
    lines = in_lines.expandtabs().splitlines()
    spaces = indent_size(lines)
    if spaces == 0:
        return in_lines
    return "\n".join([line[spaces:] for line in lines])


#a = np.arange(3*4).reshape(3,4)
#s, out = redent(frmt.format("test line", str(a)))



