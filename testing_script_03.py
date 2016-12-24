# -*- coding: UTF-8 -*-
"""
:Script:   testing_script_03.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-10-27
:Purpose:  Demonstration functions for a variety of small examples.
:Functions:  help(<function name>) for help
:---------
:Functions list .........
:...... np functions .....
:    num_40()  np.genfromtext example
:    num_41()  Documenting code using inspect
:    num_42()  list files in folder
:    num_43()  Blocking an array 
:    num_44()  a variant on array_split
:    num_45()  bulk create structured array fields
:    num_46()  Masked array from ill-formed list
:    num_47()  Block with padding and reshaping
:    num_48()  Formatting again, using indent and dedent
:    num_49()  kroneker product and array construction
:    num_50()  fancy indexing ....
:    num_52()  Closeness Manahatten
:    num_53()  formatting output
:    num_54()  Producing uniform distribution data
:    num_55()  combinations and frequency
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
from textwrap import dedent, indent
from io import StringIO, BytesIO
from plot_arr import _f
import arr_tools as art


ft = {'bool': lambda x: repr(x.astype('int32')),
      'float': '{: 0.3f}'.format}
np.set_printoptions(edgeitems=10, linewidth=80, precision=2,
                    suppress=True, threshold=100, 
                    formatter=ft)
np.ma.masked_print_option.set_display('-')
script = sys.argv[0]

# ---- functions ----

# ----- Dummy comment ------------------------------------------------
# dummy used by num_41
def dummy():
    """dummy...
    : Demonstrates retrieving and documenting module and function info.
    :
    """
    def sub():
       """sub in dummy"""
       pass
    return None


#----------------------------------------------------------------------
# num_40
def num_40(in_file):
    """(num_40)...
    : Demonstrates reading data into structured array format using 
    : bytes/strings from a file and a string to represent a file.
    : Using structured and recarrays quick demo
    """
    frmt = """
    :------------------------------------------------------------------
    {}
    :Read from a text file and generate an array
    :  Python 3.5 used...\n
    :Text file output (a)...
    {!s:}\n
    :String output (b)......
    {!s:}\n
    :Both equal?....{}\n
    :Viewed in column format using a.reshape(a.shape[0], 1)
    :{}\n
    :Structured (a) vs recarray (c) data access
    :  a['Age'].min() {} <==> c.Age.min() ==> {}
    :-------------------------------------------------------------------
    """
    from io import BytesIO
    # read from file
    dt = [('Name', 'U10'), ('Age', 'i8'), ('Test1', 'f8'), ('Test2', 'f8')]
    a = np.genfromtxt(in_file, dtype=dt, delimiter=",", autostrip=True)
    # read from text string
    data = "Dan, 62, 8.5, 7.0\n Cali, 5, 9.5, 8.3\n Rocky, 10, 9.2, 8.1\n Obi, 82, 8.0, 8.2"
    s = BytesIO(data.encode())
    b = np.genfromtxt(s, dtype=dt, delimiter=',', autostrip=True)
    c = a.view(np.recarray)
    args = [num_40.__doc__, a, b, np.all(a==b), a.reshape(a.shape[0], 1),
            a['Age'].min(), c.Age.min()]
    print(dedent(frmt).format(*args))
    return a, b

# ----------------------------------------------------------------------
# num_41 
def num_41(func=None):
    """(num_41)...Documenting code using inspect
    :Requires:
    :--------
    :  import inspect  # module
    :Source code for...
    :  module level   => inspect.getsourcelines(sys.modules[__name__])[0]
    :  function level 
    :       as a list => inspect.getsourcelines(num_41)[0]
    :     as a string => inspect.getsource(num_41)
    :  file level => script = sys.argv[0]
    :Returns:  a listing of the source code with line numbers
    :
    :>>> dir(num_41)
      ['__annotations__', '__call__', '__class__', '__closure__', '__code__', 
      '__defaults__', '__delattr__', '__dict__', '__dir__', '__doc__',
      '__eq__', '__format__', '__ge__', '__get__', '__getattribute__',
      '__globals__', '__gt__', '__hash__', '__init__', '__kwdefaults__',
      '__le__', '__lt__', '__module__', '__name__', '__ne__', '__new__',
      '__qualname__', '__reduce__', '__reduce_ex__', '__repr__',
      '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
    :

    :dir(num_41.__code__:)
    :   [ '__class__', '__delattr__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__le__', '__lt__', '__ne__', '__new__', '__reduce__', 
    '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', 
    '__subclasshook__', 'co_argcount', 'co_cellvars', 'co_code', 'co_consts', 
    'co_filename', 'co_firstlineno', 'co_flags', 'co_freevars', 'co_kwonlyargcount', 'co_lnotab', 'co_name', 'co_names', 'co_nlocals', 
    'co_stacksize', 'co_varnames']]
    : num_41.__defaults__  # (None,)
    : num_41.__dict__      # {} 
    : num_41.__getattribute__('__name__')  # 'num_41'
    : num_41.__module__   # '__main__'
    : num_41.__name__     # 'num_41'
    :-------
    :
    """
    def predicates(func):
        """   """
        predicate = [['isfunction', ['__doc__', '__name__', '__code__', '__defaults__', '__globals__', '__kwdefaults__']],
                     ['ismodule',[]], ['isroutine', []],
                      ['ismethod'], []
                     ]
    def demo_def():
        """dummy...
        : Demonstrates retrieving and documenting module and function info.
        :
        """
        def sub():
           """sub in dummy"""
           pass
        return None
    import inspect
    if func is None:
        func = demo_def
    script = sys.argv[0]  # a useful way to get a file's name
    lines, line_num = inspect.getsourcelines(func)
    code = "".join(["{:4d}  {}".format(idx, line)
                    for idx, line in enumerate(lines)])
    defs = [key for key, value in globals().items()
            if callable(value) and value.__module__ == __name__]
    args = [line_num, code,
            inspect.getcomments(func), inspect.isfunction(func),
            inspect.ismethod(func), inspect.getmoduleinfo(script),
            defs
            ]
    members = []
    funcs = []
    if inspect.ismodule(func): #ismodule, isfunction
        m_s = inspect.getmembers(func)
        for m in m_s:
            members.append(m[0])
    if inspect.isfunction(func):
        f_s = inspect.getmembers(func)
        for f in f_s:
            funcs.append(f[0])
    # **** work on this
    mem = [i[0] for i in inspect.getmembers(art)]
    frmt = """
    :----------------------------------------------------------------------
    :Code for a function on line...{}...
    {}
    :Comments preceeding function
    {}
    :function?... {} ... or method? {}
    :Module info...
    {}
    :
    :Module functions...
    {}    
    :----------------------------------------------------------------------
    """
    print(dedent(frmt).format(*args))
    print("function member names\n{}".format(members))
    return None


# ----------------------------------------------------------------------
# num_42 
def num_42():
    """(num_42)...unique while maintaining order from the original ndarray
    :
    :Notes:
    :-----
    : format tips for creating field names....
    : simple  ["f{}".format(i) for i in range(2)]
    :         ['f0', 'f1']
    : padded  ["a{:0>{}}".format(i,3) for i in range(5)]
    :         ['a000', 'a001', 'a002', 'a003', 'a004']
    """
    frmt = """
    :----------------------------------------------------------------------
    :Unique on ndarrays with uniform dtype...
    :  Exploiting structured and recarrays to facilitate tasks.
    :  Also covered sorting and/or keeping order.
    :
    :Input array: type {}  dtype {}
    {!r:}\n
    :dtype construction to produce field names
    {}\n
    :Creating a view into an ndarray in structured or recarray format
    {!r:}\n
    :Unique values and their indices
    : - unique...{}
    : - indices..{}\n
    :Order by sorting...
    {!r:}\n
    :Order using original order...
    {!r:}\n
    :Returning the original array with unique values and same order...
    {!r:}    
    :----------------------------------------------------------------------
    """
    a = np.array([[2, 0], [1, 0], [0, 1], [1, 0], [1, 2], [1, 2]])
    shp = a.shape
    dt_name = a.dtype.name
    flds = ["f{:0>{}}".format(i, 2) for i in range(shp[1])]
    dt = [(fld, dt_name) for fld in flds]
    b = a.view(dtype=dt).squeeze()  # type=np.recarray,
    c, idx = np.unique(b, return_index=True)
    d0 = b[idx]
    d1 = b[np.sort(idx)]
    #print("\n{}".format(num_42.__doc__))
    e = d1.view(dtype=a.dtype).reshape(d1.shape[0], a.shape[1])
    args = [a, type(a).__name__, a.dtype.name, dt, b, c, idx, d0, d1, e]
    print(dedent(frmt).format(*args))
    return a, b, c


# ----------------------------------------------------------------------
# num_43 
def num_43():
    """(num_43)...Blocking an array options
    :Notes:
    :-----
    :An ndarray can be blocked or subdivided into chunks (not moving
    :  windows) in several ways.  This demo shows how the resultant
    :  arrays are configured and their results.
    :
    :References:
    :----------
    : - https://github.com/numpy/numpy/blob/master/numpy/lib/shape_base.py
    :
    :Notes: there is a hierarchy in split with array_split being called
    :  if the divisions are to be unequal...
        sub_arys = []
        sary = _nx.swapaxes(ary, axis, 0)
        for i in range(Nsections):
            st = div_points[i]
            end = div_points[i + 1]
            sub_arys.append(_nx.swapaxes(sary[st:end], axis, 0))
    :  array_split(ary, indices_or_sections, axis=0) allows for unequal
    
    :  split(ary, indices_or_sections, axis=0)
    :  np.vsplit - return split(ary, indices_or_sections, 0)
    :  np.hsplit 
    :    split(ary, indices_or_sections, 0) - len(a.shape) = 1
    :    split(ary, indices_or_sections, 1) - len(a.shape) > 1
    :
    :  np.ndindex(3,3) from <numpy.lib.index_tricks.ndindex
    """
    
    def block(a, r=3, cs=3, row_order=True):
        """Block slice an array using a window of (rs, cs) size
        """
        lenr = a.shape[0]//rs
        lenc = a.shape[1]//cs
        if row_order:
            iter = [(i, j) for (i, j) in np.ndindex(lenr, lenc)]
        else:
            iter = [(j, i) for (i, j) in np.ndindex(lenr, lenc)]
        b = np.array([a[i*rs:(i+1)*rs, j*cs:(j+1)*cs] for (i,j) in iter])
        #b = np.array([a[i*rs:(i+1)*rs, j*cs:(j+1)*cs] 
        #              for (i, j) in np.ndindex(lenr, lenc)])
        return b
    r = 6
    c = 6
    a = np.arange(r*c).reshape(r, c)
    vs = np.array(np.vsplit(a, 2))
    hs = np.array(np.hsplit(a, 2))
    #a.squeeze(axis=(2,3))
    rs = 3
    cs = 4
    #lenr = a.shape[0]//rs
    #lenc = a.shape[1]//cs
    #b = np.array([a[i*rs:(i+1)*rs, j*cs:(j+1)*cs] 
    #              for (i, j) in np.ndindex(lenr, lenc)])
    #b1 = np.array([a[i*rs:(i+1)*rs, j*cs:(j+1)*cs] 
    #              for (j, i) in np.ndindex(lenr, lenc)])
    e = block(a, 3, 4, row_first=False)
    b = block(a, rs, cs, True)
    b1 = block(a, rs, cs, False)
    c = np.array([np.vsplit(i, 2) for i in np.hsplit(a, 2)])
    d = np.array([np.hsplit(i, 2) for i in np.vsplit(a, 2)])
    #c = c.reshape(lenr*lenc, rs, cs) 
    return a, b, b1, c, d, e

# ----------------------------------------------------------------------
# num_44 comment line above def
def num_44():
    """Blocking arrays... via split and also, a reshape
    [(i, j, (s + (i,j)* np.mod(s, (i,j))- np.mod(s, (i,j)))) for i in range(2,5) for j in range(2,5)]
    
    b0, b1 = rows, cols  # Blocksize
    x, y = b.shape       # padded array
    #b.reshape((x//b0,b0,y//b1,b1))
    b.reshape((x//b0,b0,y//b1,b1)).swapaxes(1,2).reshape(-1,b0,b1)
    m,n = a.shape
    out = a.reshape(m//B,B,n//B,B).swapaxes(1,2).reshape(-1,B,B)
    """
    def block_array(a, rows=3, cols=4, col_first=True, nodata=-1):
        """ a variant on array_split
        requires a N*m array
        """
        s = np.array(a.shape)
        w = np.array([rows, cols])
        m = divmod(s, w)
        new_shape = w*m[0] + w*(m[1]!=0)
        ypad, xpad = new_shape - a.shape  
        b = np.pad(a, pad_width=((0, ypad),(0, xpad)), 
                                 mode='constant', 
                                 constant_values=((nodata, nodata),(nodata, nodata)))
        rn, cn = new_shape
        x_s = np.arange(0, cn+cols, cols)[1:] #.tolist()
        y_s = np.arange(0, rn+rows, rows)[1:] #.tolist()
        print("x_s {}\ny_s {}".format(x_s, y_s))
        #c = np.array([i for i in np.hsplit(b, x_s) if len(i) > 0])
        c = np.array([i for i in np.split(b, x_s, axis=1) if len(i) > 0])
        d = np.array([i for i in np.split(c, y_s, axis=1) if len(i) > 0])
        e = d.swapaxes(0, 1)
        ix = np.in1d(e.ravel(), nodata).reshape(e.shape)
        f = np.ma.array(e, mask=ix, fill_value=-1)
        return b, c, d, e, f
    y, x = 9, 11
    a = np.arange(x*y).reshape(y,x)
    b, c, d, e, f = block_array(a)
    print("\n{}".format(num_44.__doc__))    
    for i in [a, b, c, d, e, f]:
        _f(i)
    return a, b, c, d, e, f
        
# ----------------------------------------------------------------------
# num_45
def num_45():
    """(num_45)...bulk create structured array fields
    """
    import numpy as np
    flds = ["f{:0>{}}".format(i,2) for i in range(7)]
    dt = [(fld, 'float32') for fld in flds]
    dt.append(('i01', 'int8'))
    a = np.zeros((10,), dtype=dt)
    b = np.arange(10*8).reshape(10,8)
    c = np.copy(a)
    names = a.dtype.names
    N = len(names)
    for i in range(N):
        c[names[i]] = b[:,i]
    #
    n = ['It', 'is', 'easy']
    dt = [(n[0], '<f8'), (n[1], '<i8'), (n[2], 'U5')]
    d = np.zeros((10,), dtype=dt)
    for i in range(len(n)):
        d[n[i]] = b[:, i]
    print("\n{}".format(num_45.__doc__))
    return a, b, c, d 


# ----------------------------------------------------------------------
# num_46 
def num_46():
    """(num_46)... Masked array from ill-formed list
    :  http://stackoverflow.com/questions/40289943/
    :  converting-a-3d-list-to-a-3d-numpy-array
    :  A =[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], 
    :      [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0], [0], [0]]]
    """
    frmt = """
    :Input list...
    {}\n
    :Masked array data
    {}\n
    :A sample calculations:
    :  a.count(axis=0) ... a.count(axis=1) ... a.count(axis=2)
    {}\n
    {}\n
    {}\n
    : and finally:  a * 2
    {}\n
    :Return it to a list...
    {}
    """
    a_list = [[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[9, 10, 11], [12, 13, 14], [15, 16, 17]],
              [[18, -1, -1], [21, -1, -1], [24, -1, -1]]]
    mask_val = -1
    a = np.ma.masked_equal(a_list, mask_val)
    a.set_fill_value(mask_val)
    final = a.tolist(mask_val)
    print("\n{}".format(num_46.__doc__))
    args = [a_list, a,
            a.count(axis=0), a.count(axis=1), a.count(axis=2),
            a*2, final]
    print(dedent(frmt).format(*args))
    return a_list, a, final


# ----------------------------------------------------------------------
# num_47 
def num_47():
    """(num_47)... Block with padding and reshaping
       saved in array_tools as well
       """

    def block_reshape(a, rows, cols, nodata=-1, as_masked=True):
        """  """
        s = np.array(a.shape)
        w = np.array([rows, cols])
        m = divmod(s, w)
        new_shape = w*m[0] + w*(m[1]!=0)
        ypad, xpad = new_shape - a.shape
        pad = ((0, ypad), (0, xpad))
        p_with =((nodata, nodata), (nodata, nodata))
        b = np.pad(a, pad_width=pad, mode='constant', constant_values=p_with)
        w_y, w_x = w  # Blocksize
        y, x = b.shape       # padded array
        c = b.reshape((y//w_y, w_y, x//w_x, w_x))
        c = c.swapaxes(1, 2).reshape(-1, w_y, w_x)
        if as_masked:
            mask_val = nodata
            c = np.ma.masked_equal(c, mask_val)
            c.set_fill_value(mask_val)
        return b, c
    y, x = 5, 6
    rows, cols = [3, 4]
    nodata = -1
    a = np.arange(x*y).reshape(y,x)
    b, c = block_reshape(a, rows, cols, nodata)
    print("\n{}".format(num_47.__doc__))
    print("a\n{}\nb\n{}\nc\n{}".format(a, b, c))
    return a, b, c

    
# ----------------------------------------------------------------------
# num_48 
def num_48():
    """(num_48)... Formatting again, using indent and dedent
    :Requires:
    :--------
    :  from textwrap import indent, dedent
    :  - indent(text, prefix, predicate=None)
    :        If predicate not set prefix will be added to all lines.
    :Returns:
    :-------
    :   :    Some text to indent
    :   :Some text to dedent by 5
    :Examples:
    :--------
    :  print(indent(str(a), ":   ", lambda line: True))
    :   [[ 0  1  2  3]
    :    [ 4  5  6  7]
    :    [ 8  9 10 11]]
    :
    :  print(indent(repr(a), ":   ", lambda line: True))
    :   array([[ 0,  1,  2,  3],
    :          [ 4,  5,  6,  7],
    :          [ 8,  9, 10, 11]])
    >>> print(indent(repr(a), ":   "))  same as above
    :   array([[ 0,  1,  2,  3],
    :          [ 4,  5,  6,  7],
    :          [ 8,  9, 10, 11]])
    : b = "1\n:  --2\n:  ---3"
    : print(indent(b, ":  ", lambda line: len(line)>10)
     1          Notice, no indentation because the line is > 10 characters
     --2
     ---3
    : print(indent(b, ":  ", lambda line: len(line)<10)
    :  1
    :  --2
    :  ---3
    :
    : >>> c = dedent("    5\n     56\n") + indent("5\n56",":   ")
     5           You can concatenate dedented and indented strings.
     56
    :   5
    :   56
    :------------------------------
    """
    frmt = """
    :Input formatting option ({}) ...
    :{}\n
    :Subtitle...
    :{}\n
    :An array, double indent..
    {}\n
    :Final line
    """
    pad = ":..."
    pad2 = "   "
    a = "Section title..."
    b = "{}Text indented by 4 spaces".format(pad2)
    c = np.arange(4*5).reshape(4, 5)
    # f = "\n".join([i.strip() for i in frmt.split(":")])
    # print(f.format(1, a, b, c))
    # print(f.format(2, a, b, indent(str(c), pad2)))
    f = dedent(frmt).format(3, a, b, indent(str(c), pad2*2))
    print(f)
    print(indent(f, pad))
    return frmt, c


# ----------------------------------------------------------------------
# num_49 
def num_49():
    """(num_49)... kroneker product and array construction
    """
    frmt = """
    :Kroneker product...
    :Ones array (a) ...
    {}
    :Base array (b) ...
    {}
    :np.kron(a, b)
    {}
    :np.kron(b, a)
    {}
    """
    pad = " "*4
    x = 3
    y = 2
    a = np.ones((y, x), dtype='<i8')
    b = np.arange(y*x).reshape(y, x)
    c = np.kron(a, b)
    d = np.kron(b, a)
    args = [indent(str(i), pad) for i in [a, b, c, d]]
    print("\n{}".format(num_49.__doc__))
    print(dedent(frmt).format(*args))
    return a, b, c, d


# ----------------------------------------------------------------------
# num_50 
def num_50():
    """(num_50)... fancy indexing ....
    """
    a = np.arange(8*8).reshape(8,8)
    b = a[:,::2]
    c = a[:,::-2]
    d = a[::2,::2]
    e = a[::-1,::-1]
    f = a[::-2,::]
    g = a[::2,::]
    args = [a, b, c, d, e, f, g]
    print("\n{}".format(num_50.__doc__))
    print(("{}\n\n"*len(args)).format(*args))


# ----------------------------------------------------------------------
# num_51 
def num_51(): # doesn't work with numpy 1.4
    """
    :  http://central.scipy.org/item/84/1/simple-interactive-matplotlib-plots
    :  https://geonet.esri.com/thread/185110-matplotlib-show-prevents-
    :        script-from-completing
    :  matplotlib.__version__   # '1.4.0'
    """
    import numpy
    import matplotlib.pyplot as plt
    t = np.arange(1, 10, 0.1)
    s = np.sin(t)
    c = np.cos(t)
    plt.plot(t,s)
    plt.title("test of timed pause...")   
    plt.show()    
    #duration = 2
    #plt.pause(duration)
    plt.close()
    return plt


# ----------------------------------------------------------------------
# num_52 
def num_52():
    """num_52() Closeness Manahatten....
    :Reference:
    :---------
    :  http://stackoverflow.com/questions/40434139/
    :       generate-a-closeness-to-center-matrix-in-numpy
    :  http://stackoverflow.com/questions/40126853/fastest-way-to-
    :       build-a-matrix-with-a-custom-architecture
    """
    def closeness_manhattan(N):
        r = np.arange(N)
        a = np.minimum(r,r[::-1])
        return a[:,None] + a
    for i in range(3,8,1):
        a = closeness_manhattan(i)
        print("\nN = {}\n{}".format(i, a))
    a = np.array([1, 3, 5, 7, 9, 7, 5, 3, 1])
    b = np.array([1, 3, 5, 7, 3, 5, 3, 1, 0]) # Biasing array
    c = np.minimum(b[:, None], a)             # using above
    d = np.minimum(a[:, None], b)             # switch arrays
    print("\n{}".format(num_52.__doc__))
    print("\nWith bias\n{}\nBias swapped\n{}".format(c, d))
    return a, b


# ----------------------------------------------------------------------
# num_53 
def num_53():
    """num_53() formatting output ....
    :Reference:
    :---------
    :  http://stackoverflow.com/questions/40443888/print-two-arrays-side-
    :         by-side-using-numpy/40444199#40444199
    :
    """
    a = np.array([[i, np.cos(np.deg2rad(i)), np.sin(np.deg2rad(i))]
                   for i in range(0,361,30)])
    print("\n{}".format(num_53.__doc__))
    args = ["Angle", "Cos", "Sin"]
    print(("{:^6}"+"{:>8}"*2).format(*args))
    frmt = ("{:>6.0f}"+"{:>8.3f}"*2)
    for i in a:
        print(frmt.format(*i))


# ----------------------------------------------------------------------
# num_54
def num_54():
    """num_54() Producing uniformly distributed data
    :Requires:
    :--------
    :  The class numbers have to be specified and the number of repeats
    :  to give you a total population size.
    :Reference:
    :---------
    :  https://geonet.esri.com/thread/185566-creating-defined-lists
    """
    frmt = """
    :{}
    :Generate Data that conform to a uniform distribution.
    :
    :Class values: {}
    :Population size: {}
    :Results:
    :  values:
    {}
    :  table:
    {}
    :  histogram: (class, frequency)
    {}
    :Then use NumPyArrayToTable to get your table.
    """
    # import numpy as np
    st = 1
    end = 7
    vals = np.arange(st,end)
    reps = 10
    z = np.repeat(vals,reps)
    np.random.shuffle(z)
    ID = np.arange(len(z))
    tbl = np.array(list(zip(ID, z)), 
                   dtype = [('ID', 'int'), ('Class', 'int')])
    h = np.histogram(z, np.arange(st, end+1))
    h = np.array(list(zip(h[1], h[0])))
    pad = "    "
    args =[num_54.__doc__, vals, reps*len(vals),
           indent(str(z.reshape(3,20)), pad),
           indent(str(tbl), pad), indent(str(h), pad)]
    print(dedent(frmt).format(*args))
 #then use NumPyArrayToTable to get your table

# ----------------------------------------------------------------------
# num_55
import itertools as IT
def num_55():
    """num_55() combinations and frequency
    : Produce a combinations array from some class values.
    : From the above, get the frequency distribution of the values for
    : a particular axis, (0 for rows, 1 for columns).
    : Solve and present the results in standard and transposed formats.
    : The transpose, is just another was of swapping the axis.  So doing
    : the transpose is lie switching the axis from 0 to 1.
    """
    import itertools as IT
    axis=0
    cs = [-1, 0, 1]
    bins = [-1, 0, 1, 2]
    n = len(cs)
    a = np.array([i for i in IT.combinations_with_replacement(cs, n)])
    r = np.vstack([np.histogram(a[i], bins)[0] for i in range(len(a))])
    r_t = np.vstack([np.histogram(a.T[i], bins)[0] for i in range(len(a.T))])
    frmt = """
    {}
    :classes: {}
    :values (a):
    {}
    :frequency for 'a' by row, axis=0
    {}
    :values (a_t)
    {}
    :frequency for 'r_t', by col, axis=1
    :  Note...  a.T = a_t
    :    transform, a from axis 0 to axis 1 orientation
    {}
    """
    p = "   . "
    args = [num_55.__doc__, cs, 
            indent(str(a), prefix=p),
            indent(str(r), prefix=p),
            a.T, r_t]
    print(dedent(frmt).format(*args))
    return a, r, r.T
#----------------------
if __name__ == "__main__":
    """Main section...   """
    #print("Script... {}".format(script))
    script = sys.argv[0]
#    in_file = os.path.dirname(script) + '/data/csv.txt'
#    a, b = num_40(in_file)  # np.genfromtext example *** requires a file
#    code = num_41()  # Documenting code using inspect
#    a, b, c = num_42()  # Documenting code using inspect
#    a, b, b1, c, d,e  = num_43()  # Blocking an array options
#    a, b, c, d, e, f = num_44()  # a variant on array_split
#    a, b, c, d = num_45()  # bulk create structured array fields
#    A, a, c = num_46()  # masked array from ill-formed list
#    a, b, c = num_47()  # Block with padding and reshaping
#    frmt, c = num_48()  # Formatting again, using indent and dedent
#    a, b, c, d = num_49()  # kroneker product and array construction
#    num_50()  # fancy indexing
#    plt =num_51()  # matplotlib... needs 1.5.x
#    a, b = num_52()  #closeness Manhattan
#    num_53()  # formatting output
#    num_54()  #Producing uniform distribution data
#    a, r, r_t = num_55()  # combinations and frequency
