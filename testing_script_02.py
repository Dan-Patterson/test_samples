# -*- coding: UTF-8 -*-
"""
:Script:   testing_script_02.py
:Author:   Dan.Patterson@carleton.ca
:Modified: 2016-10-27
:Purpose:  A continuation of sample functions related to python
:   and numpy etc.
:
:Notes:
:
:References:
: - https://geonet.esri.com/thread/162225 
:   my reference on nan's versus masked arrays and all the stats functions.
:
:Functions list .........
:...... np functions .....
:    num_25()  Numpy typecheck for array data types
:    num_26()  masks, nan and data outputs
:    num_27()  list files in folder
:    num_28()  place random values/nan in arrays
:    num_29()  remove string entities using LC's
:    num_30()  encoding and decoding in python
:    num_31()  masked_array formatting
:    num_32()  2D array to XYZ
:    num_33()  nested formatting demos
:    num_34()  Structured arrays
:    num_35()  math on structured arrays
:    num_36()  char and recarray construction
:    num_37()  playing with 3D arrangements...
:    num_38()  combinations in data
:    num_39()  recfunctions join
:    
:....... python general ..
:
:
"""
#---- imports, formats, constants ----
import sys
import numpy as np
from textwrap import dedent
from numpy.lib import recfunctions as rfn

ft={'bool': lambda x: repr(x.astype('int32')),
    'float': '{: 0.2f}'.format}
np.set_printoptions(edgeitems=5, linewidth=120, precision=2,
                    suppress=True, threshold=20,
                    formatter=ft)

script = sys.argv[0]
# ---- one-liners ---
# now the one liners...
small_tips = """
---------------
fld = 'Group'
out = np.array([np.mean(sub.tolist())
                for sub in np.split(a, np.argwhere(np.diff(a[fld])))])
---------------
cond = np.isfinite(np.nan, np.inf, np.NINF)
cond = (a == 0) | a == -99)
mask = np.all(np.isnan(a), axis=1) | np.all(a == 0, axis=1)
arr = arr[~mask]

---------------
for masked arrays, get rid of the -- and replace with -
np.ma.masked_print_option.set_display('-') 
---------------

---------------
"""

#---- functions, constants ----
r, c = [6, 5]
ai = np.arange(r*c).reshape(r, c)
af = np.arange(r*c*1.0).reshape(r, c)
frmt = """
    :--------------
    :Input array...
    :{}\n
    :  ....
    :{}\n
    :Final array...
    :{}\n
    :--------------
    """
#----------------------------------------------------------------------
# num_25  typechar information
def num_25():
    """(num_25)... typechars
    :Requires:
    :--------
    :  just run as is with no inputs
    :
    :Returns
    :-------:
    : typechars, typecodes and typeDictt values
    :-------
    : typechar see:  np.lib.type_check module
    """
    typechars = ['S1', '?',  'O', 'S', 'U', 'V', 'd',
                 'g', 'f', 'i', 'h', 'l']
    typecode = ['Character', 'Datetime',
                 'Integer', 'UnsignedInteger' 'AllInteger',
                 'float', 'Complex', 'AllFloat','All']
    num_keys = np.arange(0, 24).tolist()
    #
    print("\nBy typechar.....\n")
    for typechar in typechars:
        print("{:<3}: {} ".format(typechar, np.typename(typechar)))
    print("\nBy typecode.....\n")
    typecodes = np.typecodes
    for typecode in typecodes:
        print("{:<16}: {}".format(typecode, typecodes.get(typecode)))
    print("\nBy typeDict (numeric)...\n")
    td = np.typeDict
    for k in num_keys:
        print("{:<5}: {}".format(k, td.get(k).__name__))


#----------------------------------------------------------------------
# num_26  masks and queries
def num_26(prn=True):
    """(num_26)... masks, nan and data outputs
    :Requires
    :--------
    :
    :Returns
    :-------
    :
    :Notes
    :-----
    : am = np.ma.MaskedArray(data, mask=nomask, dtype=None,
    :                 copy=False, subok=True, ndmin=0, fill_value=None,
    :                 keep_mask=True, hard_mask=None, shrink=True)
    a = r,c = [6, 5]
    ai = np.arange(r*c).reshape(r,c)
    """
    r, c = [6, 5]
    a = np.arange(r*c).reshape(r, c)
    this = ((a<5) | (a>25))
    b = np.ma.MaskedArray(a, mask=this)
    c = np.where(this, np.nan, a)
    frmt = """
    :--------------
    :{}\n
    :Input array... a
    {}\n
    :mask ....      this = ((a<5) | (a>25))
    {!r:}\n
    :Final array... b = np.ma.MaskedArray(a, mask=this)
    {!r:}\n
    :min: {}  max:{}
    :Nan array...   c = np.where(this, np.nan, a) # upcast dtype
    {}\n
    :min: {}  max:{}
    :--------------
    """
    args = [num_26.__doc__, a, this, b, b.min(), b.max(),
            c, np.nanmin(c), np.nanmax(c)]
    
    if prn:
        print(dedent(frmt).format(*args))
        return None
    else:
        print("\nreturned... a, this, b, c = num_26(prn=False)\n\n")
        return a, this, b, c

#----------------------------------------------------------------------
# num_27 list files in folder
import os
def num_27():
    """(num_27)... list files in folder
    :Requires
    :--------
    :
    :Returns
    :-------
    :
    """    
    def get_dirlist(path):
        """
        Return a sorted list of all entries in path.
        This returns just the names, not the full path to the names.
        """
        dirlist = os.listdir(path)
        dirlist.sort()
        return dirlist

    def print_files(path, prefix = ""):
        """ Print recursive listing of contents of path """
        if prefix == "":  # Detect outermost call, print a heading
            print("Folder listing for", path)
            prefix = "|     "
        dirlist = get_dirlist(path)
        for f in dirlist:
            print(prefix + "- " + f)            # Print the line
            fullname = os.path.join(path, f)   # Turn name into full pathname
            if os.path.isdir(fullname):        # If a directory, recurse.
                print_files(fullname, prefix + "|   ")
        return None
    """dir check"""
    #path = os.getcwd()
    path = '/private/var/mobile/Containers/Shared/AppGroup/A9DDA80F-9432-45DA-B931-2E9386579AE6/Pythonista3/Documents'
    #path = '/private/var/mobile/Containers/Shared/AppGroup/A9DDA80F-9432-45DA-B931-2E9386579AE6'

    print_files(path)
    return None #dirlist
 
#----------------------------------------------------------------------
# num_28 place random values/nan in arrays
def num_28(rows=8, cols=5, sample=10, use_float=False, mask_val=-99):
    """(num_28)... place random values/nan in arrays
    :Requires
    :--------
    : This sample produces a n*m shaped array which you could pass 
    : parameters to, but it is simpler to edit them here. It then
    : produces a random choice set of locations and sets them to a 
    : value which is masked and a true masked array is produced.
    : The location of the selection and their original values is also
    : returned.
    :
    :Returns
    :-------
    :
    """
    np.ma.masked_print_option.set_display('-') 
    fac = [1, 1.0][use_float]
    a = np.arange(rows*cols).reshape(rows, cols) * fac
    sampl = [np.random.choice(a.size, sample, replace=False)]
    b = np.copy(a)
    b = b.ravel()
    b[sampl] = mask_val
    b.shape = a.shape
    m = np.where(b==mask_val, 1, 0)
    c = np.ma.array(b, mask=m, fill_value=mask_val)
    x, y = np.where(c.mask==1)
    xy = np.array(list(zip(x, y)))
    vals = np.array([a[x, y] for x, y in xy])
    dt = [('X', 'i8'), ('Y', 'i8'), ('value', 'float')]
    z = np.zeros((len(vals),), dtype=dt)
    z['X'] = xy[:, 0]
    z['Y'] = xy[:, 1]
    z['value'] = vals
    frmt = """
    :--------------
    {}\n
    :Random values in array
    :Input array...
    {!r:}\n
    :  ....
    :Final array...
    {!r:}\n
    :  ....
    :Masked array 
    {!r:}
    :  ....
    :Mask locations
    :{}
    {}
    :--------------
    """
    args = [num_28.__doc__, a, b, c, z.dtype.names,
            z.reshape(z.shape[0], -1)]
    #print(dedent(frmt).format(*args))
    t = str(c)
    r = [['[[', " "],
         ['[', ""],
         [']', ""],
         [']]', ""]
         ]
    for i in r:
        t = t.replace(i[0], i[1])
    #u = t.split(" ")
    u0 = t.split('\n')
    #u = [ j for j in [i.strip().split(' ') for i in u0] if j != "" ]
    u = [i.strip().split(' ') for i in u0]
    v =[]
    for i in range(len(u)):
        f = ('{:>4} '*len(u[i])).format(*u[i])
        v.append(f)
        print(f)
    #v = (' {:>4}'*len(u1)).format(*u)
    #print(v)
    return t, u, v, c #a, b, c, z

#----------------------------------------------------------------------
# num_29 remove string entities using LC's
def num_29():
    """(num_29)... remove string entities using LC's
    :Requires
    :--------
    :
    :Returns
    :-------
    : good   : returns a string '26 48 18.3431  E  39 36 6.979  N'  
    : asList : alternate:  ['26', '48', '18.3431', 'E',
    :                       '39', '36', '6.979', 'N']  
    : 
    : b0 = [[b for b in num] for num in nums]
    :      [[b'0', b'1', b'2', b'3'], [b'0', b'1', b'2', b'3']]
    : b1 = [[b.decode('utf8') for b in num] for num in nums]
    :      [['0', '1', '2', '3'], ['0', '1', '2', '3']]
    :References:
    :   https://geonet.esri.com/thread/179675 
    :   http://stackoverflow.com/questions/38338335/obtaining-integer-
    :           from-string-when-converting-from-python-2-to-python-3
    :
    """
    DATA = r"26°48'18.3431\"E  39°36'6.979\"N"
    nogood = ["°", "'", '\\','"']  
    good = ''.join([ [i, ' '][i in nogood] for i in DATA])  
    asList = [ i for i in good.split(" ") if i!='' ]
    #
    nums = [[b'0', b'1', b'2', b'3'], [b'0', b'1', b'2', b'3']]
    b0 = [[b for b in num] for num in nums]
    b1 = [[b.decode('utf8') for b in num] for num in nums]    
    b2 = [int("".join(t)) for t in b1]    # b0 returns an error
    args = [good, asList, b0, b1, b2]
    frmt = "good: {}\nasList:  {}\nb0:  {}\nb1:  {}\nb2:  {}"
    print(frmt.format(*args))
    return None

#----------------------------------------------------------------------
# num_30  ... encoding and decoding in python
def num_30():
    """(num_30)... encoding and decoding in python
    :Requires
    :--------
    :
    :Returns
    :-------
    :
    : vals = list('ABCDEF012345')  # done in python 3.4.x
    : old_27 = [ v.encode('ascii') for v in vals]
    :   ==>    [b'A', b'B', b'C', b'D', b'E', b'F',
    :           b'0', b'1', b'2', b'3', b'4', b'5']
    : new_3x = [ old.decode('utf8') for old in old_27]
    :   ==>    ['A', 'B', 'C', 'D', 'E', 'F',
    :           '0', '1', '2', '3', '4', '5']
    :
    : >>> s = u'\u0103'
    : >>> print s.encode('raw_unicode_escape')
    : \u0103
    :
    :  https://geonet.esri.com/thread/121816  my post
    :  print '\noption 1:  List comprehension with unicode'  
    :  a =  tuple([unicode(item[key]) for key in keys])  
    :  # list comprehension with unicode
    : 
    :References:
    :  https://docs.python.org/3/library/codecs.html#standard-encodings ***
    :  http://www.unicode.org/charts/nameslist/ ***
    :  http://unicodelookup.com
    :  http://unicode-table.com/en/sets/arrows-symbols/
    :  http://www.unicode.org/charts/
    :    http://unicode.org/charts/PDF/U2190.pdf
    :    http://www.unicode.org/emoji/charts/emoji-released.html
    :    http://www.unicode.org/charts/PDF/U2200.pdf
    :    http://www.unicode.org/charts/PDF/U2B00.pdf
    :
    """

    def test_(base = r'\u00', err='replace',  prn=True):
        """print Miscellaneous Symbols and Arrows
        : - Range: 2B00–2BFF  Note... there a number missing
        : - http://www.unicode.org/charts/PDF/U2B00.pdf
        : - errors = 'ignore', strict', 'replace', 'xmlcharrefreplace'
        :
        :Examples
        : Controls and Latin  u00
        : Latin extended A    u01
        : Latin extended B    u02
        : Greek and Coptic    u03 0370 - 03FF
        : Cyrillic            u04
        : Cyrillic suppl      u05
        : Arabic              u06
        : Cdn Unified Aborig  u14 1400- 16FF
        : Phonetics Ext       u1D
        : Latin Ext. Addit.   u1E
        : Greek Extended      u1F
        : General Punctuation u20
        : Letterlike Symbols  u21
        : Math Operators      u22 **
        : Misc Tech           u23
        : Misc. symbols       u26
        : Dingbats            u27
        : Braille             u28
        : Suppl Math operat   u2A
        :
        :  t_ = test_(r'\\u22', prn=False) # format base = r'\\u20'
        :  return t_
        """
        #base = r'\u2B'
        chars = list('0123456789ABCDEF')
        encoding = 'raw_unicode_escape'
        r = [(base + ch).encode(encoding, errors=err) for ch in chars]
        x = [[r[i] + chars[j].encode(encoding, errors=err)
             for i in range(len(r))] 
             for j in range(len(chars))]
        xu = [[x[j][i].decode('unicode_escape',errors=err)
              for i in range(len(x[j]))]
              for j in range(len(x))]
        a = np.array(xu, dtype='<U3')
        #for i in xu: print(i)
        if prn:
            print("\nTest cases... base {}\n{}".format(base, a))
        return a
    test_(base=r'\u03')  # change base using above

#----------------------------------------------------------------------
# num_32 masked_array formatting
def num_31(prn=True):
    """(num_31)...masked array formatting
    :Requires
    :--------
    : a masked array for input
    :Returns
    :-------
    : either a print out or a string representation of the array
    :
    """
    frmt = """
    :
    :--------------------
    :masked array.......
    :  ndim: {} size: {}
    :  shape: {}
    :....
    """
    np.ma.masked_print_option.set_display('-') 
    a = np.array([[ 0, 1, 2, -99,   4], [ 5, 6, 7, -99, 9],
       [-99,  11, -99,  13, -99], [ 15,  16,  17,  18,  19],
       [-99, -99,  22,  23,  24], [ 25,  26,  27,  28, -99],
       [ 30,  31,  32,  33, -99], [ 35, -99,  37,  38,  39]])
    m = np.where(a == -99, 1, 0)
    mask_val = -99
    a = np.ma.array(a, mask=m, fill_value=mask_val)
    # 
    # ---- on to the formatting
    tmp = str(a)
    r = [['[[', " "],
         ['[', ""],
         [']', ""],
         [']]', ""]
         ]
    for i in r:
        tmp = tmp.replace(i[0], i[1])
    tmp0 = [ i.strip().split(' ') for i in tmp.split('\n')]
    out =[]
    for i in range(len(tmp0)):
        out.append(('{:>4} '*len(tmp0[i])).format(*tmp0[i]))
    v = "\n".join([i for i in out])
    f = dedent(frmt).format(a.ndim, a.size, a.shape)
    v = f + '\n' + v
    if prn:
        print(v)
    return v
#----------------------------------------------------------------------
# num_32 2D ndarray xyz array
def num_32():
    """(num_32)...2D array to XYZ
    :Requires
    :--------
    :
    :Returns
    :-------
    :
    """
    a = np.random.randint(0, 10, size=(4, 3))
    a_st = []
    idx = np.indices(a.shape)
    x = idx[0].flatten()
    y = idx[1].flatten()
    vals = a.flatten()
    xy = list(zip(x, y, vals))
    #dt = [('Shape',[('X','<f8'),('Y','<f8')]), ('ID','<i4')]
    #dt = [('XY', ('<f8', (2,))), ('Val','<i4')]
    dt = [('X', '<f8'), ('Y', '<f8'), ('Val', '<i4')]
    xy = np.asarray(xy, dtype=dt)
    print("syntax... a\n{!r:}\nxy ...\n{}".format(a, xy))
    return a, xy

#----------------------------------------------------------------------
# num_33
def num_33():
    """(num_33)... nested formatting demos
    :Requires
    :--------
    :
    :Returns
    :-------
    :
    : Format strings
    : - f     "| {{: >{}}} "
    : - txt   ": >","!s: <"
    : - frmt  " ({}) "
    : - m     "|  {{!s:^{}}} "
    : - Row numbers from a header line.
    :  fac = int(len(hdr)/10.)
    :  nums = "0123456789"*fac + "\n"
    :  msg = hdr + nums + "-"*len(hdr) + "\n"
    : - see ...title =... selection using booleans
    :
    """
    a0 = [1, 10, 100]
    nums = "{{:> {width}.{dec}f}}".format(width=8, dec=2) # float format
    f_n = nums*len(a0)
    print("Float values...\n" + f_n.format(*a0))
    #
    nums = "{{:> {width}.{dec}f}}".format(width=8, dec=0) # integer format
    i_n = nums*len(a0)
    print("Integer values...\n" + i_n.format(*a0))
    #
    return a0

#----------------------------------------------------------------------
# num_34  Structured array
def num_34():
    """(num_34)... Structured array
    :Requires
    :--------
    : numpy.lib._iotools import easy_dtype as easy
    :   ndtype = np.dtype(list(zip(names, formats)))
    :
    :Returns
    :
    :Notes:   x2[['x', 'y']].view((np.float64, 2)) 
    :-------
    :
    """
    from numpy.lib._iotools import easy_dtype as easy
    # ---- Generate array data, a list of tuples rather than a list of lists
    #
    fac = 1  # 1 for int, 1.0 for float
    a = np.arange(4*3).reshape(4, 3)*fac
    N, M = shp = a.shape
    dt = a.dtype.name
    data = [tuple(i) for i in a]
    frmt =" Array...\n{}\ndtype... {}\nData as tuples...\n{}"
    print(frmt.format(a, dt, data))
    #
    #ndtype = np.dtype(list(zip(names, formats)))
    def_names = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"][:shp[1]]
    def_dtype = [a.dtype.name]
    dt = [(i, j) for i in def_names for j in def_dtype]
    z0 = np.empty((N,), dtype=dt)
    for idx, name  in enumerate(def_names):
        z0[name] = a[:,idx]
    print("New structured array...\n{}\ndtype... {}".format(z0, z0.dtype)) 
    #
    print("\nNow the 'easy' way...")
    dt0 = easy(float, names="A, B, C")
    dt1 = easy(("i4, f8, f8"))
    dt2 = easy(["<i4", "<f8", "U5"])
    dt3 = easy("i4, f8, U8", defaultfmt="Col_%02i")
    dt4 = easy((int, float, float), names="A, B, C")
    dts = [dt0, dt1, dt2, dt3, dt4]
    for dt in dts:
        zt = np.asarray(data, dtype=dt)
        print("\n{!r:}".format(zt.reshape(N, 1)))
    #
    # very simple example
    names = ['A','B','C']
    types = [int, float, "U5"]
    N = a.shape[0]
    dt = list(zip(names, types))
    a_st = np.empty((N,), dtype=dt)
    for idx, name  in enumerate(names):
        a_st[name] = a[:, idx]
    #return a, z, a_st

#----------------------------------------------------------------------
# num_35  math on structured arrays
def num_35():
    """  math on structured arrays
    :Requires
    :--------
    :
    :Returns
    :-------
    : http://stackoverflow.com/questions/38722747/unable-to-subtract-specific-fields-within-structured-numpy-arrays
    """
    raw = [('residue', '<i4'), ('pos', [('x', '<f8'), ('y', '<f8'),('z', '<f8')])]
    a= np.empty([0, 2], dtype=raw)
    b = np.empty([0, 2], dtype=raw)
    a = np.append(a, np.array([(1, (1, 2, 3))], dtype=raw))
    b = np.append(b, np.array([(1, (1, 2, 3))], dtype=raw))

    print(a['pos'], b['pos'])  # prints fine
    diff = a['pos'].view((float,(3,))) - b['pos'].view((float,(3,)))
    print(diff) # errors with ufunc error
    #return a, b


#----------------------------------------------------------------------
# num_36 char and recarray construction
def num_36(prn=True):
    """(num_36)... char and recarray construction
    :Requires
    :--------
    :  print  - num_36(prn=True)
    :  values - vals, a, dt = num_36(prn=False)
    :
    :Returns
    :-------
    :  Print or return as above
    :
    :Notes:
    :-----
    : - class chararray in ... numpy/core/defchararray.py
    : - chararray(shape, itemsize=1, unicode=False,  buffer=None,
    :             offset=0, strides=None, order=None)
    : - chararray always strips spaces from the right side of the array but
    :   not the left. You can apply functions to a whole array at once
    :   I used np.char.array to create the array so string functions work.
    :
    :     vals = [' a', 'b    ', 'c', ' D ', '  e ']
    :     a = np.array(vals)
    :     a_c = np.char.array(vals)
    :     a.upper()    # produces an error
    :     a_c.upper()  # upper  - [' A' 'B' 'C' ' D' '  E']
    :     a_c   capitalize(), lower(), lstrip(), title(), upper()   etc.
    :
    :     a = np.array(vals)
    :       = array([' a', 'b    ', 'c', ' D ', 'e  f '], dtype='<U5')
    :     type(a)  -  <class 'numpy.ndarray'>
    :
    :     a1 = np.char.array(vals)
    :     a2 = np.rec.array(vals)
    :          np.rec.array(vals, dtype=dt)
    :          np.core.records.fromrecords(vals)
    :     type(a1) -  <class 'numpy.core.defchararray.chararray'>
    :     type(a4) -  <class 'numpy.core.records.recarray'>
    :     a1  - chararray([' a', 'b', 'c', ' D', 'e  f'], dtype='<U5')
    :     a2  -rec.array((' a', 'b    ', 'c', ' D ', 'e  f '), 
    :             dtype=[('f0', '<U8'), ('f1', '<U20'), ('f2', '<U4'),
    :                    ('f3', '<U12'), ('f4', '<U20')])
    """    
    vals = [[' a', 'b    ', ' C ', 'd  e ', 1, 2.0],
            ['1 ', '2    ', ' 4 ', '5  6 ', 10, 20.0]
            ]
    t = [['U20',
         ['<f8', '<i4'][isinstance(i, int)]][isinstance(i, (int, float))]
         for i in vals[0]]
    lets = list('abcdefghijk')[:len(vals[0])]
    dt = list(zip(lets, t))
    a0 = np.array(vals)
    a1 = np.char.array(vals)
    a2 = np.string_(vals)
    a3 = np.rec.array(vals, dtype=dt)
    a4 = np.core.records.fromrecords(vals)
    a5 = np.rec.array(vals)   
    frmt = """
    :--------------------
    : Input values (vals)
    :  - {}
    :
    """
    out = dedent(frmt).format(vals)
    #    
    sub = """    :--------------------
    : {}
    : {}
    : {}
    : shape {}
    : descr{}
    :
    """
    args = [['np.array(vals)', a0, type(a0), a0.shape, a0.dtype.descr],
            ['np.char.array(vals)', a1, type(a1), a1.shape, a1.dtype.descr],
            ['np.string_(vals)', a2, type(a2), a2.shape, a2.dtype.descr],
            ['np.rec.array(vals, dtype=dt)', a3, type(a3), a3.shape, a3.dtype.descr],
            ['np.core.records.fromrecords(vals)', a4, type(a4), a4.shape, a4.dtype.descr],
            ['np.rec.array(vals)', a5, type(a5), a5.shape, a5.dtype.descr]
           ]
    for arg in args:
        out += dedent(sub).format(*arg)
    if prn:
        print(out)
    else:
        print("syntax... vals, a, dt = num_36(prn=False)")
        return vals, a0, dt

#----------------------------------------------------------------------
# num_37
def num_37():
    """(num_37) playing with 3D arrangements... 
    :Requires:
    :--------
    :  Arrays are generated within... nothing required
    :Returns:
    :-------
    :  An array of 24 sequential integers with shape = (2, 3, 4)
    :Notes:
    :-----
    :  References to numpy, transpose, rollaxis, swapaxes and einsum.
    :  The arrays below are all the possible combinations of axes that can be 
    :  constructed from a sequence size of 24 values to form 3D arrays.
    :  Higher dimensionalities will be considered at a later time.
    :
    :  After this, there is some fancy formatting as covered in my previous blogs. 
    """
    nums = np.arange(24)        #  whatever, just shape appropriately
    a = nums.reshape(2, 3, 4)   #  the base 3D array shaped as (z, y, x) 
    a0 = nums.reshape(2, 4, 3)  #  y, x axes, swapped
    a1 = nums.reshape(3, 2, 4)  #  add to z, reshape y, x accordingly to main size
    a2 = nums.reshape(3, 4, 2)  #  swap y, x
    a3 = nums.reshape(4, 2, 3)  #  add to z again, resize as before
    a4 = nums.reshape(4, 3, 2)  #  swap y, x
    frmt = """
    Array ... {} :..shape  {}
    {}
    """
    args = [['nums', nums.shape, nums],
            ['a', a.shape, a], ['a0', a0.shape, a0],
            ['a1', a1.shape, a1], ['a2', a2.shape, a2],
            ['a3', a3.shape, a3], ['a4', a4.shape, a4],
            ]
    for i in args:
        print(dedent(frmt).format(*i))
    #return a
#----------------------------------------------------------------------
# num_38
def num_38():
    """(num_38)... combinations in data
    :Requires:
    :--------
    :  Alter the 'a' list, for the desired number of classes.
    :
    :Returns:
    :-------
    :  The combinations of the inputs, essentially a unique set.
    :
    :References:
    :  http://stackoverflow.com/questions/8560440/removing-duplicate-columns
    :  -and-rows-from-a-numpy-2d-array
    :-------
    """
    import itertools as IT
    #
    def unique_rows(a):
        a = np.ascontiguousarray(a)
        u_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
        u_a = u_a.view(a.dtype).reshape((u_a.shape[0], a.shape[1]))
        return u_a
    #
    frmt = """
    :------------------------------------------------------------------:
    :Given {0} variables and {1} classes/variable, the following shows     :
    :  (1) the combinations                                            :
    :  (2) all arrangements, {0} variables with {1} classes/variable &     :
    :  (3) those combinations, where positions are not important.      :
    :
    :Input categories per variable... {2}\n
    :Combinations: no duplicates... n={3}
    {4}\n
    :mesh form: {0} variables, all arrangements... n={5}
    :  transposed for viewing...
    {6}\n
    :unique from mesh: {0} variables, arrangement not important... n={7} 
    :  transposed for viewing...
    {8}
    :
    :------------------------------------------------------------------:
    """
    a = [-1, 0, 1] #[0, 1, 2, 3]  #  classes
    n = len(a)
    m = 3 #2
    c = [i for j in range(n+1) for i in IT.combinations(a, j)]
    z = np.array(a*m).reshape(m, n)
    ms = np.array(np.meshgrid(*z)).T.reshape(-1,m)
    s = np.sort(ms, axis=1)
    u = unique_rows(s)
    if n == 4:
        args = [[c[0]], c[1:5], c[5:11], c[11:15], c[-1]]
        c2 = "{}\n{}\n{}\n{}\n{}".format(*args)
    elif n == 3:
        args = [[c[0]], c[1:4], c[4:7], c[7:]]
        c2 = "{}\n{}\n{}\n{}".format(*args)
    args2 = [m, n, a, len(c), c2, len(ms), ms.T, u.shape[0], u.T]
    print(dedent(frmt).format(*args2))
    #return a, c, m, u

#----------------------------------------------------------------------
# num_xx
def num_39():
    """(num_39)...recfunctions join
    :Requires
    :--------
    :
    :Returns
    :-------
    :
    : desired array([(u'a', 1, -1), (u'b', 2, -2), 
    :                (u'b', 2, 0), (u'c', 3, 0)], dtype=[])
    """
    from numpy.lib import recfunctions as rfn
    from textwrap import dedent 
    dt = [('key', '<U5'), ('x', '<i4')]
    a = np.array([('a', 1), ('b', 2), ('b', 2), ('c', 3)], dtype=dt)
    au = np.unique(a)
    b = np.array([('a', -1), ('b', -2)], dtype=dt)
    a_b = rfn.join_by('key', a, b, jointype='outer')
    au_b = rfn.join_by('key', au, b, jointype='outer')
    b_a = rfn.join_by('key', b, a, jointype='outer')
    frmt = """
    :------------------------------------------------------------------:
    :Joins in structured arrays, recfunctions.join_by
    :Arrays a, a_u and b\n    :a....
    {!r:}\n    :a_u....  
    {!r:}\n    :b....
    {!r:}\n\n
    :Join a <== b  (b to a)
    {}\n
    :Join au <== b  (b to au)
    {}\n
    :Join b <== a  (a to b)
    {}\n\n
    :au <== b  (b to au) in detail\n    :
    {!r:}
    :
    :------------------------------------------------------------------:
    """
    args = [a, au, b, a_b, au_b, b_a, au_b.reshape(au_b.shape[0], 1)]
    print(dedent(frmt).format(*args))
    #return a, au, b, au_b

#----------------------------------------------------------------------
# num_40
def num_40():
    """(num_40)... np.r_ and irregular/discontinuous ranges
    :Notes:
    :-----
    : example given: a = np.r_[0:2:0.1, 2:10:0.5, 10:20:1]
    :  np.logspace(2.0, 3.0, num=8, base=2)
    : 
    """
    frmt = """
    :------------------------------------------------------------------:
    :Sequential data with discontinuous ranges
    :(1) 0-2 by 0.1....
    {}
    :(2) 2-10 by 0.5...
    {}
    :(3) 10-20 by 1.0..
    {}
    :
    :------------------------------------------------------------------:
    """
    ft2={'float': '{: 0.1f}'.format}
    np.set_printoptions(edgeitems=5,precision=1, threshold=15, formatter=ft2)
    a = np.r_[0:2:0.1, 2:10:0.5, 10:20:1]
    print(dedent(frmt).format(a[:20], a[20:36], a[36:]))
    np.set_printoptions(edgeitems=5, linewidth=120, precision=2,
                        suppress=True, threshold=20,
                        formatter=ft)
    return a
#----------------------------------------------------------------------
# make various arrays in case they are needed
"""
a = np.array([[-2, 1], [2, 1], [1, -2], [1, 2], [1, 1]])
b = np.array([[3, 3], [5, 5]])
c = np.vstack((a + b[:, None, :]))
d = a + b[:, None, :]  # produces a 3D array
e = (a + b[:, None, :]).reshape(-1, a.shape[1])
f = np.vstack((a*b[:, None, :]))
g = a[:, None, :] + b

# ---- a structured array ----
x = [[[194, 438]], [[495, 431]], [[512, 519]], [[490, 311]], 
     [[548,  28]], [[407, 194]], [[181, 698]], [[169,  93]], 
     [[408,  99]], [[221, 251]], [[395, 692]], [[574, 424]], 
     [[431, 785]], [[538, 249]], [[397, 615]], [[306, 237]]]
 
x = np.array(x).squeeze()
dt = [('x', '<i4'), ('y', '<i4')]
z = np.zeros((len(x),), dtype=dt)
z['x'] = x[:, 0]
z['y'] = x[:, 1]
"""

#----------------------
if __name__=="__main__":
    """   """
    #print("Script... {}".format(script))
    # ......  construction .....
    #num_25() # Numpy typecheck for array data types
    #num_26() # masks and data outputs
    #num_27()
    #num_28() # place random values in an array
    #num_29() # remove string entries using LCs
    #num_30() # encoding and decoding in python
    #num_31() # masked array formatting
    #num_32() # 2D array to XYZ
    #num_33() # nested formatting demos
    #num_34() # Structured array
    #num_35() # math on structured arrays
    #num_36(prn=True) # char and recarray construction
    #num_37() # playing with 3D arrangements...
    #num_38() # combinations is data
    #num_39()  # recfunctions join
    a = num_40()

