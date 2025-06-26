from numpy.lib import format
import warnings
import numpy
import pickle
from numpy.lib.format import header_data_from_array_1_0, isfileobj

# TODO modify Numpy once again so that those functions are not needed anymore

# not available anymore since Numpy 2.3
EXPECTED_KEYS = {'descr', 'fortran_order', 'shape'}

# not available anymore since Numpy 2.3
_header_size_info = {
    (1, 0): ('<H', 'latin1'),
    (2, 0): ('<I', 'latin1'),
    (3, 0): ('<I', 'utf8'),
}

# not available anymore since Numpy 2.3
_MAX_HEADER_SIZE = 10000

# not available anymore since Numpy 2.3
def _filter_header(s):
    """Clean up 'L' in npz header ints.

    Cleans up the 'L' in strings representing integers. Needed to allow npz
    headers produced in Python2 to be read in Python3.

    Parameters
    ----------
    s : string
        Npy file header.

    Returns
    -------
    header : str
        Cleaned up header.

    """
    import tokenize
    from io import StringIO

    tokens = []
    last_token_was_number = False
    for token in tokenize.generate_tokens(StringIO(s).readline):
        token_type = token[0]
        token_string = token[1]
        if (last_token_was_number and
                token_type == tokenize.NAME and
                token_string == "L"):
            continue
        else:
            tokens.append(token)
        last_token_was_number = (token_type == tokenize.NUMBER)
    return tokenize.untokenize(tokens)

# not available anymore since Numpy 2.3
def _read_bytes(fp, size, error_template="ran out of data"):
    """
    Read from file-like object until size bytes are read.
    Raises ValueError if not EOF is encountered before size bytes are read.
    Non-blocking objects only supported if they derive from io objects.

    Required as e.g. ZipExtFile in python 2.6 can return less data than
    requested.
    """
    data = bytes()
    while True:
        # io files (default in python3) return None or raise on
        # would-block, python2 file will truncate, probably nothing can be
        # done about that.  note that regular files can't be non-blocking
        try:
            r = fp.read(size - len(data))
            data += r
            if len(r) == 0 or len(data) == size:
                break
        except BlockingIOError:
            pass
    if len(data) != size:
        msg = "EOF: reading %s, expected %d bytes got %d"
        raise ValueError(msg % (error_template, size, len(data)))
    else:
        return data

# not available anymore since Numpy 2.3
def _check_version(version):
    if version not in [(1, 0), (2, 0), (3, 0), None]:
        msg = "we only support format version (1,0), (2,0), and (3,0), not %s"
        raise ValueError(msg % (version,))

# not available anymore since Numpy 2.3
def _read_array_header(fp, version, max_header_size=_MAX_HEADER_SIZE):
    """
    see read_array_header_1_0
    """
    # Read an unsigned, little-endian short int which has the length of the
    # header.
    import ast
    import struct
    hinfo = _header_size_info.get(version)
    if hinfo is None:
        raise ValueError("Invalid version {!r}".format(version))
    hlength_type, encoding = hinfo

    hlength_str = _read_bytes(fp, struct.calcsize(hlength_type), "array header length")
    header_length = struct.unpack(hlength_type, hlength_str)[0]
    header = _read_bytes(fp, header_length, "array header")
    header = header.decode(encoding)
    if len(header) > max_header_size:
        raise ValueError(
            f"Header info length ({len(header)}) is large and may not be safe "
            "to load securely.\n"
            "To allow loading, adjust `max_header_size` or fully trust "
            "the `.npy` file using `allow_pickle=True`.\n"
            "For safety against large resource use or crashes, sandboxing "
            "may be necessary.")

    # The header is a pretty-printed string representation of a literal
    # Python dictionary with trailing newlines padded to a ARRAY_ALIGN byte
    # boundary. The keys are strings.
    #   "shape" : tuple of int
    #   "fortran_order" : bool
    #   "descr" : dtype.descr
    # Versions (2, 0) and (1, 0) could have been created by a Python 2
    # implementation before header filtering was implemented.
    #
    # For performance reasons, we try without _filter_header first though
    try:
        d = ast.literal_eval(header)
    except SyntaxError as e:
        if version <= (2, 0):
            header = _filter_header(header)
            try:
                d = ast.literal_eval(header)
            except SyntaxError as e2:
                msg = "Cannot parse header: {!r}"
                raise ValueError(msg.format(header)) from e2
            else:
                warnings.warn(
                    "Reading `.npy` or `.npz` file required additional "
                    "header parsing as it was created on Python 2. Save the "
                    "file again to speed up loading and avoid this warning.",
                    UserWarning, stacklevel=4)
        else:
            msg = "Cannot parse header: {!r}"
            raise ValueError(msg.format(header)) from e
    if not isinstance(d, dict):
        msg = "Header is not a dictionary: {!r}"
        raise ValueError(msg.format(d))

    if EXPECTED_KEYS != d.keys():
        keys = sorted(d.keys())
        msg = "Header does not contain the correct keys: {!r}"
        raise ValueError(msg.format(keys))

    # Sanity-check the values.
    if (not isinstance(d['shape'], tuple) or
            not all(isinstance(x, int) for x in d['shape'])):
        msg = "shape is not valid: {!r}"
        raise ValueError(msg.format(d['shape']))
    if not isinstance(d['fortran_order'], bool):
        msg = "fortran_order is not a valid bool: {!r}"
        raise ValueError(msg.format(d['fortran_order']))
    try:
        dtype = format.descr_to_dtype(d['descr'])
    except TypeError as e:
        msg = "descr is not a valid dtype descriptor: {!r}"
        raise ValueError(msg.format(d['descr'])) from e

    return d['shape'], d['fortran_order'], dtype

# slightly modified (hopefully one day published) version of
# https://github.com/numpy/numpy/blob/main/numpy/lib/format.py
GROWTH_AXIS_MAX_DIGITS = 21  # = len(str(8*2**64-1)) hypothetical int1 dtype

def _wrap_header(header, version, header_len=None):
    """
    Takes a stringified header, and attaches the prefix and padding to it
    """
    import struct
    assert version is not None
    fmt, encoding = _header_size_info[version]
    header = header.encode(encoding)
    hlen = len(header) + 1
    padlen = format.ARRAY_ALIGN - ((
        format.MAGIC_LEN + struct.calcsize(fmt) + hlen
    ) % format.ARRAY_ALIGN)
    try:
        header_prefix = format.magic(*version) + struct.pack(
            fmt, hlen + padlen
        )
    except struct.error:
        msg = "Header length {} too big for version={}".format(hlen, version)
        raise ValueError(msg) from None

    # Pad the header with spaces and a final newline such that the magic
    # string, the header-length short and the header are aligned on a
    # ARRAY_ALIGN byte boundary. This supports memory mapping of dtypes
    # aligned up to ARRAY_ALIGN on systems like Linux where mmap()
    # offset must be page-aligned (i.e. the beginning of the file).
    header = header_prefix + header + b' '*padlen 
    
    if header_len is not None:
        actual_header_len = len(header) + 1
        if actual_header_len > header_len:
            msg = (
                "Header length {} too big for specified header "+
                "length {}, version={}"
            ).format(actual_header_len, header_len, version)
            raise ValueError(msg) from None
        header += b' '*(header_len - actual_header_len)
    
    return header + b'\n'


def _wrap_header_guess_version(header, header_len=None):
    """
    Like `_wrap_header`, but chooses an appropriate version given the contents
    """
    try:
        return _wrap_header(header, (1, 0), header_len)
    except ValueError:
        pass

    try:
        ret = _wrap_header(header, (2, 0), header_len)
    except UnicodeEncodeError:
        pass
    else:
        warnings.warn("Stored array in format 2.0. It can only be"
                      "read by NumPy >= 1.9", UserWarning, stacklevel=2)
        return ret

    header = _wrap_header(header, (3, 0))
    warnings.warn("Stored array in format 3.0. It can only be "
                  "read by NumPy >= 1.17", UserWarning, stacklevel=2)
    return header


def _write_array_header(fp, d, version=None, header_len=None):
    """ Write the header for an array and returns the version used

    Parameters
    ----------
    fp : filelike object
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    version : tuple or None
        None means use oldest that works. Providing an explicit version will
        raise a ValueError if the format does not allow saving this data.
        Default: None
    header_len : int or None
        If not None, pads the header to the specified value or raises a
        ValueError if the header content is too big.
        Default: None
    """
    header = ["{"]
    for key, value in sorted(d.items()):
        # Need to use repr here, since we eval these when reading
        header.append("'%s': %s, " % (key, repr(value)))
    header.append("}")
    header = "".join(header)
    
    # Add some spare space so that the array header can be modified in-place
    # when changing the array size, e.g. when growing it by appending data at
    # the end. 
    shape = d['shape']
    header += " " * ((GROWTH_AXIS_MAX_DIGITS - len(repr(
        shape[-1 if d['fortran_order'] else 0]
    ))) if len(shape) > 0 else 0)
    
    if version is None:
        header = _wrap_header_guess_version(header, header_len)
    else:
        header = _wrap_header(header, version, header_len)
    fp.write(header)

def write_array(fp, array, version=None, allow_pickle=True, pickle_kwargs=None):
    """
    Write an array to an NPY file, including a header.
    If the array is neither C-contiguous nor Fortran-contiguous AND the
    file_like object is not a real file object, this function will have to
    copy data in memory.
    Parameters
    ----------
    fp : file_like object
        An open, writable file object, or similar object with a
        ``.write()`` method.
    array : ndarray
        The array to write to disk.
    version : (int, int) or None, optional
        The version number of the format. None means use the oldest
        supported version that is able to store the data.  Default: None
    allow_pickle : bool, optional
        Whether to allow writing pickled data. Default: True
    pickle_kwargs : dict, optional
        Additional keyword arguments to pass to pickle.dump, excluding
        'protocol'. These are only useful when pickling objects in object
        arrays on Python 3 to Python 2 compatible format.
    Raises
    ------
    ValueError
        If the array cannot be persisted. This includes the case of
        allow_pickle=False and array being an object array.
    Various other errors
        If the array contains Python objects as part of its dtype, the
        process of pickling them may raise various errors if the objects
        are not picklable.
    """
    _check_version(version)
    _write_array_header(fp, header_data_from_array_1_0(array), version)

    if array.itemsize == 0:
        buffersize = 0
    else:
        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffersize = max(16 * 1024 ** 2 // array.itemsize, 1)

    if array.dtype.hasobject:
        # We contain Python objects so we cannot write out the data
        # directly.  Instead, we will pickle it out
        if not allow_pickle:
            raise ValueError("Object arrays cannot be saved when "
                             "allow_pickle=False")
        if pickle_kwargs is None:
            pickle_kwargs = {}
        pickle.dump(array, fp, protocol=3, **pickle_kwargs)
    elif array.flags.f_contiguous and not array.flags.c_contiguous:
        if isfileobj(fp):
            array.T.tofile(fp)
        else:
            for chunk in numpy.nditer(
                    array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='F'):
                fp.write(chunk.tobytes('C'))
    else:
        if isfileobj(fp):
            array.tofile(fp)
        else:
            for chunk in numpy.nditer(
                    array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='C'):
                fp.write(chunk.tobytes('C'))