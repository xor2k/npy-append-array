import numpy as np
import os.path
from io import BytesIO, SEEK_END, SEEK_SET

class NpyAppendArray:
    """
    appends/writes numpy arrays to file.

    :Example:
    ----------------
    >>> fname = 'c:/temp/temp.npy'
    >>> arr = np.random.normal(0,1, (100,10))
    >>> with NpyAppendArray(fname) as npa:
    >>>     npa.write(arr)
    >>>     npa.append(arr)    
    >>> assert np.load(fname).shape == (200, 10)
    """
    def __init__(self, filename):
        self.filename = filename
        self.fp = None
        self.__is_init = None

    def make_file_appendable(self):
        """
        if format of file is not amenable to append, resave it 
        """
        if os.path.isfile(self.filename):
            with open(self.filename, mode="rb+") as fp:
                magic = np.lib.format.read_magic(fp)
        if magic != (2, 0):
            arr = np.load(self.filename)
            self.write(arr)

    def __create_header_bytes(self, spare_space = True):
        from struct import pack
        header_map = {
            'descr': np.lib.format.dtype_to_descr(self.dtype),
            'fortran_order': self.fortran_order,
            'shape': tuple(self.shape)
        }
        io = BytesIO()
        np.lib.format.write_array_header_2_0(io, header_map)

        # create array header with 64 byte space space for shape to grow
        io.getbuffer()[8:12] = pack("<I", int(
            io.getbuffer().nbytes-12+(64 if spare_space else 0)
        ))
        if spare_space:
            io.getbuffer()[-1] = 32
            io.write(b" "*64)
            io.getbuffer()[-1] = 10

        return io.getbuffer()

        
    def __init(self):
        if not os.path.isfile(self.filename):
            self.__is_init = False
            return

        self.fp = open(self.filename, mode="rb+")
        fp = self.fp
        magic = np.lib.format.read_magic(fp)

        if magic != (2, 0):
            raise NotImplementedError(
                "version (%d, %d) not implemented" % magic
            )

        header = np.lib.format.read_array_header_2_0(fp)
        shape, self.fortran_order, self.dtype = header
        self.shape = list(shape)

        if self.fortran_order == True:
            raise NotImplementedError("fortran_order not implemented")

        self.header_length = fp.tell()

        header_length = self.header_length

        new_header_bytes = self.__create_header_bytes()

        if len(new_header_bytes) != header_length:
            raise TypeError("no spare header space in target file %s" % (
                self.filename
            ))

        self.fp.seek(0, SEEK_END)
        self.__is_init = True

    def __write_header(self):
        fp = self.fp
        fp.seek(0, SEEK_SET)

        new_header_bytes = self.__create_header_bytes()
        header_length = self.header_length

        if header_length != len(new_header_bytes):
            new_header_bytes = self.__create_header_bytes(False)

            # This can only happen if array became so large that header space
            # space is exhausted, which requires more energy than is necessary
            # to boil the earth's oceans:
            # https://hbfs.wordpress.com/2009/02/10/to-boil-the-oceans
            if header_length != len(new_header_bytes):
                raise TypeError(
                    "header length mismatch, old: %d, new: %d" % (
                        header_length, len(new_header_bytes)
                    )
                )

        fp.write(new_header_bytes)
        fp.seek(0, SEEK_END)
        
    def write(self, arr):
        """
        writes an array to self.filename, overwriting existing file if there
        """
        fp = self.fp  = open(self.filename, mode="wb")
        self.shape, self.fortran_order, self.dtype = list(arr.shape), False, arr.dtype
        fp.write(self.__create_header_bytes())
        self.header_length = fp.tell()
        arr.tofile(fp)        

    def append(self, arr):
        if not arr.flags.c_contiguous:
            raise NotImplementedError("ndarray needs to be c_contiguous")

        if self.__is_init is None:
            self.__init()
        
        if self.__is_init is False:
            self.write(arr)

        if arr.dtype != self.dtype:
            raise TypeError("incompatible ndarrays types %s and %s" % (
                arr.dtype, self.dtype
            ))

        shape = self.shape

        if len(arr.shape) != len(shape):
            raise TypeError("incompatible ndarrays shape lengths %s and %s" % (
                len(arr.shape), len(shape)
            ))

        if shape[1:] != list(arr.shape[1:]):
            raise TypeError("ndarray shapes can only differ on zero axis")

        self.shape[0] += arr.shape[0]

        arr.tofile(self.fp)

        self.__write_header()
        
    def close(self):
        if self.__is_init:
            self.fp.close()

            self.__is_init = False

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
