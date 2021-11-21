import numpy as np
import os.path
from io import BytesIO, SEEK_END, SEEK_SET

class NpyAppendArray:
    def __init__(self, filename):
        self.filename = filename
        self.fp = None
        self.__is_init = False
        if os.path.isfile(filename):
            self.__init()

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

    def __init(self, arr = None):
        self.fp = open(self.filename, mode="rb+" if arr is None else "wb")
        fp = self.fp

        if arr is None:
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

        else:
            self.shape, self.fortran_order, self.dtype = \
                list(arr.shape), False, arr.dtype

            fp.write(self.__create_header_bytes())

            self.header_length = fp.tell()

            arr.tofile(fp)

        self.__is_init = True

    def append(self, arr):
        if not arr.flags.c_contiguous:
            raise NotImplementedError("ndarray needs to be c_contiguous")

        if not self.__is_init:
            self.__init(arr)
            return

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

    def close(self):
        if self.__is_init:
            fp = self.fp
            fp.seek(0, SEEK_SET)

            new_header_bytes = self.__create_header_bytes()
            header_length = self.header_length

            print(len(new_header_bytes))

            if header_length != len(new_header_bytes):
                new_header_bytes = self.__create_header_bytes(False)

                print(len(new_header_bytes))

                # This can only happen if array became so large that header space
                # space is exhausted, which requires more energy than is necessary
                # to boil the earth's oceans:
                # https://hbfs.wordpress.com/2009/02/10/to-boil-the-oceans
                if header_length != len(new_header_bytes):
                    raise TypeError("header length mismatch, old: %d, new: %d" % (
                        header_length, len(new_header_bytes)
                    ))

            fp.write(new_header_bytes)
            fp.close()

            self.__is_init = False

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
