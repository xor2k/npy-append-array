import numpy as np
import os.path
from struct import pack
from io import BytesIO

def _create_header_bytes(header_map, spare_space=False):
    io = BytesIO()
    np.lib.format.write_array_header_2_0(io, header_map)

    if spare_space:
        io.getbuffer()[8:12] = pack("<I", int(
            io.getbuffer().nbytes-12+64
        ))
        io.getbuffer()[-1] = 32
        io.write(b" "*64)
        io.getbuffer()[-1] = 10

    return io.getbuffer()
class NpyAppendArray:
    def __init__(self, filename):
        self.filename = filename
        self.fp = None
        self.__is_init = False
        if os.path.isfile(filename):
            self.__init()

    def __init(self):
        self.fp = open(self.filename, mode="rb+")
        fp = self.fp

        magic = np.lib.format.read_magic(fp)

        if not (magic[0] == 2 and magic[1] == 0): raise NotImplementedError(
            "version (%d, %d) not implemented" % magic
        )

        self.header = np.lib.format.read_array_header_2_0(fp)

        if self.header[1] != False:
            raise NotImplementedError("fortran_order not implemented")

        self.header_length = fp.tell()

        self.__is_init = True

    def append(self, arr):
        if not arr.flags.c_contiguous:
            raise NotImplementedError("ndarray needs to be c_contiguous")

        arr_descr = np.lib.format.dtype_to_descr(arr.dtype)

        if not self.__is_init:
            with open(self.filename, "wb") as fp0:
                fp0.write(_create_header_bytes({
                    'descr': arr_descr,
                    'fortran_order': False,
                    'shape': arr.shape
                }, True))
                arr.tofile(fp0)

            # np.save(self.filename, arr)
            self.__init()
            return

        descr = self.header[2]

        if arr_descr != descr:
            raise TypeError("incompatible ndarrays types %s and %s"%(
                arr_descr, descr
            ))

        shape = self.header[0]

        if len(arr.shape) != len(shape):
            raise TypeError("incompatible ndarrays shape lengths %s and %s"%(
                len(arr.shape), len(shape)
            ))

        if not all(l1 == l2 for l1, l2 in zip(shape[1:], arr.shape[1:])):
            raise TypeError("ndarray shapes can only differ on zero axis")

        new_shape = list(shape)
        new_shape[0] += arr.shape[0]
        new_shape = tuple(new_shape)
        self.header = (new_shape, self.header[1], self.header[2])

        self.fp.seek(0)

        new_header = self.header
        new_header_map = {
            'descr': np.lib.format.dtype_to_descr(new_header[2]),
            'fortran_order': new_header[1],
            'shape': new_header[0]
        }

        new_header_bytes = _create_header_bytes(new_header_map, True)
        header_length = self.header_length

        if header_length != len(new_header_bytes):
            new_header_bytes = _create_header_bytes(new_header_map)

            if header_length != len(new_header_bytes):
                raise TypeError("header length mismatch, old: %d, new: %d"%(
                    header_length, len(new_header_bytes)
                ))

        self.fp.write(new_header_bytes)

        self.fp.seek(0, 2)

        arr.tofile(self.fp)

    def __del__(self):
        if self.fp is not None:
            self.fp.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()
