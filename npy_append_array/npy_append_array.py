import numpy as np
import os.path
from struct import pack, unpack
from io import BytesIO

def header_tuple_dict(tuple_in):
    return {
        'shape': tuple_in[0],
        'fortran_order': tuple_in[1],
        'descr': np.lib.format.dtype_to_descr(tuple_in[2])
    }

def peek(fp, length):
    pos = fp.tell()
    tmp = fp.read(length)
    fp.seek(pos)
    return tmp

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
        self.is_version_1 = magic[0] == 1 and magic[1] == 0
        self.is_version_2 = magic[0] == 2 and magic[1] == 0

        if not self.is_version_1 and not self.is_version_2:
            raise NotImplementedError(
                "version (%d, %d) not implemented"%magic
            )

        header_length_tmp, = unpack("<H", peek(fp, 2)) if self.is_version_1 \
            else unpack("<I", peek(fp, 4))

        self.header = np.lib.format.read_array_header_1_0(fp) if \
            self.is_version_1 else np.lib.format.read_array_header_2_0(fp)

        if self.header[1] != False:
            raise NotImplementedError("fortran_order not implemented")

        fp.seek(0)

        self.header_bytes = fp.read(header_length_tmp + (
            10 if self.is_version_1 else 12
        ))

        self.header_length = len(self.header_bytes)

        fp.seek(0, 2)

        self.__is_init = True

    def __create_header_bytes(self, header_map, spare_space=False):
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

    def append(self, arr):
        if not arr.flags.c_contiguous:
            raise NotImplementedError("ndarray needs to be c_contiguous")

        arr_descr = np.lib.format.dtype_to_descr(arr.dtype)

        if not self.__is_init:
            with open(self.filename, "wb") as fp0:
                fp0.write(self.__create_header_bytes({
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

        new_header_map = header_tuple_dict(self.header)

        new_header_bytes = self.__create_header_bytes(new_header_map, True)
        header_length = self.header_length

        if header_length != len(new_header_bytes):
            new_header_bytes = self.__create_header_bytes(new_header_map)

            if header_length != len(new_header_bytes):
                raise TypeError("header length mismatch, old: %d, new: %d"%(
                    header_length, len(new_header_bytes)
                ))

        self.header_bytes = new_header_bytes

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
