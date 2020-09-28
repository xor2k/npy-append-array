import numpy as np
import os.path

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
            raise NotImplementedError("version (%d, %d) unsupported"%magic)

        self.header = np.lib.format.read_array_header_1_0(fp) if \
            self.is_version_1 else np.lib.format.read_array_header_2_0(fp)

        if self.header[1] != False:
            raise NotImplementedError("fortran_order unsupported")

        fp.seek(0, 2)

        self.__is_init = True

    def append(self, arr):
        if not self.__is_init:
            np.save(self.filename, arr)
            self.__init()
            return

        new_header = np.lib.format.header_data_from_array_1_0(arr)
        if arr.flags.f_contiguous or not arr.flags.c_contiguous:
            raise NotImplementedError("ndarray needs to be c_contiguous")

        descr = self.header[2]
        arr_descr = np.lib.format.dtype_to_descr(arr.dtype)

        if arr_descr != descr:
            raise TypeError("ndarrays have incompatible types %s and %s"%(
                arr_descr, descr
            ))

        shape = self.header[0]

        if len(arr.shape) != len(shape):
            raise TypeError("ndarrays have incompatible shape lengths %s and %s"%(
                len(arr.shape), len(shape)
            ))

        for i, e in enumerate(shape):
            if i > 0 and e != arr.shape[i]:
                raise TypeError("ndarray shapes can only differ on zero axis")

        # we do not account for super large shapes

        new_shape = list(shape)
        new_shape[0] += arr.shape[0]
        new_shape = tuple(new_shape)
        self.header = (new_shape, self.header[1], self.header[2])

        self.fp.seek(0)

        new_header_map = {
            'descr': arr_descr,
            'fortran_order': False,
            'shape': new_shape
        }

        if self.is_version_1:
            np.lib.format.write_array_header_1_0(self.fp, new_header_map)

        else:
            np.lib.format.write_array_header_2_0(self.fp, new_header_map)

        self.fp.seek(0, 2)

        arr.tofile(self.fp)

    def __del__(self):
        if self.fp is not None:
            self.fp.close()