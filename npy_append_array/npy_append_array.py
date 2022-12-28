import os, numpy, tempfile, threading
from numpy.lib import format
from .format import _write_array_header, write_array
from io import BytesIO, SEEK_END, SEEK_SET

class _HeaderInfo():
    def __init__(self, fp):
        version = format.read_magic(fp)
        shape, fortran_order, dtype = format._read_array_header(fp, version)
        self.shape, self.fortran_order, self.dtype = (
            shape, fortran_order, dtype
        )

        header_size = fp.tell()
        self.header_size = header_size

        new_header = BytesIO()
        _write_array_header(new_header, {
            "shape": shape,
            "fortran_order": fortran_order,
            "descr": format.dtype_to_descr(dtype)
        })
        self.new_header = new_header.getvalue()

        fp.seek(0, SEEK_END)
        self.data_length = fp.tell() - header_size

        self.is_appendable = len(self.new_header) <= header_size

        self.needs_recovery = not (
            dtype.hasobject or self.data_length ==
            numpy.multiply.reduce(shape) * dtype.itemsize
        )

def is_appendable(filename):
    with open(filename, mode="rb") as fp:
        return  _HeaderInfo(fp).is_appendable

def needs_recovery(filename):
    with open(filename, mode="rb") as fp:
        return  _HeaderInfo(fp).needs_recovery

def ensure_appendable(filename, inplace=False):
    with open(filename, mode="rb+") as fp:
        hi = _HeaderInfo(fp)

        new_header_size = len(hi.new_header)

        if hi.is_appendable:
            return True

        new_header, header_size = hi.new_header, hi.header_size
        data_length = hi.data_length

        # Set buffer size to 16 MiB to hide the Python loop overhead, see
        # https://github.com/numpy/numpy/blob/main/numpy/lib/format.py
        buffersize = min(16 * 1024 ** 2, data_length)
        buffer_count = int(numpy.ceil(data_length / buffersize))

        if inplace:
            for i in buffersize * numpy.arange(buffer_count - 1, -1, -1):
                fp.seek(header_size + i, SEEK_SET)
                content = fp.read(buffersize)
                fp.seek(new_header_size + i, SEEK_SET)
                fp.write(content)

            fp.seek(0, SEEK_SET)
            fp.write(new_header)

            return True

        dirname, basename = os.path.split(fp.name)

        fp2 = open(tempfile.NamedTemporaryFile(
            prefix=basename, dir=dirname, delete=False
        ).name, 'wb+')
        fp2.write(new_header)

        fp.seek(header_size, SEEK_SET)
        for _ in range(buffer_count):
            fp2.write(fp.read(buffersize))

    fp2.close()
    os.rename(fp2.name, fp.name)

    return True

def recover(filename, zerofill_incomplete=False):
    with open(filename, mode="rb+") as fp:
        hi = _HeaderInfo(fp)
        shape, fortran_order, dtype = hi.shape, hi.fortran_order, hi.dtype
        header_size, data_length = hi.header_size, hi.data_length

        if not hi.needs_recovery:
            return True

        # if the old header is larger than the new one, it's fine
        if not hi.is_appendable:
            msg = "header not appendable, please call ensure_appendable first"
            raise ValueError(msg)

        append_axis_itemsize = numpy.multiply.reduce(
            shape[slice(None, None, -1 if fortran_order else 1)][1:]
        ) * dtype.itemsize

        trailing_bytes = data_length % append_axis_itemsize

        if trailing_bytes != 0:
            if zerofill_incomplete is True:
                zero_bytes_to_append = append_axis_itemsize - trailing_bytes
                fp.write(b'\0'*(zero_bytes_to_append))
                data_length += zero_bytes_to_append
            else:
                fp.truncate(header_size + data_length - trailing_bytes)
                data_length -= trailing_bytes

        new_shape = list(shape)
        new_shape[-1 if fortran_order else 0] = \
            data_length // append_axis_itemsize

        fp.seek(0, SEEK_SET)
        _write_array_header(fp, {
            "shape": tuple(new_shape),
            "fortran_order": fortran_order,
            "descr": format.dtype_to_descr(dtype)
        }, header_len=header_size)

    return True

class NpyAppendArray:
    fp = None
    __lock, __is_init, __header_length = threading.Lock(), False, None

    def __init__(
        self, filename, delete_if_exists=False,
        rewrite_header_on_append=True
    ):
        self.filename = filename
        self.__rewrite_header_on_append = rewrite_header_on_append

        if os.path.exists(filename):
            if delete_if_exists:
                os.unlink(filename)
            else:
                self.__init_from_file()

    def __init_from_file(self):
        fp = open(self.filename, "rb+")
        self.fp = fp

        hi = _HeaderInfo(fp)
        self.shape, self.fortran_order, self.dtype, self.__header_length = (
            hi.shape, hi.fortran_order, hi.dtype, hi.header_size
        )

        if self.dtype.hasobject:
            raise ValueError("Object arrays cannot be appended to")

        if not hi.is_appendable:
            msg = "header of {} not appendable, please call " + \
            "npy_append_array.ensure_appendable".format(self.filename)
            raise ValueError(msg)

        if hi.needs_recovery:
            msg = "cannot append to {}: file needs recovery, please call " + \
            "npy_append_array.recover".format(self.filename)
            raise ValueError(msg)

        self.__is_init = True

    def __write_array_header(self):
        fp = self.fp
        fp.seek(0, SEEK_SET)

        _write_array_header(fp, {
            "shape": self.shape,
            "fortran_order": self.fortran_order,
            "descr": format.dtype_to_descr(self.dtype)
        }, header_len = self.__header_length)

    def update_header(self):
        with self.__lock:
            self.__write_array_header()

    def append(self, arr):
        with self.__lock:
            if not self.__is_init:
                with open(self.filename, 'wb') as fp:
                    write_array(fp, arr)
                self.__init_from_file()
                return

            shape = self.shape
            fortran_order = self.fortran_order
            fortran_coeff = -1 if fortran_order else 1

            if shape[::fortran_coeff][1:][::fortran_coeff] != \
            arr.shape[::fortran_coeff][1:][::fortran_coeff]:
                msg = "array shapes can only differ on append axis: " \
                "0 if C order or -1 if fortran order"

                raise ValueError(msg)

            self.fp.seek(0, SEEK_END)

            arr.astype(self.dtype, copy=False).flatten(
                order='F' if self.fortran_order else 'C'
            ).tofile(self.fp)

            self.shape = (*shape[:-1], shape[-1] + arr.shape[-1]) \
                if fortran_order else (shape[0] + arr.shape[0], *shape[1:])

            if self.__rewrite_header_on_append:
                self.__write_array_header()

    def close(self):
        with self.__lock:
            if self.__is_init:
                if not self.__rewrite_header_on_append:
                    self.__write_array_header()

                self.fp.close()

                self.__is_init = False

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__del__()