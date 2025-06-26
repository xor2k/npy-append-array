import os, io, npy_append_array, threading
import numpy as np
from itertools import product
from pathlib import Path
from npy_append_array import NpyAppendArray

tmpfile = Path('./tmp/tmp.npy')
tmpfile.unlink(missing_ok=True)

tmpfile.parent.mkdir(exist_ok=True)

total_array_size = 16 * 1024**2 + 17
half_array_size = total_array_size // 2

for fortran_order in [False, True]:
    shape = (27, 11, -1) if fortran_order else (-1, 11, 27)
    order = 'F' if fortran_order else 'C'
    dtype_base = np.uint64
    arr = np.arange(total_array_size, dtype=dtype_base)
    arr = arr.reshape(shape, order=order)

    append_axis_item_count = np.multiply.reduce(arr.shape[
        slice(None, None, -1 if arr.flags.fnc else 1)
    ][1:])

    # very simple method, may break in the future and might need fixing then
    def get_array_header_bytes_from_file():
        npy_bytes = tmpfile.read_bytes()
        possible_sizes = [64, 128, 192]
        header_size = [x for x in possible_sizes if npy_bytes[x-1:x] == b'\n']

        assert(len(header_size) == 1)
        
        return npy_bytes[:header_size[0]]

    # ensure_appendable
    for inplace in [False, True]:
        # "fortran_order": True uses one byte less than "fortran_order": False
        dtype_template = np.dtype([(
            '_'*(24+(1 if fortran_order else 0)), dtype_base
        )])

        np.save(tmpfile, arr.astype(dtype=dtype_template))
        orig_header = get_array_header_bytes_from_file()
         
        with open(tmpfile, 'rb+') as fp:
            # overwrite with a non-appendable header
            space_to_remove = len(orig_header) - 127
            fp.write(orig_header.replace(
                b' ' * space_to_remove + b'\n', b'\n'
            ).replace(b'\'_', b'\'_' + b'_' * space_to_remove))

        assert not npy_append_array.is_appendable(tmpfile)

        npy_append_array.ensure_appendable(tmpfile, inplace=inplace)

        assert npy_append_array.is_appendable(tmpfile)
        assert np.all(arr == np.load(tmpfile).astype(dtype_base))

    # recover
    for zerofill_incomplete in [False, True]:
        np.save(tmpfile, arr)

        os.truncate(tmpfile, (
            tmpfile.stat().st_size + len(get_array_header_bytes_from_file())
        ) // 2)

        npy_append_array.recover(tmpfile, zerofill_incomplete)

        data_length = int((np.ceil if zerofill_incomplete else np.floor)(
            half_array_size / append_axis_item_count
        )) * append_axis_item_count

        arr2 = np.load(tmpfile).flatten(order=order)

        assert data_length == arr2.shape[0]
        assert np.all(arr2[half_array_size + 1:] == 0)
        assert np.all((
            arr.flatten(order=order)[:data_length] == arr2
        )[:half_array_size if zerofill_incomplete else data_length])

tmpfile.unlink()

# test regular append for C order and Fortran arrays
for (
    use_np_save, rewrite_header_on_append, delete_if_exists,
    is_fortran_array1, is_fortran_array2
) in product(*[[False, True]]*5):    
    dtype1 = None
    dtype2 = None
    
    order1 = 'F' if is_fortran_array1 else 'C'
    order2 = 'F' if is_fortran_array2 else 'C'
    # We need at least three shape entries, none being 1, especially to
    # test what happens if one appends a fortran to a non-fortran array
    # and vice versa.
    shape1 = (2,3,4)
    shape2 = (2,3,5) if is_fortran_array1 else (5,3,4)
    product1 = np.multiply.reduce(shape1)
    product2 = np.multiply.reduce(shape2)

    arr1 = np.arange(product1, dtype=dtype1).reshape(shape1, order=order1)
    arr2 = np.arange(
        product1, product1 + product2, dtype=dtype2
    ).reshape(shape2, order=order2)

    arr2_append_count = 10
    threads = []

    if use_np_save:
        np.save(tmpfile, arr1)

    with NpyAppendArray(
        tmpfile, delete_if_exists=delete_if_exists,
        rewrite_header_on_append=rewrite_header_on_append
    ) as npaa:
        if delete_if_exists or not use_np_save:
            npaa.append(arr1)

        def task():
            npaa.append(arr2)

        for i in range(arr2_append_count):
            thread = threading.Thread(target=task)
            threads += [thread]
            thread.start()
    
        # make sure to join threads within the "with NpyAppendArray ..."
        for thread in threads:
            thread.join()

    arr = np.load(tmpfile)
    arr_ref = np.concatenate(
        [arr1, *[arr2]*arr2_append_count],
        axis = -1 if is_fortran_array1 else 0,
        dtype = arr1.dtype
    )

    assert np.all(arr == arr_ref)

    tmpfile.unlink(missing_ok=True)

for i in range(40):
    with NpyAppendArray(tmpfile) as npaa:
        npaa.append(np.zeros((50000, 76, 3)))

tmpfile.unlink(missing_ok=True)