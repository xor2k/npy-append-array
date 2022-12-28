# NpyAppendArray

Create Numpy `.npy` files by appending on the growth axis (0 for C order, -1
for Fortran order). It behaves like `numpy.concatenate` with the difference
that the result is stored out-of-memory in a `.npy` file and can be reused for
further appending. After creation, the file can then be read with memory
mapping (e.g. by adding `mmap_mode="r"`) which altogether allows to create and
read files (optionally) larger than the machine's main memory.

Some possible applications:
1. efficiently create large `.npy` (optionally database-like) files
   * Handling of offsets not included, can be done in an extra array
   * Large legacy files can be made appendable by calling `ensure_appendable`
       * can (optionally) be performed in-place to minimize disk space usage
2. create binary log files (optionally on low-memory embedded devices)
   * Check the option `rewrite_header_on_append=False` for extra efficiency
   * Binary log files can be accessed very efficiently without parsing
   * Incomplete files can be recovered efficiently by calling `recover`

Another feature of this library is the (above mentioned) `recover` function,
which makes incomplete `.npy` files readable by `numpy.load` again, no matter
whether they should be appended to or not.

Incomplete files can be the result of broken downloads or unfinished writes.
Recovery works by rewriting the header and inferring the growth axis (see
above) by the file size. As the data length may not be evenly divisible by the
non-append-axis shape, incomplete entries can either be ignored
(`zerofill_incomplete=False`), which probably makes sense in most scenarios.
Alternatively, to squeeze out the as much information from the file as
possible, `zerofill_incomplete=True` can be used, which fills the incomplete
last append axis item with zeros.

Raises `ValueError` instead of `TypeError` since version 0.9.14 to be more
consistent with Numpy.

NpyAppendArray can be used in multithreaded environments.

## Installation
```bash
conda install -c conda-forge npy-append-array
```
or
```bash
pip install npy-append-array
```
## Usage

```python
from npy_append_array import NpyAppendArray
import numpy as np

arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[1,2],[3,4],[5,6]])

filename = 'out.npy'

with NpyAppendArray(filename) as npaa:
    npaa.append(arr1)
    npaa.append(arr2)
    npaa.append(arr2)
    
data = np.load(filename, mmap_mode="r")

print(data)
```

## Concurrency
Concurrency can be achieved by multithreading: A single `NpyAppendArray`
object (per file) needs to be created. Then, `append` can be called from
multiple threads and locks will ensure that file writes do not happen in
parallel. When using with a `with` statement, make sure the `join` happens
within it, compare `test.py`.

Multithreaded writes are not the pinnacle of what is technically possible with
modern operating systems. It would be highly desirable to use `async` file
writes. However, although modules like `aiofile` exist, this is currently not
supported natively by Python or Numpy, compare

https://github.com/python/cpython/issues/76742

## Implementation Details
NpyAppendArray contains a modified, partial version of `format.py` from the
Numpy package. It ensures that array headers are created with 21
(`=len(str(8*2**64-1))`) bytes of spare space. This allows to fit an array of
maxed out dimensions (for a 64 bit machine) without increasing the array
header size. This allows to simply rewrite the header as we append data to the
end of the `.npy` file.

## Suppored Systems
Testes with Ubuntu Linux and macOS.