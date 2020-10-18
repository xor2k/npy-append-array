# NpyAppendArray

Create Numpy NPY files that are larger than the main memory of the machine by
appending on the zero axis. The file can then be read with `mmap_mode="r"`.

## Installation

```bash
pip install npy-append-array
```

## Usage

```python
from npy_append_array import NpyAppendArray
import numpy as np

arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[1,2],[3,4],[5,6]])

filename='out.npy'

# Appending to an array created by np.save is possible, but can fail in certain
# corner cases: e.g. a record as dtype, dim surpassing a critical threshold.
# Initialize the array with npaa.append directly (see below) so the
# header will be created with 64 byte of spare header space for growth.

# np.save(filename, arr1)

npaa = NpyAppendArray(filename)
npaa.append(arr1)
npaa.append(arr2)
npaa.append(arr2)

data = np.load(filename, mmap_mode="r")

print(data)
```

## Limitations

1. Only tested with Linux. For Windows, consider using WSL (version 2 or above).
2. Exception thrown when Fortran order is used.
3. NPY version 3 is unsupported because there is no
  `numpy.lib.format.read_array_header_3_0` function defined in
  https://numpy.org/devdocs/reference/generated/numpy.lib.format.html