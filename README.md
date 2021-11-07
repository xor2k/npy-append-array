# NpyAppendArray

Create Numpy NPY files by appending on the zero axis. Application examples:
1. Binary log files which can be processed without parsing
2. Efficiently create arrays which are larger than the main memory
    1. Embedded devices might have limited memory
    2. Certain workflows (e.g. Deep Learning) may require to handle large 
    amounts of data

After creation, the file can then be read with memory mapping, e.g. by adding
`mmap_mode="r"`.

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

# Remove out.npy and use this dtype and np.save below to provoke an error
# dtype=[('____a_not_uncommonly_long_descriptor_la_la_la_la', int)]

dtype = int

arr1 = np.array([[1,2],[3,4]], dtype=dtype)
arr2 = np.array([[1,2],[3,4],[5,6]], dtype=dtype)

filename = 'out.npy'

# Appending to an array created by np.save might be possible under certain
# circumstances, since the .npy total header byte count is required to be evenly
# divisible by 64. Thus, there might be some spare space to grow the shape entry
# in the array descriptor. However, this is not guaranteed and might randomly
# fail. Initialize the array with NpyAppendArray(filename) directly (see below)
# so the header will be created with 64 byte of spare header space for growth.
# Will this be enough? It allows for up to 10^64 >= 2^212 array entries or data
# bits. Indeed, this is less than the number of atoms in the universe. However,
# fully populating such an array, due to limits imposed by quantum mechanics,
# would require more energy than would be needed to boil the oceans, compare
# https://hbfs.wordpress.com/2009/02/10/to-boil-the-oceans
# Therefore, a wide range of use cases should be coverable with this approach.

# np.save(filename, arr1)

with NpyAppendArray(filename) as npaa:
    npaa.append(arr1)
    npaa.append(arr2)
    npaa.append(arr2)
    
data = np.load(filename, mmap_mode="r")

print(data)
```

## Limitations

1. Only tested with Linux. For Windows, consider using WSL (version 2 or above).
2. NotImplementedError thrown when Fortran order is used.
3. NPY version 3 is unsupported because there is no
  `numpy.lib.format.read_array_header_3_0` function defined in
  https://numpy.org/devdocs/reference/generated/numpy.lib.format.html