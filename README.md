# NpyAppendArray

Create Numpy NPY files by appending on the zero axis. The main application is to
efficiently create arrays which are larger than the main memory:
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

## Implementation Details
Appending to an array created by np.save might be possible under certain
circumstances, since the .npy total header byte count is required to be evenly
divisible by 64. Thus, there might be some spare space to grow the shape entry
in the array descriptor. However, this is not guaranteed and might randomly
fail. Initialize the array with NpyAppendArray(filename) directly (see above) so
the header will be created with 64 byte of spare header space for growth.

### Will 64 byte extra header space cover my needs?
It allows for up to 10^64 >= 2^212 array entries or data bits. Indeed, this is
less than the number of atoms in the universe. However, fully populating such an
array, due to limits imposed by quantum mechanics, would require more energy
than would be needed to boil the oceans, compare

https://hbfs.wordpress.com/2009/02/10/to-boil-the-oceans

Therefore, the extra header space might cover your needs.

## Limitations

1. Only tested with Linux. For Windows, consider using WSL (version 2 or above).
2. NotImplementedError thrown when Fortran order is used.
3. NPY version 3 is unsupported because there is no
  `numpy.lib.format.read_array_header_3_0` function defined in
  https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
4. Just like with numpy.load/numpy.save, multithreaded read/write is unsupported