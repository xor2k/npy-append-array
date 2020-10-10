# NpyAppendArray

Allows to create Numpy `.npy` files that are larger than the main memory of the
machine by appending on the zero axis. The file can then be read with
`mmap_mode="r"`.

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

# optional, .append will create file automatically if not exists
np.save(filename, arr1)

npaa = NpyAppendArray(filename)
npaa.append(arr2)
npaa.append(arr2)
npaa.append(arr2)

data = np.load(filename, mmap_mode="r")

print(data)
```