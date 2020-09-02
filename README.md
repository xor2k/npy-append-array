# NpyAppendArray

Allows to create Numpy `.npy` files that are larger than the main memory of the machine by appending on the zero axis. The file can then be read with `mmap_mode="r"`, compare `demo.py`.