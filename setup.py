import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="npy-append-array",
    version="0.9.11",
    author="Michael Siebert",
    author_email="michael.siebert2k@gmail.com",
    description="Create Numpy NPY files that are larger than the main memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xor2k/npy-append-array",
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
