# Getting started with the simulator's Python interface
These notes explain how to build the Python interface as well as how to use it. This has been tested on Ubuntu 16.04.
Prerequisites: a Python installation w/NumPy (only version 2.7 has been tested)

## Building the simulator and the Python interface
In the CMake GUI make sure the following options are checked (ON)
- "BCSIM_BUILD_PYTHON_INTERFACE"
- "BCSIM_BUILD_UTILS"
- "BUILD_SHARED_LIBS" (needed because the Python module must be a shared library)

Press "Configure". Make sure that Python and HDF5 were detected correctly. You will then get a message "Invalid NumPy include path" because
the location of the NumPy headers must be specified. To do so, search the filesystem for the file ```numpy/arrayobject.h```
```
$ locate numpy/arrayobject.h
```
On my installation the full path was
```
/usr/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h
```
Then manually set ```NumPy_INCLUDE_DIR``` to the part before "numpy/arrayobject.h". In my case above that should be
```
NumPy_INCLUDE_DIR=/usr/lib/python2.7/dist-packages/numpy/core/include
```
Press "Configure" and then "Generate". Then compile and when finished, the file ```pyrfsim.so``` will be created. This file
is a Python module that can be loaded by ```import pyrfsim``` in a Python script. 

## Using one of the example Python scripts to test the Python interface
The folder "python" in the repository contains some scripts showing how to use the Python interface to the simulator.
There is also a script ```create_all_phantoms.py```, in the top-level folder, that will generate some point-scatterer phantoms that can be
used as target when simulating scans. This script should be run first and it will put several .h5 phantom files in
a folder called "generated_phantoms".

As an example, the script ```linear_scan_phantom.py``` can be used to do a linear scan in the XZ-plane as follows, on a
carotid plaque phantom.
```
python linear_scan_phantom.py ../generated_phantoms/carotid_plaque.h5 --visualize
```
For this to work, ensure that the file ```pyrfsim.so``` is in PYTHONPATH and also that the following Python packages
are installed: SciPy (used to generate RF pulse) and Matplotlib (used to visualize).


