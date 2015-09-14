# OpenBCSim
This project is a fast C++/CUDA open-source implementation of an ultrasound simulator based on the COLE algorithm as published by Gao et al. in "A fast convolution-based methodology to simulate 
2-D/3-D cardiac ultrasound images.", IEEE TUFFC 2009.

The algorithm has been extended to optionally use B-splines for representing dynamic point scatterers.
#Features:
- Python scripts for generating point-scatterer phantoms
- Supports both fixed and dynamic (B-spline based) point scatterers
- Multicore CPU implementation (OpenMP based)
- GPU implementation (using NVIDIA CUDA)
- Python interface using Boost.Python and numpy-boost
- Qt5-based interactive GUI front-end
- Output data type can be radiofrequency (RF) or envelope-detected RF
- Cross-platform code. Successfully built on Linux (Ubuntu 15.04) and Windows 7

This code is still experimental. More documentation, examples, and instructions on how to compile the code will be added soon.
