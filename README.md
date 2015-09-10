# OpenBCSim
This project is a fast C++/CUDA open-source implementation of an ultrasound simulator based on the COLE algorithm as published by Gao et al. in "A fast convolution-based methodology to simulate 
2-D/3-D cardiac ultrasound images.", IEEE TUFFC 2009.

The algorithm has been extended to optionally use B-splines for representing dynamic point scatterers. Features:
- Python script for generating point-scatterer phantoms
- Fixed and dynamic (B-spline based) point scatterers
- Multicore CPU implementation (OpenMP based)
- GPU implementation (using NVIDIA CUDA)
- Python interface using Boost.Python and numpy-boost
- Qt5-based interactive GUI from-end

This code is still experimental.