/*
Copyright (c) 2015, Sigurd Storve
All rights reserved.

Licensed under the BSD license.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#include <stdexcept>
#include <memory>
#include <string>
#include <vector>
#include "export_macros.hpp"
#include "bcsim_defines.h"
#include "vector3.hpp"

namespace bcsim {

// Define the precision level to work at (32- or 64-bit floats)
// NOTE: currently only float should be used.
#if BCSIM_USE_DOUBLE_PRECISION
    #define bc_float double
#else
    #define bc_float float
#endif
typedef Vector3D<bc_float> vector3;


struct Interval {
public:
    Interval(float t0, float t1) : first(t0), last(t1) { }
    float first;
    float last;
};

// Radiofrequency excitation signal.
struct ExcitationSignal {
    // Discrete excitation signal samples
    std::vector<bc_float> samples;
    
    // The index of vector "samples" for which time is zero,
    // i.e. the center of the pulse.
    int center_index;                
    
    // The sampling frequency.
    float sampling_frequency;

    // Downmixing frequency to use for IQ data
    float demod_freq;
};

// Description of a single point scatterer.
struct PointScatterer {
    // Position in space
    vector3 pos;
    
    // Scattering strength
    float amplitude;
};

// Common parent for all types of scatterer configurations.
struct Scatterers {
    typedef std::shared_ptr<Scatterers> s_ptr;
    
    virtual int num_scatterers() const = 0;
};

// Stationary scatterers.
struct FixedScatterers : public Scatterers {

    typedef std::shared_ptr<FixedScatterers> s_ptr;    
        
    virtual int num_scatterers() const {
        return static_cast<int>( scatterers.size() );
    }

    std::vector<PointScatterer> scatterers;
};

// Scatterers follow trajectory described by splines.
// All splines have the same degree and are defined on the same
// knot vector to save memory.
struct SplineScatterers : public Scatterers {

    typedef std::shared_ptr<SplineScatterers> s_ptr;

    virtual int num_scatterers() const {
        // assert(control_points.size() == amplitudes.size());
        return static_cast<int>(control_points.size());
    }

    // Returns the number of control points for each spline
    // scatterer (same for all scatterers).
    size_t get_num_control_points() const {
        if (num_scatterers() == 0) {
            throw std::runtime_error("No scatterers in dataset");
        }
        return control_points[0].size();
    }
    
    // Spline degree and knot vector are common for
    // all point scatterers.
    int                         spline_degree;
    std::vector<bc_float>       knot_vector;
    
    // For each scatterer: list of control points in space and a scalar amplitude
    // Indexed by scatterer no.
    std::vector<std::vector<vector3>> control_points;
    std::vector<bc_float>             amplitudes;
};



}   // namespace
