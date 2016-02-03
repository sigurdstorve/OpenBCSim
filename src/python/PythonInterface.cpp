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

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <iostream>
#include <boost/python.hpp>
#include <boost/multi_array.hpp>
#include "numpy_boost.hpp"
#include "numpy_boost_python.hpp"
#include <vector>
#include <string>
#include <memory>
#include "../core/BCSimConfig.hpp"
#include "../core/BeamProfile.hpp"
#include "../core/to_string.hpp"
#include "../core/LibBCSim.hpp"

using namespace bcsim;

// Note: Work-in-progress

// Return std::vector with dimensions of numpy_vector. Length of this vector
// gives the rank. 
template <typename T, int Ndims>
std::vector<size_t> get_dimensions(numpy_boost<T, Ndims> v) {
    size_t num_dims = v.num_dimensions();
    const size_t* dims = v.shape();
    std::vector<size_t> res(dims, dims + num_dims);
    return res;
}

class RfSimulatorWrapper {
public:
    RfSimulatorWrapper(std::string sim_type) {
        if (m_print_debug) std::cout << "Creating simulator of type " << sim_type << std::endl;
        m_rf_simulator = Create(sim_type);
    }

    void set_print_debug(bool val) {
        m_print_debug = val;
    }
    
    void set_parameter(const std::string& key, const std::string& value) {
        m_rf_simulator->set_parameter(key, value);
    }
    
    void clear_fixed_scatterers() {
        m_rf_simulator->clear_fixed_scatterers();
    }

    void add_fixed_scatterers(numpy_boost<float, 2> data) {
        auto dimensions = get_dimensions(data);
        size_t numScatterers = dimensions[0];
        size_t numColumns = dimensions[1];
        
        if (numColumns != 4) {
            throw std::runtime_error(std::string(__FUNCTION__) + " : Number of columns must be four");
        }
        auto new_scatterers = FixedScatterers::s_ptr(new FixedScatterers);

        for (size_t row = 0; row < numScatterers; row++) {
            PointScatterer ps;
            ps.pos = vector3(data[row][0], data[row][1], data[row][2]);
            ps.amplitude = data[row][3];
            new_scatterers->scatterers.push_back(ps);
        }
        m_rf_simulator->add_fixed_scatterers(new_scatterers);
    }

    void clear_spline_scatterers() {
        m_rf_simulator->clear_spline_scatterers();
    }

    void add_spline_scatterers(int spline_degree,
                               numpy_boost<float, 1> knot_vector,
                               numpy_boost<float, 3> control_points,
                               numpy_boost<float, 1> amplitudes) {

        auto new_scatterers = SplineScatterers::s_ptr(new SplineScatterers);
        
        // HACK: for now just fill in member variables from the outside
        new_scatterers->spline_degree = spline_degree;
        auto knot_vector_dims = get_dimensions(knot_vector);
        if (knot_vector_dims.size() != 1) {
            throw std::runtime_error(std::string(__FUNCTION__) + " : invalid knot vector rank");
        }
        int num_knots = knot_vector_dims.at(0);
        new_scatterers->knot_vector.resize(num_knots);
        for (int knot_i = 0; knot_i < num_knots; knot_i++) {
            new_scatterers->knot_vector[knot_i] = knot_vector[knot_i];
        }
        
        // Get size for all three dimensions of nodes array
        auto control_points_dims = get_dimensions(control_points);
        if (control_points_dims.size() != 3) {
            throw std::runtime_error(std::string(__FUNCTION__) + " : invalid control_points rank");
        }
        int num_scatterers = control_points_dims[0];
        int num_control_points = control_points_dims[1];
        
        // Sanity check
        if (num_knots != (num_control_points + spline_degree + 1)) {
            throw std::runtime_error(std::string(__FUNCTION__) + " : mismatch in number of nodes, degree and control points");
        }
        
        std::cout << "Number of scatterers: " << num_scatterers << std::endl;
        std::cout << "Number of control points for each scatterer: " << num_control_points << std::endl;
        
        if (control_points_dims[2] != 3) {
            throw std::runtime_error(std::string(__FUNCTION__) + " : size of dimension 3 must be three (x,y,z)");
        }

        const auto amplitudes_size = get_dimensions(amplitudes);
        if (amplitudes_size[0] != num_scatterers) {
            throw std::runtime_error("Mismatch between control_points and amplitudes");
        }
                
        new_scatterers->control_points.resize(num_scatterers);
        new_scatterers->amplitudes.resize(num_scatterers);

        for (int scatterer_i = 0; scatterer_i < num_scatterers; scatterer_i++) {
            new_scatterers->amplitudes[scatterer_i] = amplitudes[scatterer_i];

            new_scatterers->control_points[scatterer_i].resize(num_control_points);
            for (int control_point_i = 0; control_point_i < num_control_points; control_point_i++) {
               
                const vector3 pos(control_points[scatterer_i][control_point_i][0],
                                  control_points[scatterer_i][control_point_i][1],
                                  control_points[scatterer_i][control_point_i][2]);
                
                new_scatterers->control_points[scatterer_i][control_point_i] = pos;
            }
        }

        m_rf_simulator->add_spline_scatterers(new_scatterers);
    }

    void set_scan_sequence(numpy_boost<float, 2> origins,
                           numpy_boost<float, 2> directions,
                           float line_length,
                           numpy_boost<float, 2> lateralDirs,
                           numpy_boost<float, 1> timestamps) {

        auto originsDims = get_dimensions(origins);
        auto directionsDims = get_dimensions(directions);
        auto lateralDirsDims = get_dimensions(lateralDirs);
        auto timestamps_dims = get_dimensions(timestamps);
        size_t numLines = originsDims[0]; // rows: lineNo, cols: components

        // Do some sanity checks
        if (originsDims.size() != 2 || directionsDims.size() != 2
            || lateralDirsDims.size() != 2 || timestamps_dims.size() != 1) {
            throw std::runtime_error("set_scan_sequence(): Invalid rank.");
        }
        if (numLines != directionsDims[0] || numLines != lateralDirsDims[0]) {
            throw std::runtime_error("set_scan_sequence(): Mismatch in number of rows.");    
        }
        
        // Build a ScanSequence object and configure simulator object.
        auto seq = ScanSequence::s_ptr(new ScanSequence(line_length));
        for (size_t lineNo = 0; lineNo < numLines; lineNo++) {
            const vector3 origin(origins[lineNo][0], origins[lineNo][1], origins[lineNo][2]);
            const vector3 dir(directions[lineNo][0], directions[lineNo][1], directions[lineNo][2]);
            const vector3 lateral_dir(lateralDirs[lineNo][0], lateralDirs[lineNo][1], lateralDirs[lineNo][2]);
            const float timestamp = timestamps[lineNo];
            const Scanline sl(origin, dir, lateral_dir, timestamp);
            seq->add_scanline(sl);
        }
        m_rf_simulator->set_scan_sequence(seq);
    }

    void set_excitation(numpy_boost<float, 1> samples, int center_index, float fs, float demod_freq) {
        auto samplesDims = get_dimensions(samples);
        if (samplesDims.size() != 1) {
            throw std::runtime_error(std::string(__FUNCTION__) + ": samples should be one-dimensional");
        }
        size_t numSamples = samplesDims[0];

        std::vector<float> s(numSamples);
        for (size_t i = 0; i < numSamples; i++) {
            s[i] = samples[i];
        }
        ExcitationSignal ex;
        ex.samples = s;
        ex.center_index = center_index;
        ex.sampling_frequency = fs;
        ex.demod_freq = demod_freq;
        m_rf_simulator->set_excitation(ex);
        if (m_print_debug) {
            std::cout << to_string(ex) << std::endl;
        }
        
    }
    
    void set_analytical_beam_profile(float sigmaLateral, float sigmaElevational) {
        // TODO: Plug memleak
        auto beam_profile = IBeamProfile::s_ptr(new GaussianBeamProfile(sigmaLateral, sigmaElevational));
        m_rf_simulator->set_analytical_profile(beam_profile);
        if (m_print_debug) {
            std::cout << "Lateral sigma is now " << sigmaLateral << " [m]" << std::endl;
            std::cout << "Elevational sigma is now " << sigmaElevational << " [m]" << std::endl;
        }
    }
    
    void set_lut_beam_profile(float rMin, float rMax, float lMin, float lMax, float eMin, float eMax,
                              numpy_boost<float, 3> samples) {
       
        // TODO: use smart-pointers to avoid memory leak.
        auto dims = get_dimensions(samples);
        size_t numDims = dims.size();
        if (numDims != 3) {
            // TODO: Allow 2d (circularly symmetric)?
            throw std::runtime_error("Number of sample dimensions must be three.");
        }
        if (m_print_debug) {
            std::cout << "rMin = " << rMin << ", rMax = " << rMax << std::endl;
            std::cout << "lMin = " << lMin << ", lMax = " << lMax << std::endl;
            std::cout << "eMin = " << eMin << ", eMax = " << eMax << std::endl;
            std::cout << "Number of sample dimensions: " << numDims << std::endl;
        }
        
        size_t numSamplesRad = dims[0];
        size_t numSamplesLat = dims[1];
        size_t numSamplesEle = dims[2];
        
	
        if (m_print_debug) {
            std::cout << "Number of radial samples: " << numSamplesRad << std::endl;
            std::cout << "Number of lateral samples: " << numSamplesLat << std::endl;
            std::cout << "Number of elevational samples: " << numSamplesEle << std::endl;
        }

        // Must temporarily work with a pointer of type LUTBeamProfile, since otherwise
        // its specific functionality will be unavailable as a BeamProfile pointer...
        Interval rangeRange(rMin, rMax);
        Interval lateralRange(lMin, lMax);
        Interval elevationalRange(eMin, eMax);
        auto lut = new LUTBeamProfile(numSamplesRad, numSamplesLat, numSamplesEle,
                                      rangeRange, lateralRange, elevationalRange);

        // Copy all samples from numpy array.
        for (int ir=0; ir < numSamplesRad; ir++) {
            for (int il=0; il < numSamplesLat; il++) {
                for (int ie=0; ie < numSamplesEle; ie++) {
                    lut->setDiscreteSample(ir, il, ie, samples[ir][il][ie]);       
                }
            }
        }
        m_rf_simulator->set_lookup_profile(IBeamProfile::s_ptr(lut));
    }

    PyObject* simulate_lines() {
        // Simulate
        std::vector<std::vector<std::complex<float>> > rf_lines;
        m_rf_simulator->simulate_lines(rf_lines);
        int num_rf_lines = rf_lines.size();
        // all lines have same number of samples
        int num_samples = rf_lines[0].size();
        
        // Copy over to a NumPy array.
        int array_dims[] = {static_cast<int>(num_samples), static_cast<int>(num_rf_lines)};
        numpy_boost<std::complex<float>, 2> array(array_dims);
        
        for (int sample_no = 0; sample_no < num_samples; sample_no++) {
            for (int line_no = 0; line_no < num_rf_lines; line_no++) {
                array[sample_no][line_no] = rf_lines[line_no][sample_no];
            }
        }

        // Return it as a PyObject
        PyObject* array_object = array.py_ptr();
        Py_INCREF(array_object);
        return array_object;
    }

    boost::python::list get_debug_data(const std::string& identifier) {
        const auto temp = m_rf_simulator->get_debug_data(identifier);
        boost::python::list res;
        for (const auto value : temp) {
            res.append(value);
        }
        return res;
    }

protected:
    IAlgorithm::s_ptr       m_rf_simulator;
    bool                    m_print_debug;

};

BOOST_PYTHON_MODULE(pyrfsim) {
    using namespace boost::python;

    import_array();

    numpy_boost_python_register_type<float, 1>();
    numpy_boost_python_register_type<float, 2>();
    numpy_boost_python_register_type<float, 3>();
    numpy_boost_python_register_type<std::complex<float>, 2>();

    class_<RfSimulatorWrapper>("RfSimulator", init<std::string>())
        .def("set_print_debug",             &RfSimulatorWrapper::set_print_debug)
        .def("set_parameter",               &RfSimulatorWrapper::set_parameter)
        .def("clear_fixed_scatterers",      &RfSimulatorWrapper::clear_fixed_scatterers)
        .def("add_fixed_scatterers",        &RfSimulatorWrapper::add_fixed_scatterers)
        .def("clear_spline_scatterers",     &RfSimulatorWrapper::clear_spline_scatterers)
        .def("add_spline_scatterers",       &RfSimulatorWrapper::add_spline_scatterers)
        .def("set_scan_sequence",           &RfSimulatorWrapper::set_scan_sequence)
        .def("set_excitation",              &RfSimulatorWrapper::set_excitation)
        .def("set_analytical_beam_profile", &RfSimulatorWrapper::set_analytical_beam_profile)
        .def("set_lut_beam_profile",        &RfSimulatorWrapper::set_lut_beam_profile)
        .def("simulate_lines",              &RfSimulatorWrapper::simulate_lines)
        .def("get_debug_data",              &RfSimulatorWrapper::get_debug_data)
    ;
}
