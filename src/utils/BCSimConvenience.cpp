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

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <stdexcept>
#include "BCSimConvenience.hpp"
#include "../core/bspline.hpp"
#include "rotation3d.hpp"

namespace bcsim {

std::vector<std::vector<float> > decimate_frame(const std::vector<std::vector<float> >& frame, int rad_decimation) {
    if (rad_decimation < 1) throw std::runtime_error("Invalid decimation value");

    auto num_beams = frame.size();
    auto num_samples = frame[0].size();
    auto num_samples_dec = num_samples / rad_decimation;
    
    std::vector<std::vector<float> > decimated_frame(num_beams);
    for (size_t beam_no = 0; beam_no < num_beams; beam_no++) {
        decimated_frame[beam_no].resize(num_samples_dec);
        for (size_t sample_no = 0; sample_no < num_samples_dec; sample_no++) {
            decimated_frame[beam_no][sample_no] = frame[beam_no][sample_no*rad_decimation];
        }
    }
    return decimated_frame;
}

float get_max_value(const std::vector<std::vector<float> >& image_lines) {
    std::vector<float> max_values;
    for (const auto& image_line : image_lines) {
        max_values.push_back(*std::max_element(image_line.begin(), image_line.end()));
    }
    return *std::max_element(max_values.begin(), max_values.end());
}

void log_compress_frame(std::vector<std::vector<float> >& image_lines, float dyn_range, float normalize_factor, float gain_factor) {

    auto num_beams   = image_lines.size();
    auto num_samples = image_lines[0].size(); 

    for (auto& beam : image_lines) {
        std::transform(beam.begin(), beam.end(), beam.begin(), [=](float pixel) {
            // log-compression
            pixel = static_cast<float>(20.0*std::log10(gain_factor*pixel/normalize_factor));
            pixel = (255.0/dyn_range)*(pixel + dyn_range);
            
            // clamp to [0, 255]
            if (pixel < 0.0f) pixel = 0.0f;
            if (pixel >= 255.0f) pixel = 255.0f;
            
            return pixel;
        });
    }
}

Scatterers::s_ptr render_fixed_scatterers(SplineScatterers::s_ptr spline_scatterers, float timestamp) {
    // TODO: can parts of this code be put in a separate function and used both
    // here and in the CPU spline algoritm to reduce code duplication?
    auto res = FixedScatterers::s_ptr(new FixedScatterers);
    const auto num_scatterers = spline_scatterers->num_scatterers();
    if (num_scatterers == 0) {
        throw std::runtime_error("No spline scatterers");
    }
    
        
    // precompute basis functions
    const auto num_cs = spline_scatterers->get_num_control_points();
    std::vector<float> basis_fn(num_cs);
    for (size_t i = 0; i < num_cs; i++) {
        basis_fn[i] = bspline_storve::bsplineBasis(i, spline_scatterers->spline_degree, timestamp, spline_scatterers->knot_vector);
    }
    
    // evaluate using cached basis functions
    res->scatterers.resize(num_scatterers);
    for (size_t spline_no = 0; spline_no < num_scatterers; spline_no++) {
        PointScatterer scatterer;
        scatterer.pos       = vector3(0.0f, 0.0f, 0.0f);
        scatterer.amplitude = spline_scatterers->amplitudes[spline_no];
        for (size_t i = 0; i < num_cs; i++) {
            scatterer.pos += spline_scatterers->control_points[spline_no][i]*basis_fn[i];
        }
        res->scatterers[spline_no] = scatterer;
    }

    return res;
}

ScanSequence CreateScanSequence(std::shared_ptr<SectorScanGeometry> geometry, size_t num_lines, float timestamp) {
    const auto line_length = geometry->depth;
    ScanSequence res(line_length);
    
    // Will be transformed by a rotation into lateral and radial unit vectors.
    // NOTE: Need to use double to avoid "not orthonormal" error (maybe the test is to strict for floats?)
    const auto unit_vector_x = unit_x<double>();
    const auto unit_vector_z = unit_z<double>();
    const vector3 origin(0.0f, 0.0f, 0.0f);

    for (int line_no = 0; line_no < static_cast<int>(num_lines); line_no++) {
        const float angle = -0.5f*geometry->width + geometry->tilt + line_no*geometry->width/(num_lines-1);

        const auto ROT_MATRIX = rotation_matrix_y<double>(angle);
        const auto temp_radial_direction  = boost::numeric::ublas::prod(ROT_MATRIX, unit_vector_z);
        const auto temp_lateral_direction = boost::numeric::ublas::prod(ROT_MATRIX, unit_vector_x); 
	
        // Copy to vector3 vectors. TODO: deduplicate
        const vector3 direction  ((float)temp_radial_direction(0),  (float)temp_radial_direction(1),  (float)temp_radial_direction(2));
        const vector3 lateral_dir((float)temp_lateral_direction(0), (float)temp_lateral_direction(1), (float)temp_lateral_direction(2));

        try {
            auto sl = Scanline(origin, direction, lateral_dir, timestamp);
            res.add_scanline(sl);
        } catch (std::runtime_error& e) {
            throw std::runtime_error(std::string("failed creating scan line: ") + e.what());
        }
        
    }
    return res;
}

ScanSequence CreateScanSequence(std::shared_ptr<LinearScanGeometry> geometry, size_t num_lines, float timestamp) {
    const auto line_length = geometry->range_max;
    ScanSequence res(line_length);

    const auto unit_vector_x = unit_x<double>();
    const auto unit_vector_z = unit_z<double>();

    // Copy to vector3 vectors. TODO: deduplicate
    const vector3 direction  (unit_vector_z(0), unit_vector_z(1), unit_vector_z(2));
    const vector3 lateral_dir(unit_vector_x(0), unit_vector_x(1), unit_vector_x(2));


    for (int line_no = 0; line_no < static_cast<int>(num_lines); line_no++) {
        try {
            const vector3 scanline_origin(-0.5f*geometry->width + line_no*geometry->width/(static_cast<int>(num_lines)-1), 0.0f, 0.0f);
            auto sl = Scanline(scanline_origin, direction, lateral_dir, timestamp);

            res.add_scanline(sl);
        } catch (std::runtime_error& e) {
            throw std::runtime_error(std::string("failed creating scan line: ") + e.what());
        }
    }
    
    return res;
}

// probe_origin is position of probe's origin in world coordinate system.
ScanSequence CreateScanSequence(ScanGeometry::ptr geometry, size_t num_lines, float timestamp) {
    auto sector_geo = std::dynamic_pointer_cast<SectorScanGeometry>(geometry); 
    auto linear_geo = std::dynamic_pointer_cast<LinearScanGeometry>(geometry);
    if (sector_geo) {
        return CreateScanSequence(sector_geo, num_lines, timestamp);
    } else if (linear_geo) {
        return CreateScanSequence(linear_geo, num_lines, timestamp);
    } else {
        throw std::runtime_error("unable to cast scan geometry");
    }
}

namespace detail {
    // Apply a 3x3 rotation matrix to a vector3;
    template <typename T>
    vector3 TransformVector(const vector3& v, const boost::numeric::ublas::matrix<T>& matrix33) {
        boost::numeric::ublas::vector<T> temp_v(3);
        temp_v(0) = static_cast<T>(v.x); temp_v(1) = static_cast<T>(v.y); temp_v(2) = static_cast<T>(v.z);
        const auto transformed = boost::numeric::ublas::prod(matrix33, temp_v);
        return vector3(static_cast<float>(transformed(0)),
                       static_cast<float>(transformed(1)),
                       static_cast<float>(transformed(2)));
    }
}

ScanSequence::s_ptr OrientScanSequence(const ScanSequence& scan_seq, const vector3& rot_angles, const vector3& probe_origin) {
    const auto rot_matrix = rotation_matrix_xyz(rot_angles.x, rot_angles.y, rot_angles.z);

    const auto line_length = scan_seq.line_length;

    auto num_lines = scan_seq.get_num_lines();
    auto res = new ScanSequence(line_length);
    for (int i = 0; i < num_lines; i++) {
        const auto old_line = scan_seq.get_scanline(i);
        const auto rotated_origin      = detail::TransformVector(old_line.get_origin(),      rot_matrix);
        const auto rotated_direction   = detail::TransformVector(old_line.get_direction(),   rot_matrix);
        const auto rotated_lateral_dir = detail::TransformVector(old_line.get_lateral_dir(), rot_matrix);
        
        res->add_scanline(Scanline(rotated_origin + probe_origin,
                                  rotated_direction,
                                  rotated_lateral_dir,
                                  old_line.get_timestamp()));
    }
    return ScanSequence::s_ptr(res);
}

}   // end namespace
