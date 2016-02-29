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

#include <stdexcept>
#include <cmath>
#include "Cartesianator.hpp"

template <typename T>
CpuCartesianator<T>::CpuCartesianator() 
    : m_num_samples_x(512),
      m_num_samples_y(512),
      m_geometry(nullptr)
{
    UpdateOutputBuffer();           
}

template <typename T>
void CpuCartesianator<T>::SetGeometry(bcsim::ScanGeometry::ptr geometry) {
    m_geometry = geometry;
    geometry->get_xy_extent(m_x_min, m_x_max, m_y_min, m_y_max);
}

template <typename T>
const T* CpuCartesianator<T>::GetOutputBuffer() {
    return m_output_buffer.data();
}

template <typename T>
void CpuCartesianator<T>::SetOutputSize(size_t num_samples_x,
                                        size_t num_samples_y) {
    m_num_samples_x = num_samples_x;
    m_num_samples_y = num_samples_y;
    UpdateOutputBuffer();
}

template <typename T>
void CpuCartesianator<T>::GetOutputSize(size_t& num_samples_x,
                                        size_t& num_samples_y) {
    num_samples_x = m_num_samples_x;
    num_samples_y = m_num_samples_y;
}

template <typename T>
void CpuCartesianator<T>::Process(T* in_buffer, int num_beams, int num_samples) {
    if (!m_geometry) {
        throw std::runtime_error("geometry not configured");
    }
    auto sector_geo = std::dynamic_pointer_cast<bcsim::SectorScanGeometry>(m_geometry);
    auto linear_geo = std::dynamic_pointer_cast<bcsim::LinearScanGeometry>(m_geometry);
    if (sector_geo) {
        DoSectorTransform(in_buffer, num_beams, num_samples, sector_geo);
    } else if (linear_geo) {
        DoLinearTransform(in_buffer, num_beams, num_samples, linear_geo);
    } else {
        throw std::runtime_error("unknown geometry");
    }
}

template <typename T>
void CpuCartesianator<T>::UpdateOutputBuffer() {
    auto num_samples = m_num_samples_x*m_num_samples_y;
    m_output_buffer.resize(num_samples);
}

template <typename T>
void CpuCartesianator<T>::DoSectorTransform(T* in_buffer, int num_beams, int num_range,
                                            std::shared_ptr<bcsim::SectorScanGeometry> geometry) {

    // deltas for cartesian grid
    const auto dx = (m_x_max - m_x_min) / (m_num_samples_x-1);
    const auto dy = (m_y_max - m_y_min) / (m_num_samples_y-1);
        
    const auto range_min = 0.0f;
    const auto range_max = geometry->depth;
        
    float theta_min, theta_max;
    geometry->get_angle_limits(theta_min, theta_max);
        
    // deltas for polar grid
    const auto dr = (range_max - range_min) / (num_range-1);
    const auto dt = (theta_max - theta_min) / (num_beams-1);

    // For each output sample determine index of closest beamspace sample
    for (size_t xi = 0; xi < m_num_samples_x; xi++) {
        for (size_t yi = 0; yi < m_num_samples_y; yi++) {
            float x = m_x_min + xi*dx;
            float y = m_y_min + yi*dy;

            // Map (x, y) to (r, theta)
            float r = std::sqrt(x*x+y*y);
            float theta;
            if (std::abs(x) > 1e-6) {
                theta = std::atan2(y, x);
            } else {
                // Avoid division by zero. TODO: Can this be solved more elegant?
                theta = static_cast<float>(std::atan(1)*2);
            }

            float value = 0.0f;

            int r_idx0 = static_cast<int>(std::floor( (r - range_min) / dr ));
            int t_idx0 = static_cast<int>(std::floor( (theta - theta_min) / dt));
            int r_idx1 = r_idx0 + 1;
            int t_idx1 = t_idx0 + 1;

            float sample00 = (r_idx0 >= 0 && r_idx0 < num_range && t_idx0 >= 0 && t_idx0 < num_beams) ? in_buffer[num_range*t_idx0 + r_idx0] : 0.0f;
            float sample01 = (r_idx0 >= 0 && r_idx0 < num_range && t_idx1 >= 0 && t_idx1 < num_beams) ? in_buffer[num_range*t_idx1 + r_idx0] : 0.0f;
            float sample10 = (r_idx1 >= 0 && r_idx1 < num_range && t_idx0 >= 0 && t_idx0 < num_beams) ? in_buffer[num_range*t_idx0 + r_idx1] : 0.0f;
            float sample11 = (r_idx1 >= 0 && r_idx1 < num_range && t_idx1 >= 0 && t_idx1 < num_beams) ? in_buffer[num_range*t_idx1 + r_idx1] : 0.0f;
            float range0 = range_min + r_idx0*dr;
            float range1 = range0 + dr;
            float theta0 = theta_min + t_idx0*dt;
            float theta1 = theta0 + dt;

            // Use Wikipedia formula
            value = 1.0f/((range1-range0)*(theta1-theta0))*( sample00*(range1-r)*(theta1-theta)
                                                            +sample10*(r-range0)*(theta1-theta)
                                                            +sample01*(range1-r)*(theta-theta0)
                                                            +sample11*(r-range0)*(theta-theta0));


            size_t res_offset = static_cast<size_t>(m_num_samples_x*yi + xi);
            m_output_buffer[res_offset] = static_cast<T>(value);
        }
    }
}

template <typename T>
void CpuCartesianator<T>::DoLinearTransform(T* in_buffer, int num_beams, int num_range,
                                            std::shared_ptr<bcsim::LinearScanGeometry> geometry) {
    // deltas for cartesian grid
    const auto dx = (m_x_max - m_x_min) / (m_num_samples_x-1);
    const auto dy = (m_y_max - m_y_min) / (m_num_samples_y-1);

    // beam-space limits. TODO: Get from geometry's function.
    const float x_min = -0.5f*geometry->width;
    const float x_max = 0.5f*geometry->width;
    const float y_min = 0.0f;
    const float y_max = geometry->range_max;

    // beamspace deltas
    const float bs_dx = (x_max-x_min) / (num_beams-1);
    const float bs_dy = (y_max-y_min) / (num_range-1);

    // for each output sample determine index of closest beamspace sample
    for (size_t xi = 0; xi < m_num_samples_x; xi++) {
        for (size_t yi = 0; yi < m_num_samples_y; yi++) {
            float x = m_x_min + xi*dx;
            float y = m_y_min + yi*dy;

            // map to inds
            int x_idx0 = static_cast<int>(std::floor( (x - x_min) / bs_dx ));
            int y_idx0 = static_cast<int>(std::floor( (y - y_min) / bs_dy ));
            int x_idx1 = x_idx0 + 1;
            int y_idx1 = y_idx0 + 1;

            float value = 0.0f;


            float sample00 = (y_idx0 >= 0 && y_idx0 < num_range && x_idx0 >= 0 && x_idx0 < num_beams) ? in_buffer[num_range*x_idx0 + y_idx0] : 0.0f;
            float sample01 = (y_idx0 >= 0 && y_idx0 < num_range && x_idx1 >= 0 && x_idx1 < num_beams) ? in_buffer[num_range*x_idx1 + y_idx0] : 0.0f;
            float sample10 = (y_idx1 >= 0 && y_idx1 < num_range && x_idx0 >= 0 && x_idx0 < num_beams) ? in_buffer[num_range*x_idx0 + y_idx1] : 0.0f;
            float sample11 = (y_idx1 >= 0 && y_idx1 < num_range && x_idx1 >= 0 && x_idx1 < num_beams) ? in_buffer[num_range*x_idx1 + y_idx1] : 0.0f;
            float y0 = y_min + y_idx0*bs_dy;
            float y1 = y0 + bs_dy;
            float x0 = x_min + x_idx0*bs_dx;
            float x1 = x0 + bs_dx;

            // Use Wikipedia formula
            value = 1.0f/((y1-y0)*(x1-x0))*( sample00*(y1-y)*(x1-x)
                                            +sample10*(y-y0)*(x1-x)
                                            +sample01*(y1-y)*(x-x0)
                                            +sample11*(y-y0)*(x-x0));

            size_t res_offset = static_cast<size_t>(m_num_samples_x*yi + xi);
            m_output_buffer[res_offset] = static_cast<T>(value);
        }
    }
}

// explicit instantiations
template class ICartesianator<unsigned char>;
template class CpuCartesianator<unsigned char>;
