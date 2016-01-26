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

#include "BeamProfile.hpp"

namespace bcsim {

GaussianBeamProfile::GaussianBeamProfile(float sigma_lateral, float sigma_elevational) {
    setSigmaLateral(sigma_lateral);
    setSigmaElevational(sigma_elevational);
}

void GaussianBeamProfile::setSigmaLateral(float new_sigma_lateral) {
    m_sigma_lateral = new_sigma_lateral;
    updateCaching();
}
    
void GaussianBeamProfile::setSigmaElevational(float new_sigma_elevational) {
    m_sigma_elevational = new_sigma_elevational;
    updateCaching();
}
    
float GaussianBeamProfile::sampleProfile(float r, float l, float e) {
    return std::exp(-(l*l/m_two_sigma_lateral_squared + e*e/m_two_sigma_elevational_squared));
}
    
void GaussianBeamProfile::updateCaching() {
    m_two_sigma_lateral_squared = 2*m_sigma_lateral*m_sigma_lateral;
    m_two_sigma_elevational_squared = 2*m_sigma_elevational*m_sigma_elevational;
}

LUTBeamProfile::LUTBeamProfile(int num_samples_rad, int num_samples_lat, int num_samples_ele,
                               Interval range_range, Interval lateral_range, Interval elevational_range) :
    m_num_samples_rad(num_samples_rad), m_num_samples_lat(num_samples_lat), m_num_samples_ele(num_samples_ele),
    m_range_range(range_range), m_lateral_range(lateral_range), m_elevational_range(elevational_range) {

    // sanity check
    if (num_samples_rad <= 1) throw std::runtime_error("Too few radial samples");
    if (num_samples_lat <= 1) throw std::runtime_error("Too few lateral samples");
    if (num_samples_ele <= 1) throw std::runtime_error("Too few elevational samples");

    // Allocate memory
    long num_samples = m_num_samples_rad*m_num_samples_lat*m_num_samples_ele;
    m_samples.resize(num_samples);
        
    // Compute sample deltas in all dimensions
    m_dr = (m_range_range.last - m_range_range.first) / (m_num_samples_rad-1);
    m_dl = (m_lateral_range.last - m_lateral_range.first) / (m_num_samples_lat-1);
    m_de = (m_elevational_range.last - m_elevational_range.first) / (m_num_samples_ele-1);
        
}

float LUTBeamProfile::sampleProfile(float r, float l, float e) {
    // map to indices
    const auto temp_r = (r-m_range_range.first) / m_dr;
    const auto temp_l = (l-m_lateral_range.first) / m_dl;
    const auto temp_e = (e-m_elevational_range.first) / m_de; 

    // dim0: radial, dim1: lateral, dim2: elevational

    const auto r0 = static_cast<int>(temp_r); const auto r1 = static_cast<int>(temp_r+1);
    const auto l0 = static_cast<int>(temp_l); const auto l1 = static_cast<int>(temp_l+1);
    const auto e0 = static_cast<int>(temp_e); const auto e1 = static_cast<int>(temp_e+1);
    
    // fractional parts
    const auto fractional_r = static_cast<float>(temp_r - r0);
    const auto fractional_l = static_cast<float>(temp_l - l0);
    const auto fractional_e = static_cast<float>(temp_e - e0);

    // return zeros outside
    if ((r0 < 0) || (r0 >= m_num_samples_rad) || (r1 < 0) || (r1 >= m_num_samples_rad)) {
        return 0.0;
    }
    if ((l0 < 0) || (l0 >= m_num_samples_lat) || (l1 < 0) || (l1 >= m_num_samples_lat)) {
        return 0.0;
    }
    if ((e0 < 0) || (e0 >= m_num_samples_ele) || (e1 < 0) || (e1 >= m_num_samples_ele)) {
        return 0.0;
    }
    
    // samples in a cube around current point
    float c000,c001,c010,c011,c100,c101,c110,c111;
    c000 = m_samples[getIndex(r0, l0, e0)];
    c001 = m_samples[getIndex(r0, l0, e1)];
    c010 = m_samples[getIndex(r0, l1, e0)];
    c011 = m_samples[getIndex(r0, l1, e1)];
    c100 = m_samples[getIndex(r1, l0, e0)];
    c101 = m_samples[getIndex(r1, l0, e1)];
    c110 = m_samples[getIndex(r1, l1, e0)];
    c111 = m_samples[getIndex(r1, l1, e1)];

    // radial interpolation
    const auto c00 = (1.0f-fractional_r)*c000 + fractional_r*c100;
    const auto c10 = (1.0f-fractional_r)*c010 + fractional_r*c110;
    const auto c01 = (1.0f-fractional_r)*c001 + fractional_r*c101;
    const auto c11 = (1.0f-fractional_r)*c011 + fractional_r*c111;

    // lateral interpolation
    const auto c0 = (1.0f-fractional_l)*c00 + fractional_l*c10;
    const auto c1 = (1.0f-fractional_l)*c01 + fractional_l*c11;

    // finally, elevational interpolation
    return (1.0f-fractional_e)*c0 + fractional_e*c1;
}

void LUTBeamProfile::setDiscreteSample(int ir, int il, int ie, float new_sample) {
    if (ir < 0 || ir >= m_num_samples_rad) return;
    if (il < 0 || il >= m_num_samples_lat) return;
    if (ie < 0 || ie >= m_num_samples_ele) return;
    m_samples[getIndex(ir, il, ie)] = new_sample;
}

}   // namespace

