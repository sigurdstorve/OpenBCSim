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

#include "bcsim_defines.h"
#include "BeamProfile.hpp"

namespace bcsim {

GaussianBeamProfile::GaussianBeamProfile(bc_float sigma_lateral, bc_float sigma_elevational) {
    setSigmaLateral(sigma_lateral);
    setSigmaElevational(sigma_elevational);
}

void GaussianBeamProfile::setSigmaLateral(bc_float new_sigma_lateral) {
    m_sigma_lateral = new_sigma_lateral;
    updateCaching();
}
    
void GaussianBeamProfile::setSigmaElevational(bc_float new_sigma_elevational) {
    m_sigma_elevational = new_sigma_elevational;
    updateCaching();
}
    
bc_float GaussianBeamProfile::sampleProfile(bc_float r, bc_float l, bc_float e) {
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
        
    // Allocate memory
    long num_samples = m_num_samples_rad*m_num_samples_lat*m_num_samples_ele;
    m_samples.resize(num_samples);
        
    // Compute sample deltas in all dimensions
    m_dr = (m_range_range.last - m_range_range.first) / m_num_samples_rad;
    m_dl = (m_lateral_range.last - m_lateral_range.first) / m_num_samples_lat;
    m_de = (m_elevational_range.last - m_elevational_range.first) / m_num_samples_ele;
        
}

bc_float LUTBeamProfile::sampleProfile(bc_float r, bc_float l, bc_float e) {
    // Compute array indices from physical beam coordinates.
    long ir = static_cast<long>((r-m_range_range.first) / m_dr);
    long il = static_cast<long>((l-m_lateral_range.first) / m_dl);
    long ie = static_cast<long>((e-m_elevational_range.first) / m_de); 
    if (ir < 0 || ir >= m_num_samples_rad) {
        return 0.0;
    }
    if (il < 0 || il >= m_num_samples_lat) {
        return 0.0;
    }
    if (ie < 0 || ie >= m_num_samples_ele) {
        return 0.0;
    }
    return m_samples[getIndex(ir, il, ie)];
}

void LUTBeamProfile::setSample(bc_float r, bc_float l, bc_float e, bc_float new_sample) {
    // Compute array indices from physical beam coordinates.
    // TODO: Don't duplicate code. Factor out into getIndex(float, float, float))?
    long ir = static_cast<long>((r-m_range_range.first) / m_dr);
    long il = static_cast<long>((l-m_lateral_range.first) / m_dl);
    long ie = static_cast<long>((e-m_elevational_range.first) / m_de); 
    if (ir < 0 || ir >= m_num_samples_rad) {
        return;
    }
    if (il < 0 || il >= m_num_samples_lat) {
        return;
    }
    if (ie < 0 || ie >= m_num_samples_ele) {
        return;
    }
    m_samples[getIndex(ir, il, ie)] = new_sample;
}

void LUTBeamProfile::setDiscreteSample(int ir, int il, int ie, bc_float new_sample) {
    if (ir < 0 || ir >= m_num_samples_rad) return;
    if (il < 0 || il >= m_num_samples_lat) return;
    if (ie < 0 || ie >= m_num_samples_ele) return;
    m_samples[getIndex(ir, il, ie)] = new_sample;
}

}   // namespace

