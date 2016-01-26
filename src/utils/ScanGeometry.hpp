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
#include <memory>

namespace bcsim {

// Interface for conventional scanning geometries.
struct ScanGeometry {
    typedef std::shared_ptr<ScanGeometry> ptr;

    virtual void get_xy_extent(float& x_min, float& x_max, float& y_min, float& y_max) const = 0;
};

// TODO: Consider adding 2D to name
struct SectorScanGeometry : public ScanGeometry {
    float width;            // Sector width [radians]
    float depth;            // Depth [meters]
    float tilt;             // Sector tilt [radians]

    virtual void get_xy_extent(float& x_min, float& x_max, float& y_min, float& y_max) const {
        float min_angle, max_angle;
        get_angle_limits(min_angle, max_angle);

        x_min = static_cast<float>(depth*std::cos(max_angle));
        x_max = static_cast<float>(depth*std::cos(min_angle));
        y_min = 0.0f;
        y_max = depth;
    }
    
    void get_angle_limits(float& /*out*/ min_angle, float& /*out*/ max_angle) const {
        const auto PI = static_cast<float>(4.0*std::atan(1.0f));
        min_angle = 0.5f*PI - 0.5f*width + tilt;
        max_angle = 0.5f*PI + 0.5f*width + tilt;
    }

};

// TODO: Consider adding 2D to name
struct LinearScanGeometry : public ScanGeometry {
    float width;        // Scan width [meters]
    float range_max;    // Maximum range distance [meters]

    virtual void get_xy_extent(float& x_min, float& x_max, float& y_min, float& y_max) const {
        x_min = -0.5f*width;
        x_max = 0.5f*width;
        y_min = 0.0f;
        y_max = range_max;
    }
};

inline void GetCartesianDimensions(ScanGeometry::ptr geometry, float& /*out*/ width, float& /*out*/ height) {
    float x_min, x_max, y_min, y_max;
    geometry->get_xy_extent(x_min, x_max, y_min, y_max);
    width = x_max-x_min;
    height = y_max-y_min;
}

}   // end namespace
