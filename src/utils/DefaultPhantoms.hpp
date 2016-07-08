#pragma once
#include <memory>
#include "core/BCSimConfig.hpp"

namespace default_phantoms {

struct LeftVentriclePhantomParameters {
    LeftVentriclePhantomParameters()
        : thickness(8e-3f),
          z_ratio(0.7f),
          x_min(-0.02f), x_max(0.02f), y_min(-0.02f), y_max(0.02f), z_min(0.008f), z_max(0.09f),
          num_scatterers(400000),
          motion_amplitude(0.25f),
          t0(0.0f), t1(1.0f),
          spline_degree(2),
          num_cs(10),
          lv_max_amplitude(1.0f),
          rotation_scale(3.0f)
    {
    }

    float thickness;                                    // Thickness of myocardium [m]
    float z_ratio;                                      // Ratio in [0,1] of where to cap ellipsoid
    float x_min, x_max, y_min, y_max, z_min, z_max;     // Cartesian extent [m]    
    size_t num_scatterers;                              // Number of scatterers in box around myocardium prior to filtering
    float motion_amplitude;                             // Amplitude of contraction, higher value gives more contraction
    float t0, t1;                                       // Start and end time [s]
    int spline_degree;                                  // Spline degree to use for individual spline scatterers
    int num_cs;                                         // Number of control points to use for each spline scatterer
    float lv_max_amplitude;                             // Max amplitude for the spline scatterers
    float rotation_scale;                               // "Gain" for rotation. Will use same signal as contraction - higher value gives more rotation.
};

// Create a LV spline phantom model by randomly distributing point-scatterers in a box and then removing
// those who are not inside a 3D capped ellipsoid. The remaining points are then scaled and rotated
// to generate control points for splines.
bcsim::Scatterers::u_ptr CreateLeftVentricle3dPhantom(const LeftVentriclePhantomParameters& type);

}   // end namespace
