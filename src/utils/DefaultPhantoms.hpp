#pragma once
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include "EllipsoidGeometry.hpp"
#include "core/BCSimConfig.hpp"
#include "core/export_macros.hpp"

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
class DLL_PUBLIC LeftVentricle3dPhantomFactory {
public:
    typedef std::function<void(const std::string&)> LogCallback;

    // csv_stream is any stream which contains csv data [for the scaling signal]
    LeftVentricle3dPhantomFactory(const LeftVentriclePhantomParameters& params, std::istream& csv_stream, LogCallback = nullptr);

    bcsim::SplineScatterers::u_ptr get();

private:
    // creates an interpolated function from samples loaded from CSV
    void load_csv_scale_signal(std::istream&);

    // fill a rectangular region of space with uniformly random scatterers
    void create_random_scatterers_in_box(size_t num_scatterers, float thickness);

    // create uniformly random amplitudes
    void create_random_amplitudes(size_t num, float low, float high);

    // use the mathematical shape model to filter out unwanted scatterers
    void remove_scatterers_outside();

    // create the scatterer splines
    void create_splines(const LeftVentriclePhantomParameters& params);

private:
    ellipsoid::Region3D                 m_box_region;
    ellipsoid::ThickCappedZEllipsoid    m_mathematical_model;
    std::vector<ellipsoid::Point3D>     m_points;
    std::vector<float>                  m_amplitudes;
    std::function<float(float)>         m_scale_function;
    LogCallback                         m_log_callback;
    bcsim::SplineScatterers::u_ptr      m_spline_scatterers;
};

}   // end namespace
