#include <vector>
#include <functional>
#include <algorithm>
#include <utility>
#include <iostream>
#include <cassert>
#include "DefaultPhantoms.hpp"
#include "CSVReader.hpp"
#include "../core/bspline.hpp"
#include "rotation3d.hpp"

namespace default_phantoms {

typedef void* ScattererCollection;

// Currently only supports nearest-neighbour interpolation!
template <typename T>
class InterpolatedFunction {
public:
    InterpolatedFunction(std::vector<T>&& xs, std::vector<T>&& ys)
        : m_xs(std::move(xs)), m_ys(std::move(ys)) { }
    
    T operator()(T x_in) {
        auto i = std::min_element(std::begin(m_xs), std::end(m_xs), [=](T x, T y) {
            return std::abs(x - x_in) < std::abs(y - x_in);
        });
        auto index = std::distance(std::begin(m_xs), i);
        return m_ys[index];
    }
private:
    std::vector<T> m_xs;
    std::vector<T> m_ys;
};

void LeftVentricle3dPhantomFactory::create_random_scatterers_in_box(size_t num_scatterers, float thickness) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> x_distr(m_box_region.x_min - thickness, m_box_region.x_max + thickness);
    std::uniform_real_distribution<float> y_distr(m_box_region.y_min - thickness, m_box_region.y_max + thickness);
    std::uniform_real_distribution<float> z_distr(m_box_region.z_min - thickness, m_box_region.z_max + thickness);
    m_points.reserve(num_scatterers);
    for (size_t i = 0; i < num_scatterers; i++) {
        m_points.emplace_back(ellipsoid::Point3D(x_distr(gen), y_distr(gen), z_distr(gen)));
    }
}

void LeftVentricle3dPhantomFactory::create_random_amplitudes(size_t num, float low, float high) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> a_distr(low, high);
    m_amplitudes.reserve(num);
    for (size_t i = 0; i < num; i++) {
        m_amplitudes.push_back(a_distr(gen));
    }
}

void LeftVentricle3dPhantomFactory::remove_scatterers_outside() {
    m_points.erase(std::remove_if(std::begin(m_points), std::end(m_points), [&](const ellipsoid::Point3D& p) {
        return !m_mathematical_model.is_point_inside(p);
    }), std::end(m_points));
}

void LeftVentricle3dPhantomFactory::load_csv_scale_signal(std::istream& csv_stream) {
    csv::CSVReader reader(std::move(csv_stream), ';');
    auto times_vector   = reader.get_column<float>("times");
    auto factors_vector = reader.get_column<float>("factors");
    if (times_vector.size() != factors_vector.size()) {
        throw std::runtime_error("invalid data loaded from csv stream");
    }
    m_log_callback("Data from CSV contains " + std::to_string(times_vector.size()) + " samples");
    m_scale_function = InterpolatedFunction<float>(std::move(times_vector), std::move(factors_vector));
    
    std::vector<float> print_times{0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    for (const auto t : print_times) {
        m_log_callback("Scale value at time " + std::to_string(t) + " : " + std::to_string(m_scale_function(t)));
    }
}

LeftVentricle3dPhantomFactory::LeftVentricle3dPhantomFactory(const LeftVentriclePhantomParameters& par, std::istream& csv_stream, LogCallback log_callback)
    : m_box_region(ellipsoid::Region3D(par.x_min, par.x_max, par.y_min, par.y_max, par.z_min, par.z_max)),
      m_mathematical_model(ellipsoid::ThickCappedZEllipsoid(m_box_region, par.thickness, par.z_ratio)),
      m_log_callback(log_callback)
{
    if (!m_log_callback) {
        m_log_callback = [](const std::string&) {};
    }
    load_csv_scale_signal(csv_stream);

    create_random_scatterers_in_box(par.num_scatterers, par.thickness);
    remove_scatterers_outside();
    const auto num_scatterers_left = m_points.size();
    m_log_callback("After filtering " + std::to_string(num_scatterers_left) + " scatterers remain");
    if (par.lv_max_amplitude < 0.0) {
        throw std::runtime_error("LV max amplitude must be positive");
    }
    create_random_amplitudes(num_scatterers_left, -par.lv_max_amplitude, par.lv_max_amplitude);
    create_splines(par);
}


bcsim::SplineScatterers::u_ptr LeftVentricle3dPhantomFactory::get() {
    return std::move(m_spline_scatterers);
}

void LeftVentricle3dPhantomFactory::create_splines(const LeftVentriclePhantomParameters& par) {
    const auto knots = bspline_storve::uniform_regular_knot_vector(par.num_cs, par.spline_degree, par.t0, par.t1);
    const auto knot_avgs = bspline_storve::control_points(par.spline_degree, knots);

    // Dump knot vector to log
    {
        std::stringstream ss;
        ss << "knot vector: ";
        for (size_t i = 0; i < knots.size(); i++) {
            ss << knots[i] << ", ";
        }
        m_log_callback(ss.str());
    }

    // Dump knot averages to log
    {
        std::stringstream ss;
        ss << "knot avgs: ";
        for (size_t i = 0; i < knot_avgs.size(); i++) {
            ss << knot_avgs[i] << ", ";
        }
        m_log_callback(ss.str());
    }
    m_spline_scatterers = std::make_unique<bcsim::SplineScatterers>();

    const auto num_splines = m_amplitudes.size();
    m_spline_scatterers->spline_degree = par.spline_degree;
    m_spline_scatterers->amplitudes = m_amplitudes;
    m_spline_scatterers->knot_vector = knots;
    m_spline_scatterers->control_points.reserve(num_splines);

    for (size_t spline_no = 0; spline_no < num_splines; spline_no++) {
        //value in[0, 1] for the normalized z coordinate of each scatterer will be used to control rotation amplitude.
        const auto zs_fractional = (m_points[spline_no].z - m_box_region.z_min) / (m_box_region.z_max - m_box_region.z_min);
        std::vector<bcsim::vector3> control_points;
        control_points.reserve(par.num_cs);
        for (size_t cs_i = 0; cs_i < par.num_cs; cs_i++) {
            const auto t_star = knot_avgs[cs_i];
            const auto cur_scale = m_scale_function(t_star);
            const auto cur_angle = zs_fractional * cur_scale*par.rotation_scale;
            const auto rot_matrix = rotation_matrix_z<float>(cur_angle);

            boost::numeric::ublas::vector<float> p(3);
            p(0) = cur_scale*m_points[spline_no].x;
            p(1) = cur_scale*m_points[spline_no].y;
            p(2) = cur_scale*m_points[spline_no].z;
            
            const auto p_rotated = boost::numeric::ublas::prod(rot_matrix, p);
            control_points.emplace_back(bcsim::vector3(p_rotated(0), p_rotated(1), p_rotated(2)));
        }
        m_spline_scatterers->control_points.emplace_back(control_points);
    }
}

}   // end namespace
