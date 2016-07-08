#include <vector>
#include <functional>
#include "DefaultPhantoms.hpp"
#include "EllipsoidGeometry.hpp"

namespace default_phantoms {

typedef void* ScattererCollection;

ScattererCollection create_random_scatterers_in_box(const ellipsoid::Region3D& box, size_t num_scatterers, float thickness) {
    // TODO
    return 0;
}

std::vector<float> create_random_amplitudes(size_t num, float low, float high) {
    // TODO
    std::vector<float> res;
    return res;
}

void remove_scatterers_outside(ScattererCollection&, const ellipsoid::ThickCappedZEllipsoid& lv_model) {
    // TODO
}

float scale_function(float time) {
    // TODO
    return 1.0f;
}

bcsim::Scatterers::u_ptr CreateLeftVentricle3dPhantom(const LeftVentriclePhantomParameters& par) {
    const auto region = ellipsoid::Region3D(par.x_min, par.x_max, par.y_min, par.y_max, par.z_min, par.z_max);
    const auto lv_model = ellipsoid::ThickCappedZEllipsoid(region, par.thickness, par.z_ratio);
    
    auto scatterers = create_random_scatterers_in_box(region, par.num_scatterers, par.thickness);
    remove_scatterers_outside(scatterers, lv_model);
    size_t num_scatterers = 0;  // TODO, after filtering
    const auto amplitudes = create_random_amplitudes(num_scatterers, 0.0f, par.lv_max_amplitude);


    // TODO: Create a knot vector for the number of control points
    //knots = uniform_regular_knot_vector(par.num_cs, par.spline_degree, par.t0, par.t1)

    //knot_avgs = bsplines.control_points(args.spline_degree, knot_vector)


    //value in[0, 1] for the normalized z coordinate of each scatterer will be used to control rotation amplitude.
    //const auto zs_fractional = (zs - args.z_min) / (args.z_max - args.z_min)

    //control_points = np.zeros((num_scatterers, args.num_cs, 3), dtype = 'float32')
    for (int cs_index = 0; cs_index < par.num_cs; cs_index++) {
        float t_star; // = compute_knot_avg(cs_index);

        const auto s = scale_function(t_star);

        //temp_in = np.vstack([s*xs, s*ys, s*zs])
        //temp_out = np.empty((3, num_scatterers))
        for (int i = 0; i < num_scatterers; i++) {
            //cur_angle = zs_fractional[i] * s*args.rotation_scale
            //temp_out[:, i] = rot_mat_z(cur_angle).dot(temp_in[:, i])

            // Compute control point position.Amplitude is unchanged.
            //control_points[:, cs_i, 0] = temp_out[0, :] #s*np.array(xs)
            //control_points[:, cs_i, 1] = temp_out[1, :] #s*np.array(ys)
            //control_points[:, cs_i, 2] = temp_out[2, :] #s*np.array(zs)
        }
    }
    
    // TODO:
    return nullptr;
}

}   // end namespace
