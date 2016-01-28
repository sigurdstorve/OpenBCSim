#include "cuda_kernels_spline2.cuh"
#include "cuda_helpers.h"   // for operator-
#include "cuda_kernels_common.cuh"
#include "common_definitions.h" // for MAX_SPLINE_DEGREE and MAX_NUM_CUDA_STREAMS
#include <math_functions.h> // for copysignf()

__constant__ float eval_basis[(MAX_SPLINE_DEGREE+1)*MAX_NUM_CUDA_STREAMS];

template <bool use_arc_projection, bool use_phase_delay, bool use_lut>
__global__ void SplineAlgKernel(SplineAlgKernelParams params) {

    const int global_idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (global_idx >= params.NUM_SPLINES) {
        return;
    }

    // step 1: evaluate spline
    // to get from one control point to the next, we have
    // to make a jump of size equal to number of splines
    float rendered_x = 0.0f;
    float rendered_y = 0.0f;
    float rendered_z = 0.0f;
    for (int i = params.cs_idx_start; i <= params.cs_idx_end; i++) {
        size_t eval_basis_i = i + params.eval_basis_offset_elements;
        rendered_x += params.control_xs[params.NUM_SPLINES*i + global_idx]*eval_basis[eval_basis_i-params.cs_idx_start];
        rendered_y += params.control_ys[params.NUM_SPLINES*i + global_idx]*eval_basis[eval_basis_i-params.cs_idx_start];
        rendered_z += params.control_zs[params.NUM_SPLINES*i + global_idx]*eval_basis[eval_basis_i-params.cs_idx_start];
    }

    // step 2: compute projections
    float3 point = make_float3(rendered_x, rendered_y, rendered_z) - params.origin;
    
    // compute dot products
    auto radial_dist  = dot(point, params.rad_dir);
    const auto lateral_dist = dot(point, params.lat_dir);
    const auto elev_dist    = dot(point, params.ele_dir);

    if (use_arc_projection) {
        // Use "arc projection" in the radial direction: use length of vector from
        // beam's origin to the scatterer with the same sign as the projection onto
        // the line.
        radial_dist = copysignf(sqrtf(dot(point,point)), radial_dist);
    }

    float weight;
    if (use_lut) {
        // Compute weight from lookup-table and radial_dist, lateral_dist, and elev_dist
        weight = ComputeWeightLUT(params.lut_tex, radial_dist, lateral_dist, elev_dist,
                                  params.lut_r_min, params.lut_r_max, params.lut_l_min, params.lut_l_max, params.lut_e_min, params.lut_e_max);
    } else {
        // Compute weight analytically
        weight = ComputeWeightAnalytical(params.sigma_lateral, params.sigma_elevational, radial_dist, lateral_dist, elev_dist);
    }

    const int radial_index = static_cast<int>(params.fs_hertz*2.0f*radial_dist/params.sound_speed + 0.5f);
    
    if (radial_index >= 0 && radial_index < params.num_time_samples) {
        if (use_phase_delay) {
            // handle sub-sample displacement with a complex phase
            const auto true_index = params.fs_hertz*2.0f*radial_dist/params.sound_speed;
            const float ss_delay = (radial_index - true_index)/params.fs_hertz;
            const float complex_phase = 6.283185307179586*params.demod_freq*ss_delay;

            // exp(i*theta) = cos(theta) + i*sin(theta)
            float sin_value, cos_value;
            sincosf(complex_phase, &sin_value, &cos_value);

            const auto w = weight*params.control_as[global_idx];
            atomicAdd(&(params.res[radial_index].x), w*cos_value);
            atomicAdd(&(params.res[radial_index].y), w*sin_value);
        } else {
            atomicAdd(&(params.res[radial_index].x), weight*params.control_as[global_idx]);
        }
    }
}
