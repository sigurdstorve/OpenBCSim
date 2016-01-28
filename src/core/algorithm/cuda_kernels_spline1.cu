#define MAX_SPLINE_DEGREE 4

__constant__ float eval_basis[MAX_SPLINE_DEGREE+1];

__global__ void RenderSplineKernel(const float* control_xs,
                                   const float* control_ys,
                                   const float* control_zs,
                                   float* rendered_xs,
                                   float* rendered_ys,
                                   float* rendered_zs,
                                   int cs_idx_start,
                                   int cs_idx_end,
                                   int NUM_SPLINES) {

    const int idx = blockDim.x*blockIdx.x + threadIdx.x;
    if (idx >= NUM_SPLINES) {
        return;
    }

    // to get from one control point to the next, we have
    // to make a jump of size equal to number of splines
    float rendered_x = 0.0f;
    float rendered_y = 0.0f;
    float rendered_z = 0.0f;
    for (int i = cs_idx_start; i <= cs_idx_end; i++) {
        rendered_x += control_xs[NUM_SPLINES*i + idx]*eval_basis[i-cs_idx_start];
        rendered_y += control_ys[NUM_SPLINES*i + idx]*eval_basis[i-cs_idx_start];
        rendered_z += control_zs[NUM_SPLINES*i + idx]*eval_basis[i-cs_idx_start];
    }

    // write result to memory
    rendered_xs[idx] = rendered_x;
    rendered_ys[idx] = rendered_y;
    rendered_zs[idx] = rendered_z;
}
