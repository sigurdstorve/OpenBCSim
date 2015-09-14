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

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <cuda.h>
#include <device_launch_parameters.h>
#include "algorithm/cuda_helpers.h"
#include "bspline.hpp"

/* Code for benchmarking different ways of rendering spline curves on the GPU.
 * The main use is to demonstrate that the memory layout used when representing
 * spline curves can have a great impact on the running speed (alost a factor
 * of 10 on Quadro K4000M Kepler architecture) when evaluating splines.
 *
 * The results from this benchmark script was used to determine the best way
 * to implement the "gpu_spline2" simulator algorithm.
 */

// Total number of splines to be rendered.
#define NUM_SPLINES (128*8000)

// Number of control points per spline.
#define NUM_CS 10

// Spline degree
#define SPLINE_DEGREE 3

// The parameter value where the curve should be evaluated.
#define PARAMETER_VAL 0.5f

struct point {
    float x, y, z;
    point() : x(0.0f), y(0.0f), z(0.0f) { }
    point(float x_, float y_, float z_) : x(x_), y(y_), z(z_) { }
};

point operator+(const point& lhs, const point& rhs) {
    return point(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

point operator*(const point& lhs, float rhs) {
    return point(lhs.x*rhs, lhs.y*rhs, lhs.z*rhs);
}

point operator*(float lhs, const point& rhs) {
    return operator*(rhs, lhs);
}

bool AlmostEqual(const point& a, const point& b, double eps=1e-6) {
    return (std::abs(a.x-b.x) < eps) && (std::abs(a.y-b.y) < eps) && (std::abs(a.z-b.z) < eps);
}

typedef std::vector<std::array<point, NUM_CS> > splines_t;

// Evaluated basis functions on the GPU
__constant__ float eval_basis_gpu[NUM_CS];

// Evaluated basis functions on the CPU
std::array<float, NUM_CS> eval_basis_cpu;

// Uncoalesced memory access pattern.
__global__ void RenderSplinesV1(const float3* __restrict__ control_points,
                                float3* __restrict__ rendered_points) {
    const int idx = blockDim.x*blockIdx.x + threadIdx.x;
    float3 rendered = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < NUM_CS; i++) {
        rendered = rendered + control_points[NUM_CS*idx + i]*eval_basis_gpu[i];
    }
    rendered_points[idx] = rendered;
}

// Coalesced memory access pattern.
__global__ void RenderSplinesV2(const float* __restrict__ control_xs,
                                const float* __restrict__ control_ys,
                                const float* __restrict__ control_zs,
                                float* __restrict__ rendered_xs,
                                float* __restrict__ rendered_ys,
                                float* __restrict__ rendered_zs) {

    const int idx = blockDim.x*blockIdx.x + threadIdx.x;
    // to get from one control point to the next, we have
    // to make a jump of size equal to number of splines
    float rendered_x = 0.0f;
    float rendered_y = 0.0f;
    float rendered_z = 0.0f;
    for (int i = 0; i < NUM_CS; i++) {
        rendered_x += control_xs[NUM_SPLINES*i + idx]*eval_basis_gpu[i];
        rendered_y += control_ys[NUM_SPLINES*i + idx]*eval_basis_gpu[i];
        rendered_z += control_zs[NUM_SPLINES*i + idx]*eval_basis_gpu[i];
    }

    // write result to memory
    rendered_xs[idx] = rendered_x;
    rendered_ys[idx] = rendered_y;
    rendered_zs[idx] = rendered_z;
}

// Using kernel with coalesced memory access pattern.
std::vector<point> EvaluateSplinesGpu1(const splines_t& all_splines, int num_rep=100) {
    // device memory to hold all control points of all splines
    const size_t total_num_cs = NUM_SPLINES*NUM_CS;
    const size_t cs_num_bytes = total_num_cs*sizeof(float3);
    DeviceBufferRAII<float3> device_cs(cs_num_bytes);
    
    // reorganize host memory and upload to device memory
    std::vector<float3> host_cs(total_num_cs);
    for (size_t spline_no = 0; spline_no < NUM_SPLINES; spline_no++) {
        for (size_t control_point_no = 0; control_point_no < NUM_CS; control_point_no++) {
            const size_t offset = NUM_CS*spline_no + control_point_no;
            const auto& p = all_splines[spline_no][control_point_no];
            host_cs[offset] = make_float3(p.x, p.y, p.z);
        }
    }
    cudaErrorCheck( cudaMemcpy(device_cs.data(), host_cs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );

    // device memory to hold rendered splines
    size_t rendered_num_bytes = NUM_SPLINES*sizeof(float3);
    DeviceBufferRAII<float3> device_rendered(rendered_num_bytes);

    EventTimerRAII event_timer;
    int num_threads = 128;
    dim3 grid_size(NUM_SPLINES/num_threads, 1, 1);
    dim3 block_size(num_threads, 1, 1);
    
    event_timer.restart();
    for (int i = 0; i < num_rep; i++) {
        RenderSplinesV1<<<grid_size, block_size>>>(device_cs.data(),
                                                   device_rendered.data());
    }
    cudaErrorCheck( cudaDeviceSynchronize() );
    auto ms = event_timer.stop() / num_rep;
    std::cout << "Kernel RenderSplinesV1 used " << ms << " millisec per kernel call.\n";
    std::cout << "Total number of calls was " << num_rep << std::endl;

    // copy rendered points back to host
    std::vector<float3> host_rendered(NUM_SPLINES);
    cudaErrorCheck( cudaMemcpy(host_rendered.data(), device_rendered.data(), rendered_num_bytes, cudaMemcpyDeviceToHost) );

    // Store in vector of points
    std::vector<point> res(NUM_SPLINES);
    for (size_t spline_no = 0; spline_no < NUM_SPLINES; spline_no++) {
        res[spline_no] = point(host_rendered[spline_no].x,
                               host_rendered[spline_no].y,
                               host_rendered[spline_no].z);
    }
    return res;
}

// Using kernel with uncoalesced memory access pattern.
std::vector<point> EvaluateSplinesGpu2(const splines_t& all_splines, int num_rep=100) {
    // device memory to hold x, y, z components of all spline control points
    const size_t total_num_cs = NUM_SPLINES*NUM_CS;
    const size_t cs_num_bytes = total_num_cs*sizeof(float);
    DeviceBufferRAII<float> control_xs(cs_num_bytes);
    DeviceBufferRAII<float> control_ys(cs_num_bytes);
    DeviceBufferRAII<float> control_zs(cs_num_bytes);

    // reorganize host memory and upload to device memory.
    std::vector<float> host_control_xs(total_num_cs);
    std::vector<float> host_control_ys(total_num_cs);
    std::vector<float> host_control_zs(total_num_cs);
    for (size_t control_point_no = 0; control_point_no < NUM_CS; control_point_no++) {
        for (size_t spline_no = 0; spline_no < NUM_SPLINES; spline_no++) {
            const size_t offset = NUM_SPLINES*control_point_no + spline_no;
            host_control_xs[offset] = all_splines[spline_no][control_point_no].x;
            host_control_ys[offset] = all_splines[spline_no][control_point_no].y;
            host_control_zs[offset] = all_splines[spline_no][control_point_no].z;
        }
    }
    cudaErrorCheck( cudaMemcpy(control_xs.data(), host_control_xs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(control_ys.data(), host_control_ys.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(control_zs.data(), host_control_zs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );

    // device memory to hold x, y, z components of rendered splines
    size_t rendered_num_bytes = NUM_SPLINES*sizeof(float);
    DeviceBufferRAII<float> rendered_xs(rendered_num_bytes);
    DeviceBufferRAII<float> rendered_ys(rendered_num_bytes);
    DeviceBufferRAII<float> rendered_zs(rendered_num_bytes);


    EventTimerRAII event_timer;
    int num_threads = 128;
    dim3 grid_size(NUM_SPLINES/num_threads, 1, 1);
    dim3 block_size(num_threads, 1, 1);
    
    event_timer.restart();
    for (int i = 0; i < num_rep; i++) {
        RenderSplinesV2<<<grid_size, block_size>>>(control_xs.data(),
                                                   control_ys.data(),
                                                   control_zs.data(),
                                                   rendered_xs.data(),
                                                   rendered_ys.data(),
                                                   rendered_zs.data());
    }
    cudaErrorCheck( cudaDeviceSynchronize() );
    auto ms = event_timer.stop() / num_rep;
    std::cout << "Kernel RenderSplinesV2 used " << ms << " millisec per kernel call.\n";
    std::cout << "Total number of calls was " << num_rep << std::endl;

    // copy rendered points back to host
    std::vector<float> host_rendered_xs(NUM_SPLINES);
    std::vector<float> host_rendered_ys(NUM_SPLINES);
    std::vector<float> host_rendered_zs(NUM_SPLINES);
    cudaErrorCheck( cudaMemcpy(host_rendered_xs.data(), rendered_xs.data(), rendered_num_bytes, cudaMemcpyDeviceToHost) );
    cudaErrorCheck( cudaMemcpy(host_rendered_ys.data(), rendered_ys.data(), rendered_num_bytes, cudaMemcpyDeviceToHost) );
    cudaErrorCheck( cudaMemcpy(host_rendered_zs.data(), rendered_zs.data(), rendered_num_bytes, cudaMemcpyDeviceToHost) );

    // Store in vector of points
    std::vector<point> res(NUM_SPLINES);
    for (size_t spline_no = 0; spline_no < NUM_SPLINES; spline_no++) {
        res[spline_no] = point(host_rendered_xs[spline_no],
                               host_rendered_ys[spline_no],
                               host_rendered_zs[spline_no]);
    }
    return res;
}

std::vector<point> EvaluateSplinesCpu(const splines_t& all_splines) {
    size_t num_splines = all_splines.size();
    std::vector<point> res(num_splines);
    for (size_t spline_no = 0; spline_no < num_splines; spline_no++) {
        res[spline_no] = point();
        for (size_t control_point_no = 0; control_point_no < NUM_CS; control_point_no++) {
            res[spline_no] = res[spline_no] + all_splines[spline_no][control_point_no]*eval_basis_cpu[control_point_no];
        }
    }
    return res;
}

// Precompute basis functions and upload to the GPU constant memory.
void PrecomputeBasisFunctions() {
    // Create a simple open knot vector
    size_t num_knots = NUM_CS+SPLINE_DEGREE+1;
    std::vector<float> knots;
    for (size_t i = 0; i < num_knots; i++) {
        knots.push_back(i / static_cast<float>(num_knots));        
    }
    for (int i = 0; i < NUM_CS; i++) {
        eval_basis_cpu[i] = bspline_storve::bsplineBasis(i, SPLINE_DEGREE, PARAMETER_VAL, knots);
    }
    cudaErrorCheck( cudaMemcpyToSymbol(eval_basis_gpu, eval_basis_cpu.data(), NUM_CS*sizeof(float)) );
}

// Create a vector of splines [which again are vectors of points].
// All control points are random numbers.
splines_t CreateSplines() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    splines_t res(NUM_SPLINES);
    for (size_t spline_no = 0; spline_no < NUM_SPLINES; spline_no++) {
        for (size_t control_point_no = 0; control_point_no < NUM_CS; control_point_no++) {
            res[spline_no][control_point_no].x = dis(gen);
        }
    }
    return res;
}

void CompareResults(const std::vector<point>& points1, const std::vector<point>& points2, double eps=1e-6) {
    auto num_points = points1.size();
    if (points2.size() != num_points) {
        throw std::runtime_error("number of points not equal");
    }
    for (size_t spline_no = 0; spline_no < NUM_SPLINES; spline_no++) {
        const auto& p1 = points1[spline_no];
        const auto& p2 = points2[spline_no];
        if (!AlmostEqual(p1, p2, eps)) {
            std::cout << "Equality check failed for spline " << spline_no << std::endl;
            std::cout << "   "   << p1.x << "," << p1.y << "," << p1.z  
                      << " vs. " << p2.x << "," << p2.y << "," << p2.z << std::endl;
            return;
        }
    }
    std::cout << "Passed.\n";
}

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    std::cout << "CUDA device count: " << device_count << std::endl;
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;


    PrecomputeBasisFunctions();
    auto all_splines = CreateSplines();
    std::cout << "Created " << all_splines.size() << " splines.\n";

    const int num_rep = 1000;

    auto cpu_res  = EvaluateSplinesCpu(all_splines);
    auto gpu_res1 = EvaluateSplinesGpu1(all_splines, num_rep);
    auto gpu_res2 = EvaluateSplinesGpu2(all_splines, num_rep);
    
    double tolerance = 1e-6;
    std::cout << "Comparing CPU and GPU v1...";
    CompareResults(cpu_res, gpu_res1, tolerance);
    std::cout << "Comparing CPU and GPU v2...";
    CompareResults(cpu_res, gpu_res2, tolerance);
    std::cout << "Comparing GPU v1 and GPU v2...";
    CompareResults(gpu_res1, gpu_res2, tolerance);
}
