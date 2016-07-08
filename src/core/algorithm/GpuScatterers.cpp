#include <stdexcept>
#include <tuple>
#include "common_definitions.h" // for MAX_SPLINE_DEGREE
#include "GpuScatterers.hpp"
#include "cuda_kernels_c_interface.h"
#include "../bspline.hpp"

namespace bcsim {

DeviceFixedScatterers::DeviceFixedScatterers(size_t num_scatterers)
    : m_num_scatterers(num_scatterers)
{
    const auto num_bytes = num_scatterers*sizeof(float);
    xs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes));
    ys = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes));
    zs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes));
    as = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes));
}

size_t DeviceFixedScatterers::get_num_scatterers() const {
    return m_num_scatterers;
}

float* DeviceFixedScatterers::get_xs_ptr() const {
    return xs->data();
}

float* DeviceFixedScatterers::get_ys_ptr() const {
    return ys->data();
}

float* DeviceFixedScatterers::get_zs_ptr() const {
    return zs->data();
}

float* DeviceFixedScatterers::get_as_ptr() const {
    return as->data();
}

// create a new dataset and fill it with data (will allocate memory on device)
void DeviceFixedScatterersCollection::add(bcsim::FixedScatterers::s_ptr host_scatterers) {
    auto new_device_scatterers = std::make_shared<DeviceFixedScatterers>(host_scatterers->num_scatterers());
    transfer_to_device(host_scatterers, new_device_scatterers);
    m_fixed_datasets.push_back(new_device_scatterers);
}

void DeviceFixedScatterersCollection::render(const DeviceSplineScatterersCollection& spline_datasets, float timestamp) {
    const auto current_num_datasets = get_num_datasets();
    const auto needed_num_datasets  = spline_datasets.get_num_datasets();

    // cache the number of splines in each spline dataset
    std::vector<size_t> spline_counts(needed_num_datasets);
    for (size_t dset_idx = 0; dset_idx < needed_num_datasets; dset_idx++) {
        spline_counts[dset_idx] = spline_datasets.get_dataset(dset_idx)->get_num_scatterers();
    }

    bool must_check_capacity = true;
    if (current_num_datasets != needed_num_datasets) {
        m_log_callback("number of datasets doesn not match, recreating");
        clear();
        for (size_t dset_idx = 0; dset_idx < needed_num_datasets; dset_idx++) {
            m_log_callback("Making empty fixed dataset with capacity of " + std::to_string(spline_counts[dset_idx]) + " scatterers");
            m_fixed_datasets.push_back(std::make_shared<DeviceFixedScatterers>(spline_counts[dset_idx]));
        }
        must_check_capacity = false;
    }

    if (must_check_capacity) {
        // reallocate if size doesn't match
        for (size_t dset_idx = 0; dset_idx < current_num_datasets; dset_idx++) {
            const auto num_splines = spline_counts[dset_idx];
            const auto cur_num_scatterers = m_fixed_datasets[dset_idx]->get_num_scatterers();
            if (cur_num_scatterers != num_splines) {
                m_log_callback("Need different number of splines. Reallocating...");
                m_fixed_datasets[dset_idx] = std::make_shared<DeviceFixedScatterers>(num_splines);
            }
        }
    }

    // at this point everything is ready for updating scatterers
    for (size_t dset_idx = 0; dset_idx < spline_datasets.get_num_datasets(); dset_idx++) {
        cudaStream_t stream_no = 0; // TODO: use streams to overlap transfers

        const auto spline_dataset = spline_datasets.get_dataset(dset_idx);
        const auto cur_knots = spline_dataset->get_knots();
        const auto num_cs    = spline_dataset->get_num_cs();
        const auto spline_degree = spline_dataset->get_spline_degree();

        // evaluate the basis functions and upload to constant memory.
        const auto num_nonzero = spline_degree+1;
        std::vector<float> host_basis_functions(num_cs);
        for (int i = 0; i < num_cs; i++) {
            host_basis_functions[i] = bspline_storve::bsplineBasis(i, spline_degree, timestamp, cur_knots);
        }

        // compute sum limits (inclusive)
        int cs_idx_start, cs_idx_end;
        std::tie(cs_idx_start, cs_idx_end) = bspline_storve::get_lower_upper_inds(cur_knots, timestamp, spline_degree);

        if (cs_idx_end-cs_idx_start+1 != num_nonzero) {
            throw std::logic_error("illegal number of non-zero basis functions");
        }

        if(!splineAlg1_updateConstantMemory(host_basis_functions.data() + cs_idx_start, num_nonzero*sizeof(float))) {
            throw std::runtime_error("Failed to upload basis functions to constant memory");
        }
        const auto num_splines = spline_dataset->get_num_scatterers();
        int num_threads = 128; // per block
        int num_blocks = round_up_div(num_splines, 128);
        stream_no = 0;
        launch_RenderSplineKernel(num_blocks, num_threads, stream_no,
                                  spline_dataset->get_xs_ptr(),
                                  spline_dataset->get_ys_ptr(),
                                  spline_dataset->get_zs_ptr(),
                                  m_fixed_datasets[dset_idx]->get_xs_ptr(),
                                  m_fixed_datasets[dset_idx]->get_ys_ptr(),
                                  m_fixed_datasets[dset_idx]->get_zs_ptr(),
                                  cs_idx_start, cs_idx_end, num_splines);
        // copy amplitudes [TODO: these can be reused]
        cudaErrorCheck(cudaMemcpy(m_fixed_datasets[dset_idx]->get_as_ptr(), spline_dataset->get_as_ptr(), num_splines*sizeof(float), cudaMemcpyDeviceToDevice));
    }
}

void DeviceFixedScatterersCollection::transfer_to_device(bcsim::FixedScatterers::s_ptr host_scatterers,
                                                         DeviceFixedScatterers::s_ptr device_scatterers) {
    const auto num_scatterers = host_scatterers->num_scatterers();

    // reorganize and transfer
    size_t bytes_per_component = num_scatterers*sizeof(float);

    // temporary host memory for scatterer points
    HostPinnedBufferRAII<float> host_temp(bytes_per_component);

    // transfer x values
    for (size_t i = 0; i < num_scatterers; i++) {
        host_temp.data()[i] = host_scatterers->scatterers[i].pos.x;
    }
    cudaErrorCheck( cudaMemcpy(device_scatterers->get_xs_ptr(), host_temp.data(),
                                bytes_per_component, cudaMemcpyHostToDevice) );

    // transfer y values
    for (size_t i = 0; i < num_scatterers; i++) {
        host_temp.data()[i] = host_scatterers->scatterers[i].pos.y;
    }
    cudaErrorCheck( cudaMemcpy(device_scatterers->get_ys_ptr(), host_temp.data(),
                                bytes_per_component, cudaMemcpyHostToDevice) );

    // z values
    for (size_t i = 0; i < num_scatterers; i++) {
        host_temp.data()[i] = host_scatterers->scatterers[i].pos.z;
    }
    cudaErrorCheck( cudaMemcpy(device_scatterers->get_zs_ptr(), host_temp.data(),
                                bytes_per_component, cudaMemcpyHostToDevice) );

    // a values
    for (size_t i = 0; i < num_scatterers; i++) {
        host_temp.data()[i] = host_scatterers->scatterers[i].amplitude;
    }
    cudaErrorCheck( cudaMemcpy(device_scatterers->get_as_ptr(), host_temp.data(),
                                bytes_per_component, cudaMemcpyHostToDevice) );
}

void DeviceFixedScatterersCollection::update(bcsim::FixedScatterers::s_ptr host_scatterers, size_t dset_idx) {
    if (dset_idx >= m_fixed_datasets.size()) throw std::runtime_error("Illegal dataset index");
    throw std::runtime_error("TODO");
    /*
    // check capacity of existsing
    // no point in reallocating if we already have allocated memory and the number of bytes
    // is the same.
    bool reallocate_device_mem = true;
    if (m_device_point_xs && m_device_point_ys && m_device_point_zs && m_device_point_as) {
        if (   (m_device_point_xs->get_num_bytes() == points_common_bytes)
            && (m_device_point_ys->get_num_bytes() == points_common_bytes)
            && (m_device_point_zs->get_num_bytes() == points_common_bytes)
            && (m_device_point_as->get_num_bytes() == points_common_bytes))
        {
            reallocate_device_mem = false;
        }
    */
    // if does not match, reallocate

    // transfer from host to device.
}

void DeviceFixedScatterersCollection::clear() {
    m_fixed_datasets.clear();
}

size_t DeviceFixedScatterersCollection::get_total_num_scatterers() const {
    size_t sum = 0;
    for (size_t i = 0; i < m_fixed_datasets.size(); i++) {
        sum += m_fixed_datasets[i]->get_num_scatterers();
    }
    return sum;
}

size_t DeviceFixedScatterersCollection::get_num_datasets() const {
    return m_fixed_datasets.size();
}

DeviceFixedScatterers::s_ptr DeviceFixedScatterersCollection::get_dataset(size_t dset_idx) const {
    if (dset_idx >= m_fixed_datasets.size()) throw std::runtime_error("Illegal dataset index");
    return m_fixed_datasets[dset_idx];
}

DeviceSplineScatterers::DeviceSplineScatterers(bcsim::SplineScatterers::s_ptr host_scatterers, LogCallback log_callback_fn)
    : m_log_callback_fn(log_callback_fn)
{
    copy_information(host_scatterers);
    reallocate_device_memory();
    transfer_to_device(host_scatterers);
}

void DeviceSplineScatterers::copy_information(bcsim::SplineScatterers::s_ptr host_scatterers) {
    m_num_scatterers = host_scatterers->num_scatterers();
    m_num_cs         = host_scatterers->get_num_control_points();
    m_spline_degree  = host_scatterers->spline_degree;
    m_knots          = host_scatterers->knot_vector;

    const auto num_splines = host_scatterers->num_scatterers();
    if (num_splines == 0) {
        throw std::runtime_error("No scatterers");
    }

    if (m_spline_degree > MAX_SPLINE_DEGREE) {
        throw std::runtime_error("maximum spline degree supported is " + std::to_string(MAX_SPLINE_DEGREE));
    }
    m_log_callback_fn("Num spline scatterers: " + std::to_string(m_num_scatterers));
}

void DeviceSplineScatterers::reallocate_device_memory() {
    m_log_callback_fn("Allocating memory on host for reorganizing spline data");
    const auto num_bytes_xyz = m_num_cs*m_num_scatterers*sizeof(float);
    const auto num_bytes_amp = m_num_scatterers*sizeof(float);
    m_control_xs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes_xyz));
    m_control_ys = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes_xyz));
    m_control_zs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes_xyz));
    m_as         = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes_amp));
}

void DeviceSplineScatterers::transfer_to_device(bcsim::SplineScatterers::s_ptr host_scatterers) {
    // temporary host memory for restructuring to get correct device layout.
    const auto total_num_cs = m_num_cs*m_num_scatterers;
    const auto cs_num_bytes = total_num_cs*sizeof(float);

    std::vector<float> host_control_xs(total_num_cs);
    std::vector<float> host_control_ys(total_num_cs);
    std::vector<float> host_control_zs(total_num_cs);
        
    // there is only one amplitude for each scatterer.
    std::vector<float> host_control_as(m_num_scatterers); 

    for (size_t spline_no = 0; spline_no < m_num_scatterers; spline_no++) {
        host_control_as[spline_no] = host_scatterers->amplitudes[spline_no];
        for (size_t i = 0; i < m_num_cs; i++) {
            const size_t offset = spline_no + i*m_num_scatterers;
            host_control_xs[offset] = host_scatterers->control_points[spline_no][i].x;
            host_control_ys[offset] = host_scatterers->control_points[spline_no][i].y;
            host_control_zs[offset] = host_scatterers->control_points[spline_no][i].z;
        }
    }
    
    // copy control points to GPU memory.
    cudaErrorCheck( cudaMemcpy(m_control_xs->data(), host_control_xs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_control_ys->data(), host_control_ys.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    cudaErrorCheck( cudaMemcpy(m_control_zs->data(), host_control_zs.data(), cs_num_bytes, cudaMemcpyHostToDevice) );
    
    // copy amplitudes to GPU memory.
    const auto amplitudes_num_bytes = m_num_scatterers*sizeof(float);
    cudaErrorCheck( cudaMemcpy(m_as->data(), host_control_as.data(), amplitudes_num_bytes, cudaMemcpyHostToDevice) );
}

void DeviceSplineScatterers::update(bcsim::SplineScatterers::s_ptr host_scatterers) {
    throw std::runtime_error("TODO: implement");
    // Something along these lines:
    /*
    copy_information(host_scatterers);
    // after the constructor has run, the device buffers are always valid
    if (already allocated memory != needed)  {
        reallocate_device_memory();
    }
    transfer_to_device(host_scatterers);
    */
}

size_t DeviceSplineScatterers::get_num_scatterers() const {
    return m_num_scatterers;
}

float* DeviceSplineScatterers::get_xs_ptr() const {
    return m_control_xs->data();
}

float* DeviceSplineScatterers::get_ys_ptr() const {
    return m_control_ys->data();
}

float* DeviceSplineScatterers::get_zs_ptr() const {
    return m_control_zs->data();
}

float* DeviceSplineScatterers::get_as_ptr() const {
    return m_as->data();
}

std::vector<float> DeviceSplineScatterers::get_knots() const {
    return m_knots;
}

int DeviceSplineScatterers::get_num_cs() const {
    return m_num_cs;
}

int DeviceSplineScatterers::get_spline_degree() const {
    return m_spline_degree;
}


void DeviceSplineScatterersCollection::add(bcsim::SplineScatterers::s_ptr host_scatterers) {
    auto new_device_scatterers = std::make_shared<DeviceSplineScatterers>(host_scatterers);
    m_spline_datasets.push_back(new_device_scatterers);
}

void DeviceSplineScatterersCollection::update(bcsim::SplineScatterers::s_ptr host_scatterers, size_t dset_idx) {
    if (dset_idx >= m_spline_datasets.size()) {
        throw std::runtime_error("Illegal dataset index");        
    }
    m_spline_datasets[dset_idx]->update(host_scatterers);        
}

void DeviceSplineScatterersCollection::clear() {
    m_spline_datasets.clear();
}

size_t DeviceSplineScatterersCollection::get_total_num_scatterers() const {
    size_t sum = 0;
    for (size_t i = 0; i < m_spline_datasets.size(); i++) {
        sum += m_spline_datasets[i]->get_num_scatterers();
    }
    return sum;
}

size_t DeviceSplineScatterersCollection::get_num_datasets() const {
    return m_spline_datasets.size();
}

DeviceSplineScatterers::s_ptr DeviceSplineScatterersCollection::get_dataset(size_t dset_idx) const {
    if (dset_idx >= m_spline_datasets.size()) throw std::runtime_error("Illegal dataset index");
    return m_spline_datasets[dset_idx];
}

}   // end namespace
