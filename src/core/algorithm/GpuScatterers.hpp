#include <memory>
#include <vector>
#include <stdexcept>
#include "cuda_helpers.h"
#include "common_definitions.h" // for MAX_SPLINE_DEGREE

namespace bcsim {

// Device memory for a fixed-scatterer dataset.
class DeviceFixedScatterers {
public:
    typedef std::shared_ptr<DeviceFixedScatterers> s_ptr;

    // Allocate space for a new dataset with num_scatterers scatterers.
    explicit DeviceFixedScatterers(size_t num_scatterers)
        : m_num_scatterers(num_scatterers)
    {
        const auto num_bytes = num_scatterers*sizeof(float);
        xs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes));
        ys = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes));
        zs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes));
        as = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes));
    }
    
    size_t get_num_scatterers() const {
        return m_num_scatterers;
    }

    float* get_xs_ptr() const {
        return xs->data();
    }

    float* get_ys_ptr() const {
        return ys->data();
    }

    float* get_zs_ptr() const {
        return zs->data();
    }

    float* get_as_ptr() const {
        return as->data();
    }

private:
    DeviceBufferRAII<float>::u_ptr xs;
    DeviceBufferRAII<float>::u_ptr ys;
    DeviceBufferRAII<float>::u_ptr zs;
    DeviceBufferRAII<float>::u_ptr as;

    size_t m_num_scatterers;
};

// Device memory for multiple fixed-scatterer datasets
class DeviceFixedScatterersCollection {
public:
    // create a new dataset and fill it with data (will allocate memory on device)
    void add(bcsim::FixedScatterers::s_ptr host_scatterers) {
        auto new_device_scatterers = std::make_shared<DeviceFixedScatterers>(host_scatterers->num_scatterers());
        transfer_to_device(host_scatterers, new_device_scatterers);
        m_fixed_datasets.push_back(new_device_scatterers);
    }

    void transfer_to_device(bcsim::FixedScatterers::s_ptr host_scatterers,
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

    // update an existing dataset (will only reallocate memory if the new size is
    // different from the previous)
    void update(bcsim::FixedScatterers::s_ptr host_scatterers, size_t dset_idx) {
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

    void clear() {
        m_fixed_datasets.clear();
    }

    size_t get_total_num_scatterers() const {
        size_t sum = 0;
        for (size_t i = 0; i < m_fixed_datasets.size(); i++) {
            sum += m_fixed_datasets[i]->get_num_scatterers();
        }
        return sum;
    }

    size_t get_num_datasets() const {
        return m_fixed_datasets.size();
    }

    DeviceFixedScatterers::s_ptr get_dataset(size_t dset_idx) const {
        if (dset_idx >= m_fixed_datasets.size()) throw std::runtime_error("Illegal dataset index");
        return m_fixed_datasets[dset_idx];
    }

private:
    std::vector<DeviceFixedScatterers::s_ptr>   m_fixed_datasets;
};

// Device memory for a spline-scatterer dataset.
class DeviceSplineScatterers {
public:
    typedef std::shared_ptr<DeviceSplineScatterers> s_ptr;

    DeviceSplineScatterers(bcsim::SplineScatterers::s_ptr host_scatterers) {
        copy_information(host_scatterers);
        reallocate_device_memory();
        transfer_to_device(host_scatterers);
    }

    // copy everything actual control points.
    void copy_information(bcsim::SplineScatterers::s_ptr host_scatterers) {
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
        std::cout << "Num spline scatterers: " << m_num_scatterers << std::endl;
    }

    // reallocate device memory to hold current dataset
    void reallocate_device_memory() {
        std::cout << "Allocating memory on host for reorganizing spline data\n";
        const auto num_bytes_xyz = m_num_cs*m_num_scatterers*sizeof(float);
        const auto num_bytes_amp = m_num_scatterers*sizeof(float);
        m_control_xs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes_xyz));
        m_control_ys = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes_xyz));
        m_control_zs = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes_xyz));
        m_as         = DeviceBufferRAII<float>::u_ptr(new DeviceBufferRAII<float>(num_bytes_amp));
    }

    // copy data from host datastructure to the device memory
    void transfer_to_device(bcsim::SplineScatterers::s_ptr host_scatterers) {
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

    // update an existing dataset (will only reallocate memory if the new size is different from the previous)
    void update(bcsim::SplineScatterers::s_ptr host_scatterers) {
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

    size_t get_num_scatterers() const {
        return m_num_scatterers;
    }

    float* get_xs_ptr() const {
        return m_control_xs->data();
    }

    float* get_ys_ptr() const {
        return m_control_ys->data();
    }

    float* get_zs_ptr() const {
        return m_control_zs->data();
    }

    float* get_as_ptr() const {
        return m_as->data();
    }

    std::vector<float> get_knots() const {
        return m_knots;
    }

    int get_num_cs() const {
        return m_num_cs;
    }

    int get_spline_degree() const {
        return m_spline_degree;
    }

private:
    // the number of spline-scatterers
    size_t                              m_num_scatterers;

    // the knot vector common for all splines
    std::vector<float>                  m_knots;
    
    // the number of spline control points
    int                                 m_num_cs;
    
    // the degree of the spline curve.
    int                                 m_spline_degree;

    // x,y,z coordinates: a fixed number of control points for each scatterer
    DeviceBufferRAII<float>::u_ptr      m_control_xs;
    DeviceBufferRAII<float>::u_ptr      m_control_ys;
    DeviceBufferRAII<float>::u_ptr      m_control_zs;
    
    // amplitudes: one per spline scatterer.
    DeviceBufferRAII<float>::u_ptr      m_as;
};

// Device memory for multiple spline-scatterers datasets
class DeviceSplineScatterersCollection {
public:
    void add(bcsim::SplineScatterers::s_ptr host_scatterers) {
        auto new_device_scatterers = std::make_shared<DeviceSplineScatterers>(host_scatterers);
        m_spline_datasets.push_back(new_device_scatterers);
    }

    void update(bcsim::SplineScatterers::s_ptr host_scatterers, size_t dset_idx) {
        if (dset_idx >= m_spline_datasets.size()) {
            throw std::runtime_error("Illegal dataset index");        
        }
        m_spline_datasets[dset_idx]->update(host_scatterers);        
    }

    void clear() {
        m_spline_datasets.clear();
    }

    size_t get_total_num_scatterers() const {
        size_t sum = 0;
        for (size_t i = 0; i < m_spline_datasets.size(); i++) {
            sum += m_spline_datasets[i]->get_num_scatterers();
        }
        return sum;
    }

    size_t get_num_datasets() const {
        return m_spline_datasets.size();
    }

    DeviceSplineScatterers::s_ptr get_dataset(size_t dset_idx) const {
        if (dset_idx >= m_spline_datasets.size()) throw std::runtime_error("Illegal dataset index");
        return m_spline_datasets[dset_idx];
    }

private:
    std::vector<DeviceSplineScatterers::s_ptr> m_spline_datasets;
};

}   // end namespace
