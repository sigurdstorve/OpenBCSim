#include <memory>
#include <vector>
#include "../LibBCSim.hpp"
#include "cuda_helpers.h"

namespace bcsim {

// forward-decl.
class DeviceSplineScatterersCollection;

// Device memory for a fixed-scatterer dataset.
class DeviceFixedScatterers {
public:
    typedef std::shared_ptr<DeviceFixedScatterers> s_ptr;

    // Allocate space for a new dataset with num_scatterers scatterers.
    explicit DeviceFixedScatterers(size_t num_scatterers);
    
    size_t get_num_scatterers() const;

    float* get_xs_ptr() const;

    float* get_ys_ptr() const;

    float* get_zs_ptr() const;

    float* get_as_ptr() const;

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
    void add(bcsim::FixedScatterers::s_ptr host_scatterers);

    // reset and make fixed scatterer datasets from evaluating all spline datasets
    void render(const DeviceSplineScatterersCollection& spline_datasets, float timestamp);

    void transfer_to_device(bcsim::FixedScatterers::s_ptr host_scatterers,
                            DeviceFixedScatterers::s_ptr device_scatterers);

    // update an existing dataset (will only reallocate memory if the new size is
    // different from the previous)
    void update(bcsim::FixedScatterers::s_ptr host_scatterers, size_t dset_idx);

    void clear();

    size_t get_total_num_scatterers() const;

    size_t get_num_datasets() const;

    DeviceFixedScatterers::s_ptr get_dataset(size_t dset_idx) const;

private:
    std::vector<DeviceFixedScatterers::s_ptr>   m_fixed_datasets;
};

// Device memory for a spline-scatterer dataset.
class DeviceSplineScatterers {
public:
    typedef std::shared_ptr<DeviceSplineScatterers> s_ptr;

    DeviceSplineScatterers(bcsim::SplineScatterers::s_ptr host_scatterers);

    // copy everything actual control points.
    void copy_information(bcsim::SplineScatterers::s_ptr host_scatterers);

    // reallocate device memory to hold current dataset
    void reallocate_device_memory();

    // copy data from host datastructure to the device memory
    void transfer_to_device(bcsim::SplineScatterers::s_ptr host_scatterers);

    // update an existing dataset (will only reallocate memory if the new size is different from the previous)
    void update(bcsim::SplineScatterers::s_ptr host_scatterers);

    size_t get_num_scatterers() const;

    float* get_xs_ptr() const;

    float* get_ys_ptr() const;

    float* get_zs_ptr() const;

    float* get_as_ptr() const;

    std::vector<float> get_knots() const;

    int get_num_cs() const;

    int get_spline_degree() const;

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
    void add(bcsim::SplineScatterers::s_ptr host_scatterers);

    void update(bcsim::SplineScatterers::s_ptr host_scatterers, size_t dset_idx);

    void clear();

    size_t get_total_num_scatterers() const;

    size_t get_num_datasets() const;

    DeviceSplineScatterers::s_ptr get_dataset(size_t dset_idx) const;

private:
    std::vector<DeviceSplineScatterers::s_ptr> m_spline_datasets;
};

}   // end namespace
