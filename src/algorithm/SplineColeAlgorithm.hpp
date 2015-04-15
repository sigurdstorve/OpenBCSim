#pragma once
#include <vector>
#include "cole_defines.h"
#include "BaseColeAlgorithm.hpp"
#include "ColeConfig.hpp"

namespace cole {

// Specialization of the COLE algorithm for using a set of scatterers
// following spline trajectories.
class SplineColeAlgorithm : public BaseColeAlgorithm {
public:
    SplineColeAlgorithm();
    
    virtual void set_scatterers(ColeScatterers::s_ptr new_scatterers);

protected:
    virtual void projection_loop(const Scanline& line, std::vector<cole_float>& time_proj_signal);

private:
    ColeSplineScatterers::s_ptr    m_scatterers;

};


}   // namespace
