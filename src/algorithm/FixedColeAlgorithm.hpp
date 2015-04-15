#pragma once
#include <vector>
#include "cole_defines.h"
#include "BaseColeAlgorithm.hpp"
#include "ColeConfig.hpp"

namespace cole {

// Specialization of the COLE algorithm for using a set of fixed
// point scatterers.
class FixedColeAlgorithm : public BaseColeAlgorithm {
public:
    FixedColeAlgorithm();
    
    virtual void set_scatterers(ColeScatterers::s_ptr new_scatterers);

protected:    
    virtual void projection_loop(const Scanline& line, std::vector<cole_float>& time_proj_signal);

private:
    ColeFixedScatterers::s_ptr     m_scatterers;

};

}   // namespace
