#pragma once
#include <memory>
#include "core/BCSimConfig.hpp"

namespace default_phantoms {

struct LeftVentriclePhantomParameters {
    // TODO
};

bcsim::Scatterers::u_ptr CreateLeftVentricle3dPhantom(const LeftVentriclePhantomParameters& type);

}   // end namespace
