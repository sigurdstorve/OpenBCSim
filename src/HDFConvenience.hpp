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

#pragma once
#include <memory>
#include <string>
#include "export_macros.hpp"
#include "bcsim_defines.h"
#include "LibBCSim.hpp"

// Two layers of convenience functions:
// 1. Load from Hdf5
// 2. Configure a simulator object with data from Hdf5 files.

namespace bcsim {

// Create a simulator object from a single HDF5 containing all required data.
IAlgorithm::s_ptr DLL_PUBLIC CreateSimulator(const std::string& config_file,
                                             std::string sim_type="");

// Create a simulator object from HDF5 data files.
IAlgorithm::s_ptr DLL_PUBLIC CreateSimulator(const std::string& scattererFile,
                                             const std::string& scanseqFile,
                                             const std::string& excitationFile,
                                             std::string sim_type="");

// Will auto-detect if the scatterers dataset is a spline or fixed dataset.
// Returns "fixed" or "spline".
// Throws on error.
std::string DLL_PUBLIC AutodetectScatteresType(const std::string& h5_file);

// Specific loader for fixed scatterers
Scatterers::s_ptr DLL_PUBLIC loadFixedScatterersFromHdf(const std::string& h5_file);

// Specific loader for spline scatterers
Scatterers::s_ptr DLL_PUBLIC loadSplineScatterersFromHdf(const std::string& h5_file);

void DLL_PUBLIC setFixedScatterersFromHdf(IAlgorithm::s_ptr sim, const std::string& h5_file);

void DLL_PUBLIC setSplineScatterersFromHdf(IAlgorithm::s_ptr sim, const std::string& h5_file);

ScanSequence::u_ptr DLL_PUBLIC loadScanSequenceFromHdf(const std::string& h5_file);

void DLL_PUBLIC setScanSequenceFromHdf(IAlgorithm::s_ptr sim, const std::string& h5_file);

ExcitationSignal DLL_PUBLIC loadExcitationFromHdf(const std::string& h5_file);

void DLL_PUBLIC setExcitationFromHdf(IAlgorithm::s_ptr sim, const std::string& h5_file);

// LUT beam profile
IBeamProfile::s_ptr loadBeamProfileFromHdf(const std::string& h5_file);

// LUT beam profile
void DLL_PUBLIC setBeamProfileFromHdf(IAlgorithm::s_ptr sim, const std::string& h5_file);

}   // namespace

