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
#include <random>
#include <chrono>
#include <stdexcept>
#include <boost/program_options.hpp>
#include "../core/LibBCSim.hpp"
#include "../utils/GaussPulse.hpp"
#include "examples_common.hpp"

/*
 * Example usage of the C++ interface
 * 
 * This example tests the maximum number of simulated single RF lines
 * per second when using the fixed-scatterers GPU algorithm and when
 * scatterers are updated for each time step.
 */

void example(int argc, char** argv) {
    std::cout << "=== GPU example 1 ===" << std::endl;
    std::cout << "Single-line scanning using the fixed-scatterers GPU algorithm." << std::endl;
    std::cout << "It is possible to disable calls to set_scatterers() and simulate_lines()" << std::endl;
    std::cout << "for detailed performance analysis." << std::endl;

    // default values
    size_t num_scatterers = 1000000;
    float  num_seconds = 5.0;
    bool   enable_set_scatterers = true;
    bool   enable_simulate_lines = true;

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "show help message")
        ("num_scatterers", boost::program_options::value<size_t>(), "set number of scatterers")
        ("num_seconds", boost::program_options::value<float>(), "set simulation running time (longer time gives better timing accuracy)")
        ("set_scatterers", boost::program_options::value<bool>(), "upload scatterers to GPU each time step")
        ("simulate_lines", boost::program_options::value<bool>(), "simulate lines each time step")
    ;
    boost::program_options::variables_map var_map;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), var_map);
    if (var_map.count("help") != 0) {
        std::cout << desc << std::endl;
        return;
    }
    if (var_map.count("num_scatterers") != 0) {
        num_scatterers = var_map["num_scatterers"].as<size_t>();
    } else {
        std::cout << "Number of scatterers was not specified, using default value." << std::endl;    
    }
    if (var_map.count("num_seconds") != 0) {
        num_seconds = var_map["num_seconds"].as<float>();
    } else {
        std::cout << "Simulation time was not specified, using default value." << std::endl;
    }
    if (var_map.count("set_scatterers") != 0) {
        enable_set_scatterers = var_map["set_scatterers"].as<bool>();
    } 
    if (var_map.count("simulate_lines") != 0) {
        enable_simulate_lines = var_map["simulate_lines"].as<bool>();
    }

    std::cout << "Number of scatterers is " << num_scatterers << ".\n";
    std::cout << "Simulations will run for " << num_seconds << " seconds." << std::endl;
    std::cout << "Calls to set_scatterers() enabled?: " << enable_set_scatterers << std::endl;
    std::cout << "Calls to simulate_lines() enables?: " << enable_simulate_lines << std::endl;

    // create an instance of the fixed-scatterer GPU algorithm
    auto sim = bcsim::Create("gpu_fixed");
    sim->set_parameter("verbose", "0");
    
    // use an analytical Gaussian beam profile
    sim->set_analytical_profile(bcsim::IBeamProfile::s_ptr(new bcsim::GaussianBeamProfile(1e-3, 3e-3)));

    // configure the excitation signal
    const auto fs          = 100e6f;
    const auto center_freq = 2.5e6f;
    const auto frac_bw     = 0.2f;
    bcsim::ExcitationSignal ex;
    ex.sampling_frequency = 100e6;
    std::vector<float> dummy_times;
    bcsim::MakeGaussianExcitation(center_freq, frac_bw, ex.sampling_frequency, dummy_times, ex.samples, ex.center_index);
    ex.demod_freq = center_freq;
    sim->set_excitation(ex);

    // define sound speed
    sim->set_parameter("sound_speed", "1540.0");

    // configure a scan sequence consisting of a single RF line
    const auto line_length = 0.12f;
    auto scanseq = bcsim::ScanSequence::s_ptr(new bcsim::ScanSequence(line_length));
    const bcsim::vector3 origin(0.0f, 0.0f, 0.0f);
    const bcsim::vector3 direction(0.0f, 0.0f, 1.0f);
    const bcsim::vector3 lateral_dir(1.0f, 0.0f, 0.0f);
    const auto timestamp = 0.0;
    bcsim::Scanline scanline(origin, direction, lateral_dir, timestamp);
    scanseq->add_scanline(scanline);
    sim->set_scan_sequence(scanseq);

    // create random scatterers - confined to box with amplitudes in [-1.0, 1.0]
    auto fixed_scatterers = new bcsim::FixedScatterers;
    fixed_scatterers->scatterers.resize(num_scatterers);
    auto scatterers = bcsim::FixedScatterers::s_ptr(fixed_scatterers);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> x_dist(-0.03f, 0.03f);
    std::uniform_real_distribution<float> y_dist(-0.01f, 0.01f);
    std::uniform_real_distribution<float> z_dist(0.04f, 0.10f);
    std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
    for (size_t i = 0; i < num_scatterers; i++) {
        fixed_scatterers->scatterers[i].amplitude = a_dist(gen);
        fixed_scatterers->scatterers[i].pos = bcsim::vector3(x_dist(gen), y_dist(gen), z_dist(gen));
    }
    sim->add_fixed_scatterers(scatterers);
    std::cout << "Created scatterers\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    size_t num_beams = 0;
    std::cout << "Simulating...";
    float elapsed;
    for (;;) {
        // Reconfigure scatterers for each beam - even though they are equal,
        // a host->device PCI express transfer will be triggered.
        // This can be ragarded as sort of the best case where updated
        // scatterers are available at no additional computational cost on
        // the CPU side.
        if (enable_set_scatterers) {
            sim->clear_fixed_scatterers();
            sim->add_fixed_scatterers(scatterers);
        }
        
        std::vector<std::vector<std::complex<float>>> sim_res;
        if (enable_simulate_lines) {
            sim->simulate_lines(sim_res);
        }
        num_beams++;

        // done?
        auto temp = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(temp-start).count()/1000.0;
        if (elapsed >= num_seconds) {
            dump_rf_line("GpuExample1_OUTPUT.txt", sim_res[0]);
            break;
        }
    }
    std::cout << "Done. Processed " << num_beams << " in " << elapsed << " seconds.\n";
    const auto prf = num_beams / elapsed;
    std::cout << "Achieved a PRF of " << prf << " Hz.\n";
    
}

int main(int argc, char** argv) {
    try {
        example(argc, argv);
    } catch (std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }

    return 0;
}