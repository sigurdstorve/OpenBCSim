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
#include "LibBCSim.hpp"
#include "GaussPulse.hpp"

/*
 * Example usage of the C++ interface
 * 
 * This example tests the maximum number of simulated single RF lines
 * per second when using the spline-catterers GPU algorithm where there
 * is no need to update the scatterers at each time step.
 */

void example(int argc, char** argv) {
    std::cout << "=== GPU example 1 ===" << std::endl;
    std::cout << "Single-line scanning using the spline-scatterers GPU algorithm." << std::endl;

    // default values
    size_t num_scatterers = 1000000;
    float  num_seconds = 5.0;
    size_t num_beams = 512; // num beams to simulate at a time.

    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "show help message")
        ("num_scatterers", boost::program_options::value<size_t>(), "set number of scatterers")
        ("num_seconds", boost::program_options::value<float>(), "set simulation running time (longer time gives better timing accuracy)")
        ("num_beams", boost::program_options::value<size_t>(), "set number of beams in each packet")
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
    if (var_map.count("num_beams") != 0) {
        num_beams = var_map["num_beams"].as<size_t>();
    } else {
        std::cout << "Number of beams in each packet not specified, using default value." << std::endl;
    }

    std::cout << "Number of scatterers is " << num_scatterers << ".\n";
    std::cout << "Number of beams in each packet is " << num_beams << ".\n";
    std::cout << "Simulations will run for " << num_seconds << " seconds." << std::endl;

    // create an instance of the fixed-scatterer GPU algorithm
    auto sim = bcsim::Create("gpu_spline2");
    sim->set_verbose(false);
    
    // use an analytical Gaussian beam profile
    sim->set_beam_profile(bcsim::IBeamProfile::s_ptr(new bcsim::GaussianBeamProfile(1e-3, 3e-3)));

    // configure the excitation signal
    const auto fs          = 100e6f;
    const auto center_freq = 2.5e6f;
    const auto frac_bw     = 0.2f;
    bcsim::ExcitationSignal ex;
    std::vector<float> dummy_times;
    bcsim::MakeGaussianExcitation(center_freq, frac_bw, ex.sampling_frequency, dummy_times, ex.samples, ex.center_index);
    sim->set_excitation(ex);

    // configure output type (currently noe effect)
    sim->set_output_type("rf");

    // configure sound speed
    sim->set_parameter("sound_speed", "1540.0");

    // configure a scan sequence consisting of a single RF line
    const auto line_length = 0.12f;
    auto scanseq = bcsim::ScanSequence::s_ptr(new bcsim::ScanSequence(line_length));
    const bcsim::vector3 origin(0.0f, 0.0f, 0.0f);
    const bcsim::vector3 direction(0.0f, 0.0f, 1.0f);
    const bcsim::vector3 lateral_dir(1.0f, 0.0f, 0.0f);
    for (size_t beam_no = 0; beam_no < num_beams; beam_no++) {
        const auto timestamp = beam_no / static_cast<float>(beam_no);
        bcsim::Scanline scanline(origin, direction, lateral_dir, timestamp);
        scanseq->add_scanline(scanline);
    }
    sim->set_scan_sequence(scanseq);

    // create random scatterers - confined to box with amplitudes in [-1.0, 1.0]
    auto spline_scatterers = new bcsim::SplineScatterers;
    spline_scatterers->spline_degree = 3;
    int num_cs = 10;
    const auto num_knots = spline_scatterers->spline_degree + num_cs + 1;
    spline_scatterers->knot_vector.resize(num_knots);
    // create a dummy, but valid, knot vector
    for (size_t i = 0; i < num_knots; i++) {
        spline_scatterers->knot_vector[i] = i/static_cast<float>(num_knots);
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> x_dist(-0.03f, 0.03f);
    std::uniform_real_distribution<float> y_dist(-0.01f, 0.01f);
    std::uniform_real_distribution<float> z_dist(0.04f, 0.10f);
    std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
    spline_scatterers->nodes.clear();
    for (size_t scatterer_no = 0; scatterer_no < num_scatterers; scatterer_no++) {
        std::vector<bcsim::PointScatterer> control_points;
        for (size_t i = 0; i < num_cs; i++) {
            bcsim::PointScatterer scatterer;
            scatterer.amplitude = a_dist(gen);
            scatterer.pos = bcsim::vector3(x_dist(gen), y_dist(gen), z_dist(gen));
            control_points.push_back(scatterer);
        }
        spline_scatterers->nodes.push_back(control_points);
    }

    auto scatterers = bcsim::Scatterers::s_ptr(spline_scatterers);
    sim->set_scatterers(scatterers);
    std::cout << "Created scatterers\n";
    
    auto start = std::chrono::high_resolution_clock::now();
    size_t num_simulate_lines = 0;
    std::cout << "Simulating.";
    float elapsed;
    for (;;) {
        std::cout << ".";
        std::vector<std::vector<bc_float>> sim_res;
        sim->simulate_lines(sim_res);
        num_simulate_lines++;

        // done?
        auto temp = std::chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(temp-start).count()/1000.0;
        if (elapsed >= num_seconds) break;
    }
    size_t total_num_beams = num_simulate_lines*num_beams;
    std::cout << "Done. Processed " << total_num_beams << " in " << elapsed << " seconds.\n";
    const auto prf = total_num_beams / elapsed;
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