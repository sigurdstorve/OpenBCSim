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
#include <fstream>
#include <iomanip>
#include <memory>
#include "BeamProfile.hpp"
#include "HDFConvenience.hpp"
#include "BCSimConfig.hpp"
#include "LibBCSim.hpp"
#include "export_macros.hpp"

using namespace bcsim;

void print_sizes() {
    std::cout << "=== size of common data structures in bytes ===\n";
    std::cout << "sizeof(Interval): " << sizeof(Interval) << std::endl;
    std::cout << "sizeof(PointScatterer): " << sizeof(PointScatterer) << std::endl;
    std::cout << "sizeof(vector3): " << sizeof(vector3) << std::endl;
}

void printHelpAndExit(char **argv) {
    std::cout << "Usage: " << argv[0] << " <config HDF5 file> [loop]" << std::endl;
    std::exit(0);
}

void dumpRfFrame(std::string filename, std::vector<std::vector<bc_float> > rfLines, bool envDetect) {
    auto numLines = rfLines.size();
    auto numSamples = rfLines[0].size();
    std::ofstream outfile;
    outfile.open(filename.c_str());

    outfile << numLines << " ";
    outfile << numSamples << " ";
    for (size_t lineNo = 0; lineNo < numLines; lineNo++) {
        for (size_t sampleNo = 0; sampleNo < numSamples; sampleNo++) {
            outfile << std::setprecision(16) << rfLines[lineNo][sampleNo] << " ";   
        }
    }
    outfile.close();
}

int main(int argc, char **argv) {
    std::cout << "=== Simple demo program for OpenBCSim ===" << std::endl;
    std::cout << "Assumes that all setup and parameters comes from one HDF5 file" << std::endl;
    std::cout << "TO BE DONE: Replace plaintext output with HDF5.\n";
    std::cout << "TO BE DONE: Load beam profile from HDF5.\n";
    std::cout << std::endl;

    print_sizes();

    if (argc < 2) {
        printHelpAndExit(argv);
    }
    bool runLoop = false;
    if (argc == 3) {
        runLoop = true;
    }
    
    std::string config_file = argv[1];
        
    // Create the simulator object through HDF convenience layer.
    IAlgorithm::s_ptr simulator;
    try {
        simulator = CreateSimulator(config_file);
    } catch (std::exception& e) {
        std::cout << "Caught exception when creating simulator instance: " << e.what() << std::endl;
        exit(0);
    }
    simulator->set_parameter("verbose", "1");

    // Create the beam profile object to use for all lines.
    // Either analytical or lookup table.
    IBeamProfile::s_ptr beam_profile;
    beam_profile = IBeamProfile::s_ptr(new GaussianBeamProfile(0.2e-3, 0.2e-3));
    simulator->set_analytical_profile(beam_profile);
    
    try {
        if (runLoop) {
            // Run forever to verify full CPU load.
            int counter = 0;
            while (1) {
                std::vector<std::vector<std::complex<bc_float>> > rfFrame;    
                std::cout << counter++ << std::endl;
                simulator->simulate_lines(rfFrame);
            }
        } else {
            // Simulate one frame and dump to disk.
            std::vector<std::vector<std::complex<bc_float>> > rfFrame;    
            simulator->simulate_lines(rfFrame);
            //std::string outfile = "frame.rf";
            //dumpRfFrame(outfile, rfFrame, true);
            //std::cout << "RF frame dumped to " << outfile << std::endl;
        }
    } catch (std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
    
}
 
