#include <iostream>
#include <complex>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cuda.h>
#include <cufft.h>
#include "../core/algorithm/cuda_helpers.h"
#include "../core/algorithm/cufft_helpers.h"

// Test program to measure the difference in speed between doing
// multiple 1D FFTs vs. batched.


// multiple 1D forward FFTs in a loop.
std::vector<std::complex<float>> compute_multiple(DeviceBufferRAII<cufftComplex>::s_ptr device_in, int num_samples, int batch_size, int num_rep) {
    const auto total_num_samples = num_samples*batch_size;
    const auto total_num_bytes = device_in->get_num_bytes();

    DeviceBufferRAII<cufftComplex> device_out(total_num_bytes);

    // clear the device output memory
    std::vector<std::complex<float>> zeros(total_num_samples, 0.0f);
    cudaErrorCheck(cudaMemcpy(device_out.data(), zeros.data(), total_num_bytes, cudaMemcpyHostToDevice));

    cufftHandle plan;
    cufftErrorCheck(cufftPlan1d(&plan, num_samples, CUFFT_C2C, 1));

    auto begin = std::chrono::high_resolution_clock::now();

    for (int rep_no = 0; rep_no < num_rep; rep_no++) {
        for (int batch_no = 0; batch_no < batch_size; batch_no++) {
            const size_t offset = batch_no*num_samples;
            cufftErrorCheck(cufftExecC2C(plan, device_in->data()+offset, device_out.data()+offset, CUFFT_FORWARD));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
    const auto ms_per_fft = static_cast<float>(duration_ms)/(num_rep*batch_size);
    std::cout << "compute_multiple(): " << ms_per_fft << " milliseconds per FFT.\n";

    // copy output data from the GPU
    std::vector<std::complex<float>> output(total_num_samples);
    cudaErrorCheck(cudaMemcpy(output.data(), device_out.data(), total_num_bytes, cudaMemcpyDeviceToHost));

    cufftErrorCheck(cufftDestroy(plan));
    return output;
}

// multiple 1D forward FFs batched together in one operation.
std::vector<std::complex<float>> compute_batched(DeviceBufferRAII<cufftComplex>::s_ptr device_in, int num_samples, int batch_size, int num_rep) {
    const auto total_num_samples = num_samples*batch_size;
    const auto total_num_bytes = device_in->get_num_bytes();

    DeviceBufferRAII<cufftComplex> device_out(total_num_bytes);

    // clear the device output memory
    std::vector<std::complex<float>> zeros(total_num_samples, 0.0f);
    cudaErrorCheck(cudaMemcpy(device_out.data(), zeros.data(), total_num_bytes, cudaMemcpyHostToDevice));
    
    cufftHandle plan;
    int rank = 1;
    int dims[] = {num_samples};
    cufftErrorCheck(cufftPlanMany(&plan, rank, dims, NULL, 1, num_samples, NULL, 1, num_samples, CUFFT_C2C, batch_size));
    
    auto begin = std::chrono::high_resolution_clock::now();
    for (int rep_no = 0; rep_no < num_rep; rep_no++) {
        cufftErrorCheck(cufftExecC2C(plan, device_in->data(), device_out.data(), CUFFT_FORWARD));
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
    const auto ms_per_fft = static_cast<float>(duration_ms)/(num_rep*batch_size);
    std::cout << "compute_batched(): " << ms_per_fft << " milliseconds per FFT.\n";

    // copy output data from the GPU
    std::vector<std::complex<float>> output(total_num_samples);
    cudaErrorCheck(cudaMemcpy(output.data(), device_out.data(), total_num_bytes, cudaMemcpyDeviceToHost));
    
    cufftErrorCheck(cufftDestroy(plan));

    return output;
}

int main(int argc, char** argv) {
    std::cout << "Test program for timing multiple single-FFTs vs. batched.\n";
    int num_samples;    // number of samples in each FFT
    int batch_size;     // number of FFTs in each batch
    int num_rep;        // number of repetitions of the batch [for accurate timing]
    if (argc == 1) {
        num_samples = 16384;
        batch_size = 128;
        num_rep = 1000;
    } else if (argc == 4) {
        num_samples = std::stoi(argv[1]);
        batch_size  = std::stoi(argv[2]);
        num_rep     = std::stoi(argv[3]);
    } else {
        std::cout << "Usage: " << argv[0] << " <num_samples> <batch_size> <num_rep>\n";
        return 0;
    }
    std::cout << "num_samples: " << num_samples << std::endl;
    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "num_rep: " << num_rep << std::endl;
    
    const auto total_num_samples = num_samples*batch_size;
    std::vector<std::complex<float>> samples(total_num_samples);

    // fill with random complex numbers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (size_t i = 0; i < total_num_samples; i++) {
        samples[i] = std::complex<float>(dis(gen), dis(gen));
    }
    
    // copy input data to the GPU
    const auto total_num_bytes = total_num_samples*sizeof(cufftComplex);
    
    auto device_in = std::make_shared<DeviceBufferRAII<cufftComplex>>(total_num_bytes);
    cudaErrorCheck(cudaMemcpy(device_in->data(), samples.data(), total_num_bytes, cudaMemcpyHostToDevice));

    const auto res1 = compute_multiple(device_in, num_samples, batch_size, num_rep);
    const auto res2 = compute_batched(device_in, num_samples, batch_size, num_rep);

    // check equality
    if (res1.size() != res2.size()) throw std::logic_error("mismatch in number of samples");
    for (size_t i = 0; i < res1.size(); i++) {
        const auto diff = std::abs(res1[i]-res2[i]);
        if (diff > 1e-6) {
            std::cout << i  << " : " << res1[i] << " vs. " << res2[i] << std::endl;
            throw std::runtime_error("error exceeds threshold");
        }
    }
}