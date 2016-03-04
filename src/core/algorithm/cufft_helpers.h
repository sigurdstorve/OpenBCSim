#pragma once
#include <string>
#include <stdexcept>
#include <memory>
#include <cufft.h>

#define cufftErrorCheck(ans) { cufftAssert((ans), __FILE__, __LINE__); }
inline void cufftAssert(cufftResult_t code, const char* file, int line) {
    if (code != CUFFT_SUCCESS) {
        std::string err_str;
        switch(code) {
        case CUFFT_INVALID_PLAN:              err_str = "CUFFT_INVALID_PLAN";              break;
        case CUFFT_ALLOC_FAILED:              err_str = "CUFFT_ALLOC_FAILED";              break;
        case CUFFT_INVALID_TYPE:              err_str = "CUFFT_INVALID_TYPE";              break;
        case CUFFT_INVALID_VALUE:             err_str = "CUFFT_INVALID_VALUE";             break;
        case CUFFT_INTERNAL_ERROR:            err_str = "CUFFT_INTERNAL_ERROR";            break;
        case CUFFT_EXEC_FAILED:               err_str = "CUFFT_EXEC_FAILED";               break;
        case CUFFT_SETUP_FAILED:              err_str = "CUFFT_SETUP_FAILED";              break;
        case CUFFT_INVALID_SIZE:              err_str = "CUFFT_INVALID_SIZE";              break;
        case CUFFT_UNALIGNED_DATA:            err_str = "CUFFT_UNALIGNED_DATA";            break;
        case CUFFT_INCOMPLETE_PARAMETER_LIST: err_str = "CUFFT_INCOMPLETE_PARAMETER_LIST"; break;
        case CUFFT_INVALID_DEVICE:            err_str = "CUFFT_INVALID_DEVICE";            break;
        case CUFFT_PARSE_ERROR:               err_str = "CUFFT_PARSE_ERROR";               break;
        case CUFFT_NO_WORKSPACE:              err_str = "CUFFT_NO_WORKSPACE";              break;
        case CUFFT_NOT_IMPLEMENTED:           err_str = "CUFFT_NOT_IMPLEMENTED";           break;
        case CUFFT_LICENSE_ERROR:             err_str = "CUFFT_LICENSE_ERROR";             break;
        default:
            err_str = "UNKNOWN ERROR";
        }

        auto msg = std::string("cuFFT error: ")
                    + err_str
                    + std::string(", FILE: ")
                    + std::string(file)
                    + std::string(", LINE: ")
                    + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

class CufftPlanRAII {
public:
    typedef std::unique_ptr<CufftPlanRAII> u_ptr;
    typedef std::shared_ptr<CufftPlanRAII> s_ptr;

    CufftPlanRAII(int nx, cufftType type, int batch) {
        cufftPlan1d(&plan, nx, type, batch);
    }

    ~CufftPlanRAII() {
        cufftDestroy(plan);
    }

    cufftHandle get() {
        return plan;
    }

private:
    cufftHandle plan;
};

class CufftBatchedPlanRAII {
public:
    typedef std::unique_ptr<CufftBatchedPlanRAII> u_ptr;

    CufftBatchedPlanRAII(int rank, int* dims, int num_samples, cufftType type, int batch) {
        cufftErrorCheck(cufftPlanMany(&plan, rank, dims, NULL, 1, num_samples, NULL, 1, num_samples, type, batch));
    }
    ~CufftBatchedPlanRAII() {
        cufftDestroy(plan);
    }

    cufftHandle get() {
        return plan;
    }
private:
    cufftHandle plan;
};

