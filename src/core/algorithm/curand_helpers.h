#pragma once
#include <curand.h>
#include <string>

#define curandErrorCheck(ans) { curandAssert((ans), __FILE__, __LINE__); }
inline void curandAssert(curandStatus_t code, const char* file, int line) {
    if (code != CURAND_STATUS_SUCCESS) {
        std::string err_str;
        switch(code) {
        case CURAND_STATUS_VERSION_MISMATCH:            err_str = "CURAND_STATUS_VERSION_MISMATCH";             break;
        case CURAND_STATUS_NOT_INITIALIZED:             err_str = "CURAND_STATUS_NOT_INITIALIZED";              break;
        case CURAND_STATUS_ALLOCATION_FAILED:           err_str = "CURAND_STATUS_ALLOCATION_FAILED";            break;
        case CURAND_STATUS_TYPE_ERROR:                  err_str = "CURAND_STATUS_TYPE_ERROR";                   break;
        case CURAND_STATUS_OUT_OF_RANGE:                err_str = "CURAND_STATUS_OUT_OF_RANGE";                 break;
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:         err_str = "CURAND_STATUS_LENGTH_NOT_MULTIPLE";          break;
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:   err_str = "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";    break;
        case CURAND_STATUS_LAUNCH_FAILURE:              err_str = "CURAND_STATUS_LAUNCH_FAILURE";               break;
        case CURAND_STATUS_PREEXISTING_FAILURE:         err_str = "CURAND_STATUS_PREEXISTING_FAILURE";          break;
        case CURAND_STATUS_INITIALIZATION_FAILED:       err_str = "CURAND_STATUS_INITIALIZATION_FAILED";        break;
        case CURAND_STATUS_ARCH_MISMATCH:               err_str = "CURAND_STATUS_ARCH_MISMATCH";                break;
        case CURAND_STATUS_INTERNAL_ERROR:              err_str = "CURAND_STATUS_INTERNAL_ERROR";               break;
        default:
            err_str = "UNKNOWN ERROR";
        }
    
        auto msg = std::string("cuRAND error: ")
                    + err_str
                    + std::string(", FILE: ")
                    + std::string(file)
                    + std::string(", LINE: ")
                    + std::to_string(line);
        throw std::runtime_error(msg);
    }
}


class CurandGeneratorRAII {
public:
    CurandGeneratorRAII() {

    }
    ~CurandGeneratorRAII() {

    }

};