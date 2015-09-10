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
#include <stdexcept>
#include <vector>
#ifdef _MSC_VER
#   pragma warning(push)
#   pragma warning(disable:4251) // needs to have dll-interface...
#endif
#include <H5Cpp.h>
#ifdef _MSC_VER
#   pragma warning(pop)
#endif
#include <boost/multi_array.hpp>

namespace SimpleHDF {

namespace detail {
    // Template trick for typemapping from C++ data types to HDF5 data type
    // through templates:
    // http://stackoverflow.com/questions/9250237/write-a-boostmulti-array-to-hdf5-dataset
    template <typename T>
    inline H5::DataType EquivH5Type() {
        throw std::runtime_error("Illegal type.");
    }
    template<>
    inline H5::DataType EquivH5Type<unsigned char>() {
        return H5::PredType::NATIVE_UCHAR;
    }
    template<>
    inline H5::DataType EquivH5Type<int>() {
        return H5::PredType::NATIVE_INT;
    }
    template<>
    inline H5::DataType EquivH5Type<float>() {
        return H5::PredType::NATIVE_FLOAT;
    }
    template<>
    inline H5::DataType EquivH5Type<double>() {
        return H5::PredType::NATIVE_DOUBLE;
    }
}

// Convenice class for using HDF5 functionality.
class SimpleHDF5Reader {
public:
    // Open a HDF5 file for reading.
    SimpleHDF5Reader(const std::string& filename) {
        H5::Exception::dontPrint();
        try {
            hdf5_file = H5::H5File(filename, H5F_ACC_RDONLY);
        } catch (...) {
            throwRuntimeError(std::string("Unable to load file: ") + filename);
        }
    }
    
    // Read a scalar value.
    template <typename T>
    T readScalar(const std::string & dataSetName) {
        const H5::DataSet & dataSet = hdf5_file.openDataSet(dataSetName);
        T res;
        try {
            dataSet.read(&res, detail::EquivH5Type<T>());
            return res;
        } catch (...) {
            throwRuntimeError("Error reading scalar.");
        }
    }
        
    // Read a std::vector of elements, uses readMultiArray internally.
    template <typename T>
    std::vector<T> readStdVector(const std::string & dataSetName) {
        auto v = readMultiArray<T, 1>(dataSetName);
        auto size = v.shape()[0];
        std::vector<T> res(size);
        for (size_t i = 0; i < size; i++) {
            res[i] = v[i];
        }
        return res;
    }
    
    template<typename T, int N>
    boost::multi_array<T, N> readMultiArray(const std::string& dataset_name) {
        H5::DataSet dataset = hdf5_file.openDataSet(dataset_name);
        typedef boost::multi_array<T, N> array_type;
        array_type res;
        try {
            std::vector<int> temp_dimensions = getDimensions(dataset);
            int rank = static_cast<int>(temp_dimensions.size());
            if (rank != N) {
                throw std::runtime_error("Rank mistmatch");
            }

            boost::array<int, N> dimensions;
            for (int dim = 0; dim < N; dim++) {
                dimensions[dim] = temp_dimensions[dim];
            }

            //res.resize(boost::array<array_type::index, N>(dimensions));
            res.resize(dimensions);
            dataset.read(res.data(), detail::EquivH5Type<T>());
        } catch (...) {
            throwRuntimeError("Unable to read N-dim array");
        }
        return res;
    }

    // Get the dimensions of a data set as a a std::vector of ints.
    static std::vector<int> getDimensions(const H5::DataSet & dataSet) {
        H5::DataSpace dataSpace = dataSet.getSpace();
        if (!dataSpace.isSimple()) {
            throwRuntimeError("Complex data space is not supported.");
        }
        int ndim = (int)dataSpace.getSimpleExtentNdims();
        // Get a vector of dimensions
        hsize_t *dims = new hsize_t[ndim];
        dataSpace.getSimpleExtentDims(dims, NULL);
        // Copy over to a std::vector
        std::vector<int> res;
        res.resize(ndim);
        for (int dim=0; dim < ndim; dim++) {
            res[dim] = static_cast<int>(dims[dim]);
        }
        delete [] dims;
        return res;
    }
       
    ~SimpleHDF5Reader() {
        hdf5_file.close();   
    }
    
protected:
    // Convenience function for appending class name to exception message.
    static void throwRuntimeError(const std::string& msg) {
        std::string func_str;
#ifdef _MSC_VER
        func_str = std::string(__FUNCTION__) + msg;
#else
        func_str = std::string(__func__);
#endif
        throw std::runtime_error(func_str + std::string(" : ") + msg);
    }
    
protected:
    H5::H5File hdf5_file;
};

} // end namespace SimpleHDF

