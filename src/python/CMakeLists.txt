# Handle Python include & libs
find_package(PythonLibs REQUIRED)
if (NOT PYTHONLIBS_FOUND)
    message("PYTHONLIBS NOT FOUND")
endif ()
include_directories(${PYTHON_INCLUDE_DIRS})

# Get the NumPy include path and verify it
set(NumPy_INCLUDE_DIR "NOT-SET" CACHE PATH "NumPy include path")
if (NOT EXISTS "${NumPy_INCLUDE_DIR}/numpy/arrayobject.h")
    message(FATAL_ERROR "Invalid NumPy include path")
endif()
include_directories(${NumPy_INCLUDE_DIR})

# Handle Boost.Python
find_package(Boost REQUIRED COMPONENTS python system)

add_library(pyrfsim
            SHARED
            PythonInterface.cpp
            numpy_boost.hpp
            numpy_boost_python.hpp
            )
target_link_libraries(pyrfsim
                      ${PYTHON_LIBRARIES}
                      Boost::python
                      Boost::system
                      LibBCSim
                      hdf5-static
                      hdf5_cpp-static
                      )
set_target_properties(pyrfsim PROPERTIES PREFIX "")
if (WIN32)
set_target_properties(pyrfsim PROPERTIES SUFFIX ".pyd")
endif()

# put it in "bin" because it needs many of the same DLLs
# as the executables.
install(TARGETS pyrfsim DESTINATION bin)
