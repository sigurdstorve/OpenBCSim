set(UTILS_LIBRARY_SOURCE_FILES "")
list(APPEND UTILS_LIBRARY_SOURCE_FILES
     BCSimConvenience.hpp
     BCSimConvenience.cpp
     GaussPulse.hpp
     GaussPulse.cpp
     ScanGeometry.hpp
     rotation3d.hpp
     SignalProcessing.hpp
     SignalProcessing.cpp
     HDFConvenience.cpp
     HDFConvenience.hpp
     SimpleHDF.hpp
     DefaultPhantoms.hpp
     DefaultPhantoms.cpp
     EllipsoidGeometry.cpp
     EllipsoidGeometry.hpp
     cartesianator/Cartesianator.cpp
     cartesianator/Cartesianator.hpp
     CSVReader.hpp
     CSVReader.cpp
     HardwareAutodetection.hpp
     HardwareAutodetection.cpp 
     )

add_library(LibBCSimUtils ${UTILS_LIBRARY_SOURCE_FILES})
target_link_libraries(LibBCSimUtils
                      hdf5-shared
                      hdf5_cpp-shared
                      Boost::boost
                      )

if (BCSIM_ENABLE_CUDA)
    # Needed because we're going to interact with the CUDA runtime API
    # to query the number and type of available GPUs.
    target_include_directories(LibBCSimUtils PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(LibBCSimUtils ${CUDA_LIBRARIES})
endif()
                      
if (BCSIM_BUILD_UNITTEST)
    add_subdirectory(unittest)
endif()

install(TARGETS LibBCSimUtils DESTINATION lib)
install(FILES HDFConvenience.hpp    DESTINATION include)
install(FILES GaussPulse.hpp        DESTINATION include)
install(FILES BCSimConvenience.hpp  DESTINATION include)
install(FILES SignalProcessing.hpp  DESTINATION include)
install(FILES GaussPulse.hpp        DESTINATION include)
