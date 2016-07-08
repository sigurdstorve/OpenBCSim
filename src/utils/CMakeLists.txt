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
     DemoPhantoms.cpp
     DemoPhantoms.hpp
     cartesianator/Cartesianator.cpp
     cartesianator/Cartesianator.hpp
     )

add_library(LibBCSimUtils STATIC ${UTILS_LIBRARY_SOURCE_FILES})
target_link_libraries(LibBCSimUtils
                      hdf5-static
                      hdf5_cpp-static
                      Boost::boost
                      )

if (BCSIM_BUILD_UNITTEST)
    add_subdirectory(unittest)
endif()

install(TARGETS LibBCSimUtils DESTINATION lib)
install(FILES HDFConvenience.hpp    DESTINATION include)
install(FILES GaussPulse.hpp        DESTINATION include)
install(FILES BCSimConvenience.hpp  DESTINATION include)
install(FILES SignalProcessing.hpp  DESTINATION include)
install(FILES GaussPulse.hpp        DESTINATION include)
