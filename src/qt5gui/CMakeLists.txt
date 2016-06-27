set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

option(BCSIM_ENABLE_QWT "Enable Qwt for plots" ON)

# In order for this to work, Qt5 must be found below the
# CMAKE_PREFIX_PATH, or the Qt5<Module>_DIR must be set
# in CMake cache to the location of Qt5WidgetsConfig.cmake
# Easiest: set CMAKE_PREFIX_PATH env.var to the install
# prefix of Qt5.
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5OpenGL)

if (BCSIM_ENABLE_QWT)
    find_package(Qwt REQUIRED)
    include_directories(${QWT_INCLUDE_DIRS})
    add_definitions(-DBCSIM_ENABLE_QWT)
endif()

add_subdirectory(trianglemesh3d)

add_executable(BCSimGUI
               main.cpp
               MainWindow.cpp
               MainWindow.hpp
               utils.hpp
               utils.cpp
               GLScattererWidget.cpp
               GLScattererWidget.hpp
               GLVisualizationWidget.cpp
               GLVisualizationWidget.hpp
               ScattererModel.hpp
               ScattererModel.cpp
               SplineCurve.hpp
               ScanSeqModel.hpp
               scanseq/SectorScanseqWidget.hpp
               scanseq/SectorScanseqWidget.cpp
               scanseq/LinearScanseqWidget.hpp
               scanseq/LinearScanseqWidget.cpp
               scanseq/ScanseqWidget.hpp
               scanseq/ScanseqWidget.cpp
               BeamProfileWidget.hpp
               BeamProfileWidget.cpp
               ExcitationSignalWidget.hpp
               ExcitationSignalWidget.cpp
               SimulationParamsWidget.hpp
               SimulationParamsWidget.cpp
               ProbeWidget.hpp
               ProbeWidget.cpp
               SimTimeManager.hpp
               SimTimeWidget.hpp
               GrayscaleTransformWidget.hpp
               GrayscaleTransformWidget.cpp
               RefreshWorker.hpp
               ScopedCpuTimer.hpp
               DisplayWidget.hpp
               DisplayWidget.cpp
               )

# Use the Widgets module from Qt5
target_link_libraries(BCSimGUI
                      Qt5::Core
                      Qt5::Widgets
                      Qt5::OpenGL
                      Boost::boost
                      LibBCSim
                      LibBCSimUtils # BCSimCUDA also?
                      hdf5-static
                      hdf5_cpp-static
                      )
if (BCSIM_ENABLE_QWT)
    target_link_libraries(BCSimGUI ${QWT_LIBRARIES})
endif()
                      

install(TARGETS BCSimGUI DESTINATION bin)
