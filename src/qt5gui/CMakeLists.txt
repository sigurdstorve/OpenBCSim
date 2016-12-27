set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

option(BCSIM_ENABLE_QWT "Enable Qwt for plots" OFF)

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

set(BCSimGUI_SOURCES
    main.cpp
    MainWindow.cpp
    utils.cpp
    GLScattererWidget.cpp
    GLVisualizationWidget.cpp
    ScattererModel.cpp
    scanseq/SectorScanseqWidget.cpp
    scanseq/LinearScanseqWidget.cpp
    scanseq/ScanseqWidget.cpp
    BeamProfileWidget.cpp
    ExcitationSignalWidget.cpp
    SimulationParamsWidget.cpp
    ProbeWidget.cpp
    GrayscaleTransformWidget.cpp
    DisplayWidget.cpp
    QFileAdapter.cpp
    QSettingsConfigAdapter.cpp
    ImageExport.cpp
    LogWidget.cpp
    )

set(BCSimGUI_HEADERS
    MainWindow.hpp
    utils.hpp
    GLScattererWidget.hpp
    GLVisualizationWidget.hpp
    ScattererModel.hpp
    SplineCurve.hpp
    ScanSeqModel.hpp
    scanseq/SectorScanseqWidget.hpp
    scanseq/LinearScanseqWidget.hpp
    scanseq/ScanseqWidget.hpp
    BeamProfileWidget.hpp
    ExcitationSignalWidget.hpp
    SimulationParamsWidget.hpp
    ProbeWidget.hpp
    SimTimeManager.hpp
    SimTimeWidget.hpp
    GrayscaleTransformWidget.hpp
    RefreshWorker.hpp
    ScopedCpuTimer.hpp
    DisplayWidget.hpp
    QFileAdapter.hpp
    IConfig.hpp
    QSettingsConfigAdapter.hpp
    ImageExport.hpp
    LogWidget.hpp
    )

set(BCSimGUI_STUFF
    ${BCSimGUI_SOURCES}     
    ${BCSimGUI_HEADERS}
    resources/resources.qrc
    )
    
if (WIN32)
    # Avoid the console window if Windows build
    add_executable(BCSimGUI WIN32 ${BCSimGUI_STUFF})
else()
    add_executable(BCSimGUI ${BCSimGUI_STUFF})
endif()

# Use the Widgets module from Qt5
target_link_libraries(BCSimGUI
                      Qt5::Core
                      Qt5::Widgets
                      Qt5::OpenGL
                      Boost::boost
                      LibBCSim
                      LibBCSimUtils # BCSimCUDA also?
                      trianglemesh3d
                      )

if (TARGET hdf5-shared AND TARGET hdf5_cpp-shared)
    target_link_libraries(BCSimGUI hdf5-shared hdf5_cpp-shared)
else()
    target_link_libraries(BCSimGUI ${HDF5_LIBRARIES})
endif()

if (WIN32)
    target_link_libraries(BCSimGUI Qt5::WinMain)
endif()

if (BCSIM_ENABLE_QWT)
    target_link_libraries(BCSimGUI ${QWT_LIBRARIES})
endif()
                      

install(TARGETS BCSimGUI DESTINATION bin)
