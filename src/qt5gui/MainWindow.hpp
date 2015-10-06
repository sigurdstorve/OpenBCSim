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
#include <memory>
#include <QMainWindow>
#include <QLabel>
#include <QSlider>
#include <QDebug>
#include "../LibBCSim.hpp"
#include "SimTimeManager.hpp"

// Forward decl.
class GLVisualizationWidget;
class QwtPlot;
class QwtPlotCurve;
class QSettings;
class ScanseqWidget;
class GaussianBeamProfileWidget;
class ExcitationSignalWidget;
class SimulationParamsWidget;
class ProbeWidget;
class SimTimeWidget;
class GrayscaleTransformWidget;
class QTimer;
namespace refresh_worker {
    class RefreshWorker;
}

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow();

    // Create the menus
    void createMenus();

    // Update the scatterers with file contents.
    // This will recreate the simulator by calling initializeSimulator()
    void loadScatterers(const QString h5_file);

    // Create and configure the simulator object
    // type: "fixed" or "spline"
    void initializeSimulator(const std::string& type);

    // Define the excitation signal using data from hdf5 file
    void setExcitation(const QString h5_file);

    // Do simulation using current config.
    void doSimulation();

private slots:
    void newScansequence(bcsim::ScanGeometry::ptr new_geometry, int new_num_lines);
    
    // Ask user for a h5 file with scatterers.
    void onLoadScatterers();

    // Ask user for a h5 file with excitation signal
    void onLoadExcitation();

    // Simulate using current config
    void onSimulate();

    // Experimental: create a new instance of the GPU simulator
    void onCreateGpuSimulator();

    // Experimental: configure a GPU simulator with new fixed scatterers
    void onGpuLoadScatterers();

    void onExit() {
        exit(0);
    }

    // Update simulator object with new ExcitationSignal from widget
    void onNewExcitation(bcsim::ExcitationSignal new_excitation);

    // Update simulator object with new BeamProfile
    void onNewBeamProfile(bcsim::IBeamProfile::s_ptr new_beamprofile);

    void onSetSimulatorNoise();

    void onStartTimer();

    void onStopTimer();

    void onSetPlaybackSpeed();

    void onTimer();

    void onAboutScatterers();

    void onGetXyExtent();

    void onToggleSaveImage(bool status) {
        m_save_images = status;
    }

    void updateOpenGlVisualization();

    void onSetSimTme();

    // Load configuration from .ini file.
    void onLoadIniSettings();

    void onLoadBeamProfileLUT();
    
    void onSetSimulatorParameter();

    void onLoadSimulatedData();

private:
    void initializeSplineVisualization(const QString& h5_file);

    void initializeFixedVisualization(const QString& h5_file);

private:
    // The simulator object.
    bcsim::IAlgorithm::s_ptr        m_sim;
    // Running count of number of frames simulated since the simulator was created.
    size_t                          m_num_simulated_frames;
    bool                            m_save_images;

    // The OpenGL based visualization widget
    GLVisualizationWidget*          m_gl_vis_widget;

    // The label that is used to show the B-mode image
    QLabel*                         m_label;

    QSettings*                      m_settings;
    GaussianBeamProfileWidget*      m_beamprofile_widget;
    ExcitationSignalWidget*         m_excitation_signal_widget;
    SimulationParamsWidget*         m_simulation_params_widget;
    ScanseqWidget*                  m_scanseq_widget;
    ProbeWidget*                    m_probe_widget;

    // The current scan geometry.
    bcsim::ScanGeometry::ptr         m_scan_geometry;


    // Invariant: should at all times mirror the configuration of the simulator object.
    bcsim::ExcitationSignal          m_current_excitation;
    bcsim::Scatterers::s_ptr     m_current_scatterers;  

    // Timer playback
    QTimer*                         m_playback_timer;
    int                             m_playback_millisec;
    SimTimeManager*                 m_sim_time_manager;
    SimTimeWidget*                  m_time_widget;

    GrayscaleTransformWidget*       m_grayscale_widget;
    
    refresh_worker::RefreshWorker*  m_refresh_worker;
};


