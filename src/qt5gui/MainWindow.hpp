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
#include <QApplication>
#include <QMainWindow>
#include <QSlider>
#include <QDebug>
#include <QSettings>
#include "../core/LibBCSim.hpp"
#include "SimTimeManager.hpp"
#include "../utils/ScanGeometry.hpp"

// Forward decl.
class DisplayWidget;
class GLVisualizationWidget;
class QwtPlot;
class QwtPlotCurve;
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

    // Configure current simulator with new scatterers. Will clear the existsing.
    void loadScatterers(const QString h5_file);

    // Create a new simulator
    void createNewSimulator(const QString sim_type);

    // Define the excitation signal using data from hdf5 file
    void setExcitation(const QString h5_file);

    // Do simulation using current config.
    void doSimulation();

private slots:
    void newScansequence(bcsim::ScanGeometry::ptr new_geometry, int new_num_lines, bool equal_timestamps);
    
    // Configure current simulator with new scatterers.
    // Ask user for a h5 file with scatterers.
    void onLoadScatterers();

    // Ask user for a h5 file with excitation signal
    void onLoadExcitation();

    // Simulate using current config
    void onSimulate();

    // Create a new simulator instance. Will ask user for CPU or GPU impl.
    void onCreateNewSimulator();

    void onExit() {
        QApplication::quit();
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

    void updateOpenGlVisualization();

    void onSetSimTme();

    // Load configuration from .ini file.
    void onLoadIniSettings();

    void onLoadBeamProfileLUT();
    
    void onSetSimulatorParameter();

    void onLoadSimulatedData();

    void onSaveIqBufferAs();

    void onResetIqBuffer();

private:
    void initializeSplineVisualization(const QString& h5_file);

    void initializeFixedVisualization(const QString& h5_file);

private:
    // The simulator object.
    bcsim::IAlgorithm::s_ptr        m_sim;
    // Running count of number of frames simulated since the simulator was created.
    size_t                          m_num_simulated_frames;
    
    // True/false checkable menu actions
    QAction*                        m_save_image_act;

    // The OpenGL based visualization widget
    GLVisualizationWidget*          m_gl_vis_widget;

    // Use Graphics View Framework for visualizing ultrasound data
    DisplayWidget*                  m_display_widget;

    std::shared_ptr<QSettings>      m_settings;
    GaussianBeamProfileWidget*      m_beamprofile_widget;
    ExcitationSignalWidget*         m_excitation_signal_widget;
    SimulationParamsWidget*         m_simulation_params_widget;
    ScanseqWidget*                  m_scanseq_widget;
    ProbeWidget*                    m_probe_widget;

    // The current scan geometry.
    bcsim::ScanGeometry::ptr         m_scan_geometry;

    // Timer playback
    QTimer*                         m_playback_timer;
    int                             m_playback_millisec;
    SimTimeManager*                 m_sim_time_manager;
    SimTimeWidget*                  m_time_widget;

    GrayscaleTransformWidget*       m_grayscale_widget;
    
    refresh_worker::RefreshWorker*  m_refresh_worker;

    // Related to IQ-buffering
    std::vector<std::vector<std::vector<std::complex<float>>>> m_iq_buffer;
    QAction*                        m_save_iq_act;
    QAction*                        m_save_iq_buffer_as_act;
    QAction*                        m_reset_iq_buffer_act;

    // Related to scan types
    QAction*                        m_enable_bmode_act;
    QAction*                        m_enable_color_act;

    // Invariant: Always equal to current scan sequence.
    // Needed for color Doppler since a packet of frames must be simulated
    // with different timestamps.
    bcsim::ScanSequence::s_ptr      m_cur_scanseq;
};


