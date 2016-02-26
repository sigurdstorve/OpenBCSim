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

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random> // for selecting scatterers

#include <QMenuBar>
#include <QMenu>
#include <QMessageBox>
#include <QHBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QDebug>
#include <QStatusBar>
#include <QSettings>
#include <QInputDialog>
#include <QTimer>
#include <QGraphicsPixmapItem>
#include "ScopedCpuTimer.hpp"

#include "MainWindow.hpp"
#include "../utils/HDFConvenience.hpp"
#include "../core/LibBCSim.hpp"
#include "utils.hpp" // needed for generating grayscale colortable
#include "../utils/SimpleHDF.hpp"    // for reading scatterer splines for vis.
#include "utils.hpp"
#include "../core/ScanSequence.hpp"
#include "DisplayWidget.hpp"

#include "GLVisualizationWidget.hpp"
#include "scanseq/ScanseqWidget.hpp"
#include "BeamProfileWidget.hpp"
#include "ExcitationSignalWidget.hpp"
#include "SimulationParamsWidget.hpp"
#include "../utils/SignalProcessing.hpp"
#include "../utils/ScanGeometry.hpp"
#include "../core/BCSimConfig.hpp"
#include "ProbeWidget.hpp"
#include "SimTimeWidget.hpp"
#include "GrayscaleTransformWidget.hpp"
#include "RefreshWorker.hpp"

MainWindow::MainWindow() {
    onLoadIniSettings();

    // Simulation time manager
    m_sim_time_manager = new SimTimeManager(0.0f, 1.0f);
    m_sim_time_manager->set_time(0.0f);
    m_sim_time_manager->set_time_delta(10e-3);

    // Create simulation time widget and configure signal connections.
    m_time_widget = new SimTimeWidget;
    connect(m_sim_time_manager, SIGNAL(min_time_changed(double)), m_time_widget, SLOT(set_min_time(double)));
    connect(m_sim_time_manager, SIGNAL(max_time_changed(double)), m_time_widget, SLOT(set_max_time(double)));
    connect(m_sim_time_manager, SIGNAL(time_changed(double)), m_time_widget, SLOT(set_time(double)));
    // Needed to update OpenGL visualization when time changes.
    connect(m_sim_time_manager, &SimTimeManager::time_changed, [&](double dummy) {
        updateOpenGlVisualization();
    });

    // Create main widget and its layout
    auto v_layout = new QVBoxLayout;
    auto h_layout = new QHBoxLayout;
    QWidget* window = new QWidget;
    window->setLayout(v_layout);
    setCentralWidget(window);

    if (m_settings->value("enable_gl_widget", true).toBool()) {
        m_gl_vis_widget = new GLVisualizationWidget;
        h_layout->addWidget(m_gl_vis_widget);
    }

    // One column of all custom wiggets
    auto left_widget_col = new QVBoxLayout;
    auto right_widget_col = new QVBoxLayout;

    // Scansequence widget
    m_scanseq_widget = new ScanseqWidget;
    m_scanseq_widget->setMaximumWidth(200);
    left_widget_col->addWidget(m_scanseq_widget);
    
    // Probe widget
    m_probe_widget = new ProbeWidget;
    m_probe_widget->setMaximumWidth(200);
    left_widget_col->addWidget(m_probe_widget);
    
    // Beam profile widget
    m_beamprofile_widget = new GaussianBeamProfileWidget;
    m_beamprofile_widget->setMaximumWidth(200);
    connect(m_beamprofile_widget, SIGNAL(valueChanged(bcsim::IBeamProfile::s_ptr)), this, SLOT(onNewBeamProfile(bcsim::IBeamProfile::s_ptr)));
    left_widget_col->addWidget(m_beamprofile_widget);
    
    // Excitation signal widget
    m_excitation_signal_widget = new ExcitationSignalWidget;
    m_excitation_signal_widget->setMaximumWidth(200);
    right_widget_col->addWidget(m_excitation_signal_widget);
    connect(m_excitation_signal_widget, SIGNAL(valueChanged(bcsim::ExcitationSignal)), this, SLOT(onNewExcitation(bcsim::ExcitationSignal)));

    // General parameters widget
    m_simulation_params_widget = new SimulationParamsWidget;
    m_simulation_params_widget->setMaximumWidth(200);
    right_widget_col->addWidget(m_simulation_params_widget);

    // grayscale transform widget
    m_grayscale_widget = new GrayscaleTransformWidget;
    m_grayscale_widget->setMaximumWidth(200);
    right_widget_col->addWidget(m_grayscale_widget);

    h_layout->addLayout(left_widget_col);
    h_layout->addLayout(right_widget_col);

    v_layout->addLayout(h_layout);
    v_layout->addWidget(m_time_widget);

    createMenus();
    /*
    auto scatterers_file = m_settings->value("default_scatterers").toString();
    if (!QFileInfo(scatterers_file).exists()) {
        scatterers_file = QFileDialog::getOpenFileName(this, tr("Invalid default file. Select h5 scatterer dataset"), "", tr("h5 files (*.h5)"));
        if (scatterers_file == "") {
            qDebug() << "No scatterer dataset selected. Exiting application.";
            onExit();
        }
    }
    loadScatterers(scatterers_file.toUtf8().constData());
    */
    onCreateNewSimulator();
    m_display_widget = new DisplayWidget;
    h_layout->addWidget(m_display_widget);
    
    // Playback timer
    m_playback_timer = new QTimer;
    m_playback_millisec = 1;
    connect(m_playback_timer, SIGNAL(timeout()), this, SLOT(onTimer()));

    int num_lines;
    auto geometry = m_scanseq_widget->get_geometry(num_lines);
    newScansequence(geometry, num_lines);

    // refresh thread setup
    qRegisterMetaType<refresh_worker::WorkTask::ptr>();
    qRegisterMetaType<refresh_worker::WorkResult::ptr>();
    m_refresh_worker = new refresh_worker::RefreshWorker(10);
    connect(m_refresh_worker, &refresh_worker::RefreshWorker::processed_data_available, [&](refresh_worker::WorkResult::ptr work_result) {
        work_result->image.setColorTable(GrayColortable());

        // get Cartesian extents from current scan geometry.
        const auto geometry = m_scanseq_widget->get_geometry(num_lines);
        float x_min, x_max, y_min, y_max;
        geometry->get_xy_extent(x_min, x_max, y_min, y_max);

        m_display_widget->update_bmode(QPixmap::fromImage(work_result->image), x_min, x_max, y_min, y_max);
        
        // TEST CODE: Demonstrate that images with an alpha channel works.
        /*
        QImage dummy_img(64, 64, QImage::Format_ARGB32);
        for (int i = 0; i < 64; i++) {
            for (int j = 0; j < 64; j++) {
                dummy_img.setPixel(i, j, QColor(255, 0, 0, i >= j ? 0 : 255).rgba());
            }
        }
        m_display_widget->update_colorflow(QPixmap::fromImage(dummy_img), x_min, x_max, y_min, y_max);
        */

        if (m_save_image_act->isChecked()) {
            // TODO: Have an object that remebers path and can save the geometry file (parameters.txt)
            const auto img_path = m_settings->value("img_output_folder", "d:/temp").toString();
            m_num_simulated_frames++;
            const QString filename =  img_path + QString("/frame%1.bmp").arg(m_num_simulated_frames, 6, 10, QChar('0'));
            qDebug() << "Simulation time is " << m_sim_time_manager->get_time() << ". Writing image to" << filename;
            work_result->image.save(filename, 0, 100);
        }
        // store updated normalization constant if enabled.
        auto temp = m_grayscale_widget->get_values();
        if (temp.auto_normalize) {
            m_grayscale_widget->set_normalization_constant(work_result->updated_normalization_const);
        }
    });
}

void MainWindow::onLoadIniSettings() {
    // TODO: Use smart pointer to avoid memleak.
    const QString ini_file("settings.ini");
    QFileInfo ini_file_info(ini_file);
    if (ini_file_info.exists()) {
        qDebug() << "Found " << ini_file << ". Using settings from this file";
    } else {
        qDebug() << "Unable to find " << ini_file << ". Using default settings.";
    }
    m_settings = new QSettings("settings.ini", QSettings::IniFormat);
}


void MainWindow::createMenus() {
    auto menuBar = new QMenuBar;
    auto fileMenu =     menuBar->addMenu(tr("&File"));
    auto simulateMenu = menuBar->addMenu(tr("&Simulate"));
    auto about_menu = menuBar->addMenu(tr("&About"));
    
    // Create all actions in "File" menu
    auto loadScatterersAct = new QAction(tr("Load scatterers"), this);
    connect(loadScatterersAct, SIGNAL(triggered()), this, SLOT(onLoadScatterers()));
    fileMenu->addAction(loadScatterersAct);

    auto loadExcitationAct = new QAction(tr("Load excitation signal"), this);
    connect(loadExcitationAct, SIGNAL(triggered()), this, SLOT(onLoadExcitation()));
    fileMenu->addAction(loadExcitationAct);

    auto new_simulator_act = new QAction(tr("Create a new simulator"), this);
    connect(new_simulator_act, SIGNAL(triggered()), this, SLOT(onCreateNewSimulator()));
    fileMenu->addAction(new_simulator_act);
    
    auto refresh_settings_act = new QAction(tr("Refresh settings"), this);
    connect(refresh_settings_act, SIGNAL(triggered()), this, SLOT(onLoadIniSettings()));
    fileMenu->addAction(refresh_settings_act);

    auto load_beamprofile_lut_act = new QAction(tr("Load LUT beamprofile"), this);
    connect(load_beamprofile_lut_act, SIGNAL(triggered()), this, SLOT(onLoadBeamProfileLUT()));
    fileMenu->addAction(load_beamprofile_lut_act);

    auto load_simdata_act = new QAction(tr("Load simulated data [experimental]"), this);
    connect(load_simdata_act, SIGNAL(triggered()), this, SLOT(onLoadSimulatedData()));
    fileMenu->addAction(load_simdata_act);

    auto exitAct = new QAction(tr("Exit"), this);
    connect(exitAct, SIGNAL(triggered()), this, SLOT(onExit()));
    fileMenu->addAction(exitAct);
    
    // Create all actions in "Simulate" menu
    auto simulateAct = new QAction(tr("Simulate"), this);
    connect(simulateAct, SIGNAL(triggered()), this, SLOT(onSimulate()));
    simulateMenu->addAction(simulateAct);

    m_save_image_act = new QAction(tr("Save images"), this);
    m_save_image_act->setCheckable(true);
    m_save_image_act->setChecked(false);
    simulateMenu->addAction(m_save_image_act);

    m_save_iq_act = new QAction(tr("Save IQ data"), this);
    m_save_iq_act->setCheckable(true);
    m_save_iq_act->setChecked(false);
    simulateMenu->addAction(m_save_iq_act);

    m_save_iq_buffer_as_act = new QAction(tr("Save IQ buffer as"), this);
    connect(m_save_iq_buffer_as_act, SIGNAL(triggered()), this, SLOT(onSaveIqBufferAs()));
    simulateMenu->addAction(m_save_iq_buffer_as_act);

    m_reset_iq_buffer_act = new QAction(tr("Reset IQ buffer"), this);
    connect(m_reset_iq_buffer_act, SIGNAL(triggered()), this, SLOT(onResetIqBuffer()));
    simulateMenu->addAction(m_reset_iq_buffer_act);

    auto save_cartesian_limits_act = new QAction(tr("Save xy extent"), this);
    connect(save_cartesian_limits_act, &QAction::triggered, [&]() {
        const auto img_path = m_settings->value("img_output_folder", "d:/temp").toString();
        const auto out_file = img_path + "/parameters.ini";
        QFile f(out_file);
        if (f.open(QIODevice::WriteOnly)) {
            float x_min, x_max, y_min, y_max;
            m_scan_geometry->get_xy_extent(x_min, x_max, y_min, y_max);
            QTextStream stream(&f);
            stream << "width_meters = " << (x_max-x_min) << "\n";
            stream << "height_meters = " << (y_max-y_min) << "\n";
        } else {
            throw std::runtime_error("failed to open file for writing");
        }
    });
    simulateMenu->addAction(save_cartesian_limits_act);

    // TODO: Finish implementation
    auto setTimeAct = new QAction(tr("Set time"), this);
    connect(setTimeAct, SIGNAL(triggered()), this, SLOT(onSetSimTme()));
    simulateMenu->addAction(setTimeAct);

    auto set_noise_act = new QAction(tr("Set noise amplitude"), this);
    connect(set_noise_act, &QAction::triggered, this, &MainWindow::onSetSimulatorNoise);
    simulateMenu->addAction(set_noise_act);

    auto start_timer_act = new QAction(tr("Start timer"), this);
    connect(start_timer_act, &QAction::triggered, this, &MainWindow::onStartTimer);
    simulateMenu->addAction(start_timer_act);

    auto stop_timer_act = new QAction(tr("Stop timer"), this);
    connect(stop_timer_act, &QAction::triggered, this, &MainWindow::onStopTimer);
    simulateMenu->addAction(stop_timer_act);

    auto playback_speed_act = new QAction(tr("Set playback speed"), this);
    connect(playback_speed_act, &QAction::triggered, this, &MainWindow::onSetPlaybackSpeed);
    simulateMenu->addAction(playback_speed_act);

    auto set_parameter_act = new QAction(tr("Set simulator parameter"), this);
    connect(set_parameter_act, &QAction::triggered, this, &MainWindow::onSetSimulatorParameter);
    simulateMenu->addAction(set_parameter_act);
    
    // Actions in about menu
    auto about_scatterers_act = new QAction(tr("Scatterers details"), this);
    connect(about_scatterers_act, &QAction::triggered, this, &MainWindow::onAboutScatterers);
    about_menu->addAction(about_scatterers_act);

    auto get_xy_extent_act = new QAction(tr("Get Cartesian scan limits"), this);
    connect(get_xy_extent_act, &QAction::triggered, this, &MainWindow::onGetXyExtent);
    about_menu->addAction(get_xy_extent_act);

    setMenuBar(menuBar);

}

void MainWindow::onLoadScatterers() {
    auto h5_file = QFileDialog::getOpenFileName(this, tr("Load h5 scatterer dataset"), "", tr("h5 files (*.h5)"));
    if (h5_file == "") {
        qDebug() << "Invalid scatterer file. Skipping";
        return;
    }
    loadScatterers(h5_file);
}

void MainWindow::onLoadExcitation() {
    auto h5_file = QFileDialog::getOpenFileName(this, tr("Load h5 excitation signal"), "", tr("h5 files (*.h5)"));
    if (h5_file == "") {
        qDebug() << "Invalid excitation file. Skipping";
        return;
    }
    setExcitation(h5_file);
}

void MainWindow::onCreateNewSimulator() {
    QStringList items;
    items << "cpu" << "gpu";
    bool ok;
    QString sim_type = QInputDialog::getItem(this, tr("Select algorithm type"),
                                         tr("Type:"), items, 0, false, &ok);
    if (ok) {
        try {
            qDebug() << "Creating simulator of type: " << sim_type;
            createNewSimulator(sim_type);
        } catch (const std::runtime_error& e) {
            qDebug() << "onCreateNewSimulator(): caught exception: " << e.what();
            onExit();
        }
    }
}

void MainWindow::onSimulate() {
    doSimulation();
}

void MainWindow::onSetSimulatorNoise() {
    bool ok;
    auto noise_amplitude = QInputDialog::getDouble(this, tr("New simulator noise value"), tr("New amplitude:"), 0.0, 0.0, 10e6, 3, &ok);
    if (!ok) {
        return;
    }

    qDebug() << "Setting new noise amplitude: " << noise_amplitude;
    m_sim->set_parameter("noise_amplitude", std::to_string(noise_amplitude));
}

void MainWindow::createNewSimulator(const QString sim_type) {
    m_sim = bcsim::Create(sim_type.toUtf8().constData());
    QString window_title_extra;
    if (sim_type == "cpu") {
        const auto num_cores = m_settings->value("cpu_sim_num_cores", 1).toInt();
        m_sim->set_parameter("num_cpu_cores", std::to_string(num_cores));
        window_title_extra = QString::number(num_cores) + " CPU cores";
    } else if (sim_type == "gpu") {
        auto device_no = m_settings->value("cuda_device_no", 0).toInt();

        // hack to query devices
        const auto num_devices = std::stoi(m_sim->get_parameter("num_cuda_devices"));
        if (num_devices == 0) {
            throw std::runtime_error("No CUDA devices was found");
        }
        if (device_no >= num_devices) {
            qDebug() << "Invalid device number specified in config. Will default to 0";
            device_no = 0;
        }
        qDebug() << "Number of CUDA devices: " << num_devices;
        for (int device_no = 0; device_no < num_devices; device_no++) {
            m_sim->set_parameter("gpu_device", std::to_string(device_no));
            const auto device_name = m_sim->get_parameter("cur_device_name");
            qDebug() << "Device " << device_no << " : " << device_name.c_str();
        }

        // switch to selected device number (or default)
        m_sim->set_parameter("gpu_device", std::to_string(device_no));
        QString device_name = m_sim->get_parameter("cur_device_name").c_str();
        window_title_extra += "GPU: " + device_name;
    } else {
        throw std::runtime_error("Should not happen");
    }
    setWindowTitle("BCSimGUI @ " + window_title_extra);
    
    // ask user for a scatterer dataset.
    onLoadScatterers();
    m_sim->set_parameter("verbose", "0");
    m_sim->set_parameter("sound_speed", "1540.0");
    m_sim->set_parameter("radial_decimation", std::to_string(m_settings->value("radial_decimation", 15).toInt()));
    m_sim->set_parameter("phase_delay", "on");

    // force-emit from all widgets to ensure a fully configured simulator.
    m_excitation_signal_widget->force_emit();

    // configure scanseq
    int num_lines;
    auto scan_geometry = m_scanseq_widget->get_geometry(num_lines);
    newScansequence(scan_geometry, num_lines);
        
    // use Gaussian beam profile by default
    const auto sigma_lateral     = m_beamprofile_widget->get_lateral_sigma();
    const auto sigma_elevational = m_beamprofile_widget->get_elevational_sigma();
    m_sim->set_analytical_profile(bcsim::IBeamProfile::s_ptr(new bcsim::GaussianBeamProfile(sigma_lateral, sigma_elevational)));

    updateOpenGlVisualization();
    m_num_simulated_frames = 0;
    qDebug() << "Created simulator";

}

void MainWindow::loadScatterers(const QString h5_file) {
    if (h5_file == "") {
        qDebug() << "Invalid scatterer file. Skipping";
        return;
    }

    m_sim->clear_fixed_scatterers();
    m_sim->clear_spline_scatterers();

    // load fixed scatterers (if found)
    try {
        auto fixed_scatterers = bcsim::loadFixedScatterersFromHdf(h5_file.toUtf8().constData());
        m_sim->add_fixed_scatterers(fixed_scatterers);
    } catch (std::runtime_error& /*e*/) {
        qDebug() << "Could not read fixed scatterers from file";
    }

    // load spline scatterers (if found)
    try {
        auto spline_scatterers = bcsim::loadSplineScatterersFromHdf(h5_file.toUtf8().constData());
        m_sim->add_spline_scatterers(spline_scatterers);

        // update simulation time limits
        float min_time, max_time;
        spline_scatterers->get_time_limits(min_time, max_time);
        m_sim_time_manager->set_min_time(min_time);
        m_sim_time_manager->set_max_time(max_time);
        m_sim_time_manager->reset();
        qDebug() << "Spline scatterers time interval is [" << min_time << ", " << max_time << "]";

    } catch (std::runtime_error& e) {
        qDebug() << "Could not read spline scatterers from file";
    }

    // Handle visualization in OpenGL - TODO: Update (sample some scatterers from all collections?)
    // This does not yet support hdf5 files with both types of scatterers!
    try {
        initializeSplineVisualization(h5_file);
    } catch (...) {
        qDebug() << "Failed to initialize visualization of spline scatterers";
    }
    try {
        initializeFixedVisualization(h5_file);
    } catch (...) {
        qDebug() << "Failed to initialize visualization of fixed scatterers";
    }
    updateOpenGlVisualization();
}

void MainWindow::newScansequence(bcsim::ScanGeometry::ptr new_geometry, int new_num_lines) {
    const auto cur_time = m_sim_time_manager->get_time();

    // Get probe origin and orientation corresponding to current simulation time.
    auto temp_probe_origin = m_probe_widget->get_origin(cur_time);
    bcsim::vector3 probe_origin(temp_probe_origin.x(), temp_probe_origin.y(), temp_probe_origin.z());
    
    auto temp_rot_angles = m_probe_widget->get_rot_angles(cur_time);
    bcsim::vector3 rot_angles(temp_rot_angles.x(), temp_rot_angles.y(), temp_rot_angles.z());

    m_scan_geometry = new_geometry;
    //qDebug() << "Probe orientation: " << rot_angles.x << rot_angles.y << rot_angles.z;
    auto new_scanseq = bcsim::OrientScanSequence(bcsim::CreateScanSequence(new_geometry, new_num_lines, cur_time), rot_angles, probe_origin);

    m_sim->set_scan_sequence(new_scanseq);
    m_cur_scanseq = new_scanseq;
    
    if (m_settings->value("enable_gl_widget", true).toBool()) {
        m_gl_vis_widget->setScanSequence(new_scanseq);
    }
}

void MainWindow::setExcitation(const QString h5_file) {
    throw std::runtime_error("this function should not be used");
    try {
        auto new_excitation = bcsim::loadExcitationFromHdf(h5_file.toUtf8().constData());
        m_sim->set_excitation(new_excitation);
        qDebug() << "Configured excitation";
    } catch (const std::runtime_error& e) {
        qDebug() << "Caught exception: " << e.what();
    }
}

void MainWindow::doSimulation() {
    // recreate scanseq to ensure correct time and probe info in case of dynamic probe.
    int new_num_scanlines;
    auto new_scan_geometry = m_scanseq_widget->get_geometry(new_num_scanlines);
    newScansequence(new_scan_geometry, new_num_scanlines);
    
    bool scan_is_color = true;

    typedef std::vector<std::vector<std::complex<float>>> IQ_Frame;
    
    if (scan_is_color) {
        // Color Doppler scan
        const auto color_packet_size = m_settings->value("color_packet_size", 16).toInt();
        const auto color_prf         = m_settings->value("color_prf", 2500.0).toFloat();
        const auto color_prt         = 1.0f/color_prf;
        std::vector<IQ_Frame> iq_frames_complex;

        try {
            int total_millisec = 0;
            for (int packet_no = 0; packet_no < color_packet_size; packet_no++) {
                // make a copy of current scan sequence to change timestamp
                auto temp_scanseq = bcsim::ScanSequence::s_ptr(new bcsim::ScanSequence(m_cur_scanseq->line_length));
                const auto num_lines = m_cur_scanseq->get_num_lines();
                float packet_timestamp;
                for (int line_no = 0; line_no < num_lines; line_no++) {
                    auto scanline = m_cur_scanseq->get_scanline(line_no);
                    const auto temp_direction = scanline.get_direction();
                    const auto temp_lateral_dir = scanline.get_lateral_dir();
                    const auto temp_origin = scanline.get_origin();
                    packet_timestamp = scanline.get_timestamp()+packet_no*color_prt;
                    temp_scanseq->add_scanline(bcsim::Scanline(temp_origin, temp_direction, temp_lateral_dir, packet_timestamp));
                }
                m_sim->set_scan_sequence(temp_scanseq);

                IQ_Frame iq_frame;
                {
                ScopedCpuTimer timer([&](int millisec) { total_millisec += millisec; });
                m_sim->simulate_lines(iq_frame);
                }
                iq_frames_complex.push_back(iq_frame);
                qDebug() << "Simulated frame in packet: timestamp is " << packet_timestamp;
            }

            // Estimate R0: TODO: Also R1
            std::vector<std::vector<float>> r0_lines;
            std::vector<std::vector<float>> velocity_lines;

            const auto num_iq_lines   = iq_frames_complex[0].size();
            const auto num_iq_samples = iq_frames_complex[0][0].size();;
            for (size_t line_no = 0; line_no < num_iq_lines; line_no++) {
                std::vector<float> temp1;
                std::vector<float> temp2;
                temp1.reserve(num_iq_samples);
                temp2.reserve(num_iq_samples);
                for (size_t i = 0; i < num_iq_samples; i++) {
                    float r0 = 0.0f;
                    std::complex<float> r1 = 0.0f;
                    for (int packet_no = 0; packet_no < color_packet_size; packet_no++) {
                        const auto z = iq_frames_complex[packet_no][line_no][i];
                        r0 += (z*std::conj(z)).real();
                    }
                    for (int packet_no = 0; packet_no < color_packet_size-1; packet_no++) {
                        r1 += std::conj(iq_frames_complex[packet_no][line_no][i])*iq_frames_complex[packet_no+1][line_no][i];
                    }
                    temp1.push_back(r0);
                    temp2.push_back(std::arg(r1));
                }
                r0_lines.push_back(temp1);
                velocity_lines.push_back(temp2);
            }

            // Draw R0: TODO: Also R1
            // Create refresh work task from current geometry and the beam space data
            auto refresh_task = refresh_worker::WorkTask::ptr(new refresh_worker::WorkTask);
            refresh_task->set_geometry(m_scan_geometry);
            refresh_task->set_data(r0_lines);
            auto grayscale_settings = m_grayscale_widget->get_values();
            refresh_task->set_normalize_const(grayscale_settings.normalization_const);
            refresh_task->set_auto_normalize(grayscale_settings.auto_normalize);
            refresh_task->set_dots_per_meter( m_settings->value("qimage_dots_per_meter", 6000.0f).toFloat() );
            refresh_task->set_dyn_range(grayscale_settings.dyn_range);
            refresh_task->set_gain(grayscale_settings.gain); 
    
            m_refresh_worker->process_data(refresh_task);
            statusBar()->showMessage("Color Doppler simulation time per packet: " + QString::number(total_millisec/static_cast<float>(color_packet_size)) + " ms.");


        } catch (std::runtime_error& e) {
            qDebug() << "Caught exception simulating color Doppler: " << e.what();
        }

    } else {
        // B-Mode scan
        try {
            IQ_Frame rf_lines_complex;
            int total_millisec;
            {
            ScopedCpuTimer timer([&](int millisec) { total_millisec = millisec; });
            m_sim->simulate_lines(rf_lines_complex);
            }

            m_display_widget->update_status(QString("Radial samples: %1").arg(rf_lines_complex[0].size()));

            if (m_save_iq_act->isChecked()) {
                m_iq_buffer.push_back(rf_lines_complex);
            }

            // Create refresh work task from current geometry and the beam space data
            auto bmode_task = std::make_shared<refresh_worker::WorkTask_BMode>();
            bmode_task->set_geometry(m_scan_geometry);
            bmode_task->set_data(rf_lines_complex);
            auto grayscale_settings = m_grayscale_widget->get_values();
            bmode_task->set_normalize_const(grayscale_settings.normalization_const);
            bmode_task->set_auto_normalize(grayscale_settings.auto_normalize);
            bmode_task->set_dots_per_meter( m_settings->value("qimage_dots_per_meter", 6000.0f).toFloat() );
            bmode_task->set_dyn_range(grayscale_settings.dyn_range);
            bmode_task->set_gain(grayscale_settings.gain); 
    
            m_refresh_worker->process_data(bmode_task);
            statusBar()->showMessage("B-mode simulation time: " + QString::number(total_millisec) + " ms.");

            const auto total_scatterers = m_sim->get_total_num_scatterers();
            const auto ns_value = static_cast<float>(1e6*total_millisec/(new_num_scanlines*total_scatterers));
            const auto msg = QString("Simulation time: %1 ms   ~   %2 nanosec. per scatterer per line")
                                .arg(total_millisec, 3)
                                .arg(ns_value, 3);
            statusBar()->showMessage(msg);

        } catch (std::runtime_error& e) {
            qDebug() << "Caught exception while simulating B-mode: " << e.what();

        } catch (...) {
            qDebug() << "Caught unknown error";
        }
    }
}

// Currently ignoring weights when visualizing
void MainWindow::initializeFixedVisualization(const QString& h5_file) {
    SimpleHDF::SimpleHDF5Reader reader(h5_file.toUtf8().constData());
    auto data =  reader.readMultiArray<float, 2>("data");
    auto shape = data.shape();
    auto dimensionality = data.num_dimensions();
    Q_ASSERT(dimensionality == 2);
    auto num_scatterers = shape[0];
    auto num_comp = shape[2];
    Q_ASSERT(num_comp == 4);

    qDebug() << "Number of scatterers is " << num_scatterers;
    int num_vis_scatterers = m_settings->value("num_opengl_scatterers", 1000).toInt();
    qDebug() << "Number of visualization scatterers is " << num_vis_scatterers;

    // Select random indices into scatterers
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(0, static_cast<int>(num_scatterers)-1);

    std::vector<bcsim::vector3> scatterer_points(num_vis_scatterers);
    for (int scatterer_no = 0; scatterer_no < num_vis_scatterers; scatterer_no++) {
        int ind = distr(eng);
        const auto p = bcsim::vector3(data[ind][0], data[ind][1], data[ind][2]);
        scatterer_points[scatterer_no] = p;
    }
    if (m_settings->value("enable_gl_widget", true).toBool()) {
        m_gl_vis_widget->setFixedScatterers(scatterer_points);
    }
}


// Currently ignoring weights when visualizing
void MainWindow::initializeSplineVisualization(const QString& h5_file) {
    SimpleHDF::SimpleHDF5Reader reader(h5_file.toUtf8().constData());
    auto control_points =  reader.readMultiArray<float, 3>("control_points");
    auto knot_vector    =  reader.readStdVector<float>("knot_vector");
    int spline_degree   = reader.readScalar<int>("spline_degree");

    auto shape = control_points.shape();
    auto dimensionality = control_points.num_dimensions();
    Q_ASSERT(dimensionality == 3);
    int num_scatterers = static_cast<int>(shape[0]);
    int num_cs = static_cast<int>(shape[1]);
    int num_comp = static_cast<int>(shape[2]);
    Q_ASSERT(num_comp == 4);
    qDebug() << "Number of scatterers is " << num_scatterers;
    qDebug() << "Each scatterer has " << num_cs << " control points";
    int num_vis_scatterers = m_settings->value("num_opengl_scatterers", 1000).toInt();
    qDebug() << "Number of visualization scatterers is " << num_vis_scatterers;

    num_vis_scatterers = std::min(num_vis_scatterers, num_scatterers);

    // Select random indices into scatterers
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(0, num_scatterers-1);

    std::vector<SplineCurve<float, bcsim::vector3> > splines(num_vis_scatterers);
    for (int scatterer_no = 0; scatterer_no < num_vis_scatterers; scatterer_no++) {
        int ind = distr(eng);
        
        // Create a SplineCurve for current scatterer
        SplineCurve<float, bcsim::vector3> curve;
        curve.knots = knot_vector;
        curve.degree = spline_degree;
        curve.cs.resize(num_cs);
        for (int cs_no = 0; cs_no < num_cs; cs_no++) {
            curve.cs[cs_no] = bcsim::vector3(control_points[ind][cs_no][0],
                                             control_points[ind][cs_no][1],
                                             control_points[ind][cs_no][2]);
        }
        splines[scatterer_no] = curve;
    }

    if (m_settings->value("enable_gl_widget", true).toBool()) {
        // Pass new splines the visualization widget
        m_gl_vis_widget->setScattererSplines(splines);
    }
}

void MainWindow::onNewExcitation(bcsim::ExcitationSignal new_excitation) {
    m_sim->set_excitation(new_excitation);
    qDebug() << "Configured excitation signal";
}

void MainWindow::onNewBeamProfile(bcsim::IBeamProfile::s_ptr new_beamprofile) {
    if (std::dynamic_pointer_cast<bcsim::GaussianBeamProfile>(new_beamprofile)) {
        m_sim->set_analytical_profile(new_beamprofile);
    } else if (std::dynamic_pointer_cast<bcsim::LUTBeamProfile>(new_beamprofile)) {
        m_sim->set_lookup_profile(new_beamprofile);
    } else {
        throw std::runtime_error("onNewBeamProfile(): all casts failed");
    }

    qDebug() << "Configured beam profile";
}

void MainWindow::onStartTimer() {
    m_playback_timer->start(m_playback_millisec);
}

void MainWindow::onStopTimer() {
    m_playback_timer->stop();
}

void MainWindow::onSetPlaybackSpeed() {
    bool ok;
    auto dt = QInputDialog::getDouble(this, "Simulation dt", "Time [s]", 1e-3, 0.0, 100.0, 5, &ok);
    if (ok) {
        m_sim_time_manager->set_time_delta(dt);
    }
}

void MainWindow::onSetSimTme() {
    bool ok;
    auto sim_time = QInputDialog::getDouble(this, "Simulation time", "Time [s]",
                                      m_sim_time_manager->get_time(),
                                      m_sim_time_manager->get_min_time(),
                                      m_sim_time_manager->get_max_time(), 5, &ok);
    if (ok) {
        m_sim_time_manager->set_time(sim_time);
    }
}

void MainWindow::onTimer() {
    m_sim_time_manager->advance();
    ScopedCpuTimer timer([](int millisec) {
        std::cout << "onTimer() used: " << millisec << " ms.\n";
    });
    onSimulate();
}

void MainWindow::onAboutScatterers() {
    if (!m_sim) {
        qDebug() << "No simulator is active";
        return;
    }
    const auto n = m_sim->get_total_num_scatterers();
    QMessageBox::information(this, "Current scatterers", QString("Phantom consists of %1 scatterers").arg(n));
}

void MainWindow::onGetXyExtent() {
    float x_min, x_max, y_min, y_max;
    m_scan_geometry->get_xy_extent(x_min, x_max, y_min, y_max);
    auto info = QString("x=%1...%2, y=%3...%4").arg(QString::number(x_min), QString::number(x_max), QString::number(y_min), QString::number(y_max));
    info += QString("\nWidth is %1. Height is %2").arg(QString::number(x_max-x_min), QString::number(y_max-y_min));
    QMessageBox::information(this, "Cartesian scan limits", info);
}

void MainWindow::updateOpenGlVisualization() {
    if (!m_gl_vis_widget || !m_settings->value("enable_gl_widget", true).toBool()) {
        return;
    }
    ScopedCpuTimer timer([](int milliseconds) {
        qDebug() << "updateOpenGlVisualization used " << milliseconds << " milliseconds";
    });
    // Update scatterer visualization
    auto new_timestamp = m_sim_time_manager->get_time();
    m_gl_vis_widget->updateTimestamp(new_timestamp);
}

void MainWindow::onLoadBeamProfileLUT() {
    if (!m_sim) {
        qDebug() << "No active simulator. Ignoring";
        return;
    }
    auto h5_file = QFileDialog::getOpenFileName(this, "Load HDF5 beam profile lookup-table", ".", "HDF5 files (*.h5)");
    if (h5_file == "") {
        qDebug() << "No lookup-table file selected. Ignoring.";
        return;
    }
    m_sim->set_lookup_profile(bcsim::loadBeamProfileFromHdf(h5_file.toUtf8().constData()));
}

void MainWindow::onLoadSimulatedData() {
    if (!m_sim) {
        qDebug() << "No active simulator. Ignoring";
        return;
    }
    auto h5_file = QFileDialog::getOpenFileName(this, "Load simulated data from HDF5 file", ".", "HDF5 files (*.h5)");
    if (h5_file == "") {
        qDebug() << "No file selected. Ignoring.";
        return;
    }

    qDebug() << "Loading replay data from " << h5_file;

    SimpleHDF::SimpleHDF5Reader hdf_reader(h5_file.toUtf8().constData());
    
    auto sim_data_dims = hdf_reader.getDimensions("sim_data_real");
    const auto sim_data_rank = sim_data_dims.size();

    if (hdf_reader.getDimensions("sim_data_imag").size() != sim_data_rank) {
        throw std::runtime_error("real/imag rank mismatch");
    }

    std::vector<std::vector<std::vector<std::complex<float>>>> iq_frames;
    if (sim_data_rank == 3) {
        // load IQ data from disk
        auto temp_real = hdf_reader.readMultiArray<float, 3>("sim_data_real");
        auto temp_imag = hdf_reader.readMultiArray<float, 3>("sim_data_imag");
        // data conversion
        const auto num_frames  = temp_real.shape()[0];
        const auto num_samples = temp_real.shape()[1];
        const auto num_lines   = temp_real.shape()[2];
        
        // sanity check
        if (temp_imag.shape()[0] != num_frames) throw std::runtime_error("real/imag mismatch in dimension 0");
        if (temp_imag.shape()[1] != num_samples) throw std::runtime_error("real/imag mismatch in dimension 1");
        if (temp_imag.shape()[2] != num_lines) throw std::runtime_error("real/imag mismatch in dimension 2");

        for (size_t frame_no = 0; frame_no < num_frames; frame_no++) {
            std::vector<std::vector<std::complex<float>>> iq_lines(num_lines);
            for (size_t i = 0; i < num_lines; i++) {
                iq_lines[i].resize(num_samples);
                for (size_t n = 0; n < num_samples; n++) {
                    iq_lines[i][n] = std::complex<float>(temp_real[frame_no][n][i], temp_imag[frame_no][n][i]);
                }
            }
            iq_frames.push_back(iq_lines);
            qDebug() << "Loaded " << num_lines << " lines, each with " << num_samples << " samples.";
        }
    } else if (sim_data_rank == 2) {
        auto temp_real = hdf_reader.readMultiArray<float, 2>("sim_data_real");
        auto temp_imag = hdf_reader.readMultiArray<float, 2>("sim_data_imag");
        // data conversion
        const auto num_samples = temp_real.shape()[0];
        const auto num_lines   = temp_real.shape()[1];
        
        // sanity check
        if (temp_imag.shape()[0] != num_samples) throw std::runtime_error("real/imag mismatch in dimension 0");
        if (temp_imag.shape()[1] != num_lines) throw std::runtime_error("real/imag mismatch in dimension 1");
        
        std::vector<std::vector<std::complex<float>>> iq_lines(num_lines);
        for (size_t i = 0; i < num_lines; i++) {
            iq_lines[i].resize(num_samples);
            for (size_t n = 0; n < num_samples; n++) {
                iq_lines[i][n] = std::complex<float>(temp_real[n][i], temp_imag[n][i]);
            }
        }
        iq_frames.push_back(iq_lines);
        qDebug() << "Loaded " << num_lines << " lines, each with " << num_samples << " samples.";
    } else {
        throw std::runtime_error("sim_data must have rank 2 or 3");
    }
    
    for (size_t frame_no = 0; frame_no < iq_frames.size(); frame_no++) {
        // Create refresh work task from current geometry and the beam space data
        auto bmode_task = std::make_shared<refresh_worker::WorkTask_BMode>();
        bmode_task->set_geometry(m_scan_geometry);
        bmode_task->set_data(iq_frames[frame_no]);
        auto grayscale_settings = m_grayscale_widget->get_values();
        bmode_task->set_normalize_const(grayscale_settings.normalization_const);
        bmode_task->set_auto_normalize(grayscale_settings.auto_normalize);
        bmode_task->set_dots_per_meter( m_settings->value("qimage_dots_per_meter", 6000.0f).toFloat() );
        bmode_task->set_dyn_range(grayscale_settings.dyn_range);
        bmode_task->set_gain(grayscale_settings.gain); 
        m_refresh_worker->process_data(bmode_task);
    }
}

void MainWindow::onSetSimulatorParameter() {
    if (!m_sim) {
        qDebug() << "No valid simulator.";
        return;
    }
    bool ok;
    auto key = QInputDialog::getText(this, tr("Parameter key"), tr("key:"), QLineEdit::Normal, "", &ok);
    if (!ok || key.isEmpty()) {
        qDebug() << "Invalid key.";
        return;
    }
    auto value = QInputDialog::getText(this, tr("Parameter value"), tr("value:"), QLineEdit::Normal, "", &ok);
    if (!ok || key.isEmpty()) {
        qDebug() << "Invalid value.";
        return;
    }
    try {
        m_sim->set_parameter(key.toUtf8().constData(), value.toUtf8().constData());
    } catch (std::runtime_error& e) {
        qDebug() << "Caught exception: " << e.what();
    } catch (...) {
        qDebug() << "Caught unknown exception.";
    }
}

void MainWindow::onSaveIqBufferAs() {
    const auto num_frames = m_iq_buffer.size();
    qDebug() << "Buffer contains IQ data for " << num_frames << " frames.";

    if (num_frames == 0) {
        qDebug() << "No frames in buffer. Skipping";
        return;
    }

    auto h5_file = QFileDialog::getSaveFileName(this, "Save IQ buffer as HDF5", ".", "HDF5 files (*.h5)");
    if (h5_file == "") {
        qDebug() << "Ignoring IQ buffer save";
        return;
    }

    // all frames must have same dimensions
    const auto& frame = m_iq_buffer[0];
    const auto num_lines   = frame.size();
    const auto num_samples = frame[0].size();
    qDebug() << "Each frame has " << num_lines << " lines of " << num_samples << " samples.";
    
    boost::array<size_t, 3> dims;
    dims[0] = num_frames;
    dims[1] = num_lines;
    dims[2] = num_samples;

    boost::multi_array<float, 3> iq_real;
    boost::multi_array<float, 3> iq_imag;
    iq_real.resize(dims);
    iq_imag.resize(dims);

    qDebug() << "Converting data";
    for (size_t frame_no = 0; frame_no < num_frames; frame_no++) {
        for (size_t line_no = 0; line_no < num_lines; line_no++) {
            for (size_t sample_no = 0; sample_no < num_samples; sample_no++) {
                iq_real[frame_no][line_no][sample_no] = m_iq_buffer[frame_no][line_no][sample_no].real();
                iq_imag[frame_no][line_no][sample_no] = m_iq_buffer[frame_no][line_no][sample_no].imag();
            }
        }
    }

    auto file = H5::H5File(h5_file.toUtf8().constData(), H5F_ACC_TRUNC);
    hsize_t dspace_dims[] = {num_frames, num_lines, num_samples};
    H5::DataSpace dspace(3, dspace_dims);
    
    qDebug() << "Writing real part";
    auto dset_real = file.createDataSet("iq_real", H5::PredType::NATIVE_FLOAT, dspace);
    dset_real.write(iq_real.data(), H5::PredType::NATIVE_FLOAT);
    
    qDebug() << "Writing imaginary part";
    auto dset_imag = file.createDataSet("iq_imag", H5::PredType::NATIVE_FLOAT, dspace);
    dset_imag.write(iq_imag.data(), H5::PredType::NATIVE_FLOAT);
    
    onResetIqBuffer();
}

void MainWindow::onResetIqBuffer() {
    m_iq_buffer.clear();
}

