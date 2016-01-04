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

#include "ExcitationSignalWidget.hpp"
#include <QDoubleSpinBox>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QPushButton>
#include <QVector>
#ifdef BCSIM_ENABLE_QWT
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#endif
#include "GaussPulse.hpp"

ExcitationSignalWidget::ExcitationSignalWidget(QWidget* parent)
    : QWidget(parent) 
{
    auto main_layout = new QVBoxLayout;
    auto group_box   = new QGroupBox("Gaussian pulse excitiation");
    auto form_layout = new QFormLayout;
#ifdef BCSIM_ENABLE_QWT
    m_plot_curve = new QwtPlotCurve("Excitation");
    m_plot = new QwtPlot;
    m_plot->setFixedSize(150, 150);
    m_plot_curve->attach(m_plot);
    form_layout->addRow(m_plot);
#endif
    m_sampling_freq_sb = new QDoubleSpinBox;
    m_sampling_freq_sb->setRange(1, 1000);
    m_sampling_freq_sb->setSingleStep(10);
    m_sampling_freq_sb->setValue(100);
    m_sampling_freq_sb->setSuffix("MHz");
    connect(m_sampling_freq_sb, SIGNAL(valueChanged(double)), this, SLOT(onSomethingChanged()));

    m_center_freq_sb = new QDoubleSpinBox;
    m_center_freq_sb->setRange(0.0, 100);
    m_center_freq_sb->setSingleStep(0.1);
    m_center_freq_sb->setValue(2.5);
    m_center_freq_sb->setSuffix("MHz");
    connect(m_center_freq_sb, SIGNAL(valueChanged(double)), this, SLOT(onSomethingChanged()));

    m_bandwidth_sb = new QDoubleSpinBox;
    m_bandwidth_sb->setRange(0.1, 150.0);
    m_bandwidth_sb->setSingleStep(1.0);
    m_bandwidth_sb->setValue(10.0);
    m_bandwidth_sb->setSuffix("%");
    connect(m_bandwidth_sb, SIGNAL(valueChanged(double)), this, SLOT(onSomethingChanged()));

    form_layout->addRow("Sampling freq.", m_sampling_freq_sb);
    form_layout->addRow("Center freq.",   m_center_freq_sb);
    form_layout->addRow("Bandwidth",      m_bandwidth_sb);

    main_layout->addWidget(group_box);
    group_box->setLayout(form_layout);
    
    setLayout(main_layout);
    
    onSomethingChanged();
}

void ExcitationSignalWidget::force_emit() {
    onSomethingChanged();
}

bcsim::ExcitationSignal ExcitationSignalWidget::construct(std::vector<float>& /*out*/ excitation_times) const {
    // construct a new ExcitationSignal
    bcsim::ExcitationSignal new_excitation;

    // read values from GUI controls
    new_excitation.sampling_frequency = static_cast<float>(m_sampling_freq_sb->value()*1e6);
    auto center_frequency             = static_cast<float>(m_center_freq_sb->value()*1e6);
    auto fractional_bandwidth         = static_cast<float>(m_bandwidth_sb->value()*0.01);

    //std::vector<float> times;
    bcsim::MakeGaussianExcitation(center_frequency,
                                 fractional_bandwidth,
                                 new_excitation.sampling_frequency,
                                 excitation_times,
                                 new_excitation.samples,
                                 new_excitation.center_index);
    new_excitation.demod_freq = center_frequency;
    return new_excitation;
}

void ExcitationSignalWidget::onSomethingChanged() {
    std::vector<float> temp_times;
    auto new_excitation = construct(temp_times);
#ifdef BCSIM_ENABLE_QWT
    auto num_samples = temp_times.size();
    // convert to double for plotting
    std::vector<double> plot_times(num_samples);
    std::vector<double> plot_samples(num_samples);
    for (size_t i = 0; i < num_samples; i++) {
        plot_times[i]   = static_cast<double>(temp_times[i]);
        plot_samples[i] = static_cast<double>(new_excitation.samples[i]);
    }

    m_plot_curve->setSamples(plot_times.data(), plot_samples.data(), static_cast<int>(num_samples));
    const auto min_value = *std::min_element(plot_times.begin(), plot_times.end());
    const auto max_value = *std::max_element(plot_times.begin(), plot_times.end());
    m_plot->setAxisScale(QwtPlot::xBottom, min_value, max_value);
    m_plot->replot();
#endif
    emit valueChanged(new_excitation);
}
