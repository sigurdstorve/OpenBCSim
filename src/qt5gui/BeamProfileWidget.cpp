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

#include "BeamProfileWidget.hpp"
#include <QDoubleSpinBox>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QFormLayout>
#include <QPushButton>

GaussianBeamProfileWidget::GaussianBeamProfileWidget(QWidget* parent)
    : QWidget(parent) 
{
    auto main_layout = new QVBoxLayout;
    auto group_box   = new QGroupBox("Gaussian beam profile");
    auto form_layout = new QFormLayout;
    
    const double MIN_VAL = 0.001;
    const double MAX_VAL = 20;
    m_sigma_lat_sb = new QDoubleSpinBox;
    m_sigma_lat_sb->setRange(MIN_VAL, MAX_VAL);
    m_sigma_lat_sb->setSingleStep(0.1);
    m_sigma_lat_sb->setValue(0.5);
    m_sigma_lat_sb->setSuffix("mm");
    m_sigma_lat_sb->setDecimals(3);
    connect(m_sigma_lat_sb, SIGNAL(valueChanged(double)), this, SLOT(onSomethingChanged()));

    m_sigma_ele_sb = new QDoubleSpinBox;
    m_sigma_ele_sb->setRange(MIN_VAL, MAX_VAL);
    m_sigma_ele_sb->setSingleStep(0.1);
    m_sigma_ele_sb->setValue(1.0);
    m_sigma_ele_sb->setSuffix("mm");
    m_sigma_ele_sb->setDecimals(3);
    connect(m_sigma_ele_sb, SIGNAL(valueChanged(double)), this, SLOT(onSomethingChanged()));

    form_layout->addRow("Lateral sigma",     m_sigma_lat_sb);
    form_layout->addRow("Elevational sigma", m_sigma_ele_sb);

    main_layout->addWidget(group_box);
    group_box->setLayout(form_layout);
    setLayout(main_layout);
}

float GaussianBeamProfileWidget::get_lateral_sigma() const {
    return m_sigma_lat_sb->value()*1e-3;
}

float GaussianBeamProfileWidget::get_elevational_sigma() const {
    return m_sigma_ele_sb->value()*1e-3;
}

bcsim::IBeamProfile::s_ptr GaussianBeamProfileWidget::getValue() {
    const auto lateral_sigma     = get_lateral_sigma();
    const auto elevational_sigma = get_elevational_sigma();
    return bcsim::IBeamProfile::s_ptr(new bcsim::GaussianBeamProfile(lateral_sigma, elevational_sigma));
}

void GaussianBeamProfileWidget::onSomethingChanged() {
    emit valueChanged(getValue());
}