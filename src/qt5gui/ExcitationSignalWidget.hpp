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
#include <QWidget>
#include <QSize>
#include "BCSimConfig.hpp"

// Forward decl.
class QDoubleSpinBox;
class QwtPlot;
class QwtPlotCurve;

class ExcitationSignalWidget : public QWidget {
Q_OBJECT
public:
    explicit ExcitationSignalWidget(QWidget* parent=0);

    virtual QSize sizeHint() const {
       return QSize(200, 500);
    }

    void force_emit();

signals:
    void valueChanged(bcsim::ExcitationSignal excitation);

private:
    // construct a new ExcitationSignal with parameters from GUI controls.
    bcsim::ExcitationSignal construct(std::vector<float>& /*out*/ excitation_times) const;

private slots:
    // called when the user changes some of the parameters.
    void onSomethingChanged();

private:
    QDoubleSpinBox*     m_sampling_freq_sb;
    QDoubleSpinBox*     m_center_freq_sb;
    QDoubleSpinBox*     m_bandwidth_sb;
    QwtPlot*            m_plot;
    QwtPlotCurve*       m_plot_curve;
};

