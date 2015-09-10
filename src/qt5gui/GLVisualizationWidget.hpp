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
#include "SplineCurve.hpp"
#include "LibBCSim.hpp"  // for vector3

class QSlider;
class GLScattererWidget;
class IScattererModel;

// Widget used to visualize scatterers and possibly also the probe
// (and scanned sector image?)
class GLVisualizationWidget : public QWidget {
    Q_OBJECT
public:
    explicit GLVisualizationWidget(QWidget* parent = 0);

    // Set new scatterers of "spline" type.
    void setScattererSplines(const std::vector<SplineCurve<float, bcsim::vector3> >& scatterer);

    // Set new scatterers of "fixed" type.
    void setFixedScatterers(const std::vector<bcsim::vector3>& scatterers);

    // Set new scansequence
    void setScanSequence(bcsim::ScanSequence::s_ptr scan_seq);

public slots:
    void updateTimestamp(float time);

private:
    QSlider* createSlider();

private:
    GLScattererWidget*      glWidget;
    QSlider*                xSlider;
    QSlider*                ySlider;
    QSlider*                zSlider;
    QSlider*                camera_slider;

    // The scatterer data model
    QSharedPointer<IScattererModel> m_scatterer_model;
};

