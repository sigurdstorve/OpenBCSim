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
#include <QOpenGLWidget>
#include <QOpenGLFunctions> // provides cross-platform access to the OpenGL ES 2.0 API.
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QMatrix4x4>
#include "SplineCurve.hpp"
#include "../core/LibBCSim.hpp"

class QOpenGLShaderProgram;
class IScattererModel;
class ScanSeqModel;

// This class should only have to worry about IScattererModel,
// i.e. instead of havingScattererTimestamp() and setScattererSpline()
// [which are specific to *spline* scatterers] it should accept an
// IScattererModel [which *then* can internally be of type spline]
// from the user.
class GLScattererWidget : public QOpenGLWidget,
                          protected QOpenGLFunctions {
    Q_OBJECT
public:
    GLScattererWidget(QWidget* parent = 0);
    ~GLScattererWidget();
    
    QSize minimumSizeHint() const Q_DECL_OVERRIDE;
    QSize sizeHint() const Q_DECL_OVERRIDE;

    // Register the data model.
    void setScatterers(QSharedPointer<IScattererModel> scatterers);

    void setScanSequence(bcsim::ScanSequence::s_ptr scan_seq);

    // Must be called when the scatterer data model has changed in order to
    // refresh the visualization.
    void updateFromModel();

public slots:
    void setXRotation(int angle);
    void setYRotation(int angle);
    void setZRotation(int angle);
    void setCameraZ(int value);
    void cleanup();

signals:
    void xRotationChanged(int angle);
    void yRotationChanged(int angle);
    void zRotationChanged(int angle);
    void cameraZChanged(int value);

protected:
    void initializeGL() Q_DECL_OVERRIDE;
    void paintGL() Q_DECL_OVERRIDE;
    void resizeGL(int width, int height) Q_DECL_OVERRIDE;
    void mousePressEvent(QMouseEvent* event) Q_DECL_OVERRIDE;
    void mouseMoveEvent(QMouseEvent* event) Q_DECL_OVERRIDE;

private:
    void setupVertexAttribs();

private:
    QOpenGLShaderProgram*           m_program;
    int                             m_projMatrixLoc;
    int                             m_mvMatrixLoc;
    int                             m_normalMatrixLoc;
    int                             m_lightPosLoc;
    QMatrix4x4                      m_proj;
    QMatrix4x4                      m_camera;
    QMatrix4x4                      m_world;
    bool                            m_transparent;

    // Data related to rotation control.
    int                             m_xRot;
    int                             m_yRot;
    int                             m_zRot;
    int                             m_camera_z;
    QPoint                          m_lastPos;
    
    // Data related to scatterers
    QSharedPointer<IScattererModel> m_scatterer_data;
    QOpenGLVertexArrayObject        m_vao;
    QOpenGLBuffer                   m_logoVbo;
    
    // Data related to scan sequence
    QSharedPointer<ScanSeqModel>    m_scanseq_data;
    QOpenGLBuffer                   m_scanseq_vbo;
};


