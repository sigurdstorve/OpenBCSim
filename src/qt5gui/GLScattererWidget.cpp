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

#include <stdexcept>
#include <QMouseEvent>
#include <QOpenGLShaderProgram>
#include <QCoreApplication>
#include <QDebug>
#include "GLScattererWidget.hpp"
#include "ScattererModel.hpp"
#include "ScanSeqModel.hpp"

GLScattererWidget::GLScattererWidget(QWidget* parent)
    : QOpenGLWidget(parent),
      m_xRot(0),
      m_yRot(0),
      m_zRot(0),
      m_camera_z(static_cast<int>(-0.18f*256))
{
    m_scatterer_data = QSharedPointer<IScattererModel>(new EmptyScattererModel);
    m_scanseq_data   = QSharedPointer<ScanSeqModel>(new ScanSeqModel);
}

GLScattererWidget::~GLScattererWidget() {
    cleanup();
}

QSize GLScattererWidget::minimumSizeHint() const {
    return QSize(50, 50);
}

QSize GLScattererWidget::sizeHint() const {
    return QSize(400, 400);
}

static void qNormalizeAngle(int& angle) {
    while (angle < 0) {
        angle += 360*16;
    }
    while (angle > 360*16) {
        angle -= 360*16;
    }
}

void GLScattererWidget::setXRotation(int angle) {
    qNormalizeAngle(angle);
    if (angle != m_xRot) {
        m_xRot = angle;
        emit xRotationChanged(angle);
        update();
    }
}

void GLScattererWidget::setYRotation(int angle) {
    qNormalizeAngle(angle);
    if (angle != m_yRot) {
        m_yRot = angle;
        emit yRotationChanged(angle);
        update();
    }
}

void GLScattererWidget::setZRotation(int angle) {
    qNormalizeAngle(angle);
    if (angle != m_zRot) {
        m_zRot = angle;
        emit zRotationChanged(angle);
        update();
    }
}

void GLScattererWidget::setCameraZ(int value) {
    if (value != m_camera_z) {
        m_camera_z = value;
        emit cameraZChanged(value);
        m_camera.setToIdentity();
        m_camera.translate(0, 0, m_camera_z/256.0f);
        update();
    }
}

void GLScattererWidget::cleanup() {
    makeCurrent();
    m_scatterers_vbo.destroy();
    m_scanseq_vbo.destroy();
    doneCurrent();
}

void GLScattererWidget::initializeGL() {
    // Handle reinitialize in case top-level widget changes
    connect(context(), &QOpenGLContext::aboutToBeDestroyed, this, &GLScattererWidget::cleanup);
    
    initializeOpenGLFunctions();
    
    qDebug("OpenGL initialized: version: %s GLSL: %s", glGetString(GL_VERSION), glGetString(GL_SHADING_LANGUAGE_VERSION));

    glClearColor(0, 0, 0, m_transparent ? 0 : 1);
    
    // Make shader program for scatterers
    m_program = std::make_unique<QOpenGLShaderProgram>();
    if (!m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/scatterer_vshader.glsl")) {
        throw std::runtime_error("Failed to compile vertex shader");
    }
    if (!m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/scatterer_fshader.glsl")) {
        throw std::runtime_error("Failed to compile fragment shader");
    }
    m_program->bindAttributeLocation("vertex", 0);
    m_program->bindAttributeLocation("normal", 1);
    if (!m_program->link()) {
        throw std::runtime_error("Failed to link shader program");
    }

    // Make shader program for scan sequence
    m_scanseq_program = std::make_unique<QOpenGLShaderProgram>();
    if (!m_scanseq_program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/scanseq_vshader.glsl")) {
        throw std::runtime_error("Failed to compile scanseq vertex shader");
    }
    if (!m_scanseq_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/scanseq_fshader.glsl")) {
        throw std::runtime_error("Failed to compile scanseq fragment shader");
    }
    m_scanseq_program->bindAttributeLocation("vertex", 0);
    m_scanseq_program->bindAttributeLocation("normal", 1);
    if (!m_scanseq_program->link()) {
        throw std::runtime_error("Failed to link scanseq shader program");
    }
    
    // Make sure there is a VAO when one is needed. 
    m_vao.create();
    QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);

    // Setup our vertex buffer object for scatterers
    if (!m_scatterers_vbo.create()) {
        throw std::runtime_error("Unable to create VBO for scatterers");
    }
    if (!m_scatterers_vbo.bind()) {
        throw std::runtime_error("Unable to bind VBO for scatterers");
    }
    m_scatterers_vbo.allocate(m_scatterer_data->constData(), m_scatterer_data->count()*sizeof(GLfloat));
    
    // Setup out vertex buffer object for scanseq
    if (!m_scanseq_vbo.create()) {
        throw std::runtime_error("Unable to create VBO for scansequence");
    }
    if (!m_scanseq_vbo.bind()) {
        throw std::runtime_error("Unable to bind VBO for scansequence");
    }
    m_scanseq_vbo.allocate(m_scanseq_data->constData(), m_scanseq_data->count()*sizeof(GLfloat));

    // Our camera never changes in this example
    m_camera.setToIdentity();
    m_camera.translate(0, 0, m_camera_z/256.0f);
    emit cameraZChanged(m_camera_z);

    // Light position is fixed. This value is used in the fragment shader
    m_program->bind();
    m_program->setUniformValue(m_program->uniformLocation("lightPos"), QVector3D(0, 0, 70));
    m_program->release();

    // Set light position for scanseq also
    m_scanseq_program->bind();
    m_scanseq_program->setUniformValue(m_scanseq_program->uniformLocation("lightPos"), QVector3D(0, 0, 70));
    m_scanseq_program->release();
}

void GLScattererWidget::paintGL() {
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glEnable(GL_PROGRAM_POINT_SIZE);
    m_world.setToIdentity();
    m_world.rotate(180.0f - (m_xRot/16.0f), 1, 0, 0);
    m_world.rotate(m_yRot / 16.0f, 0, 1, 0);
    m_world.rotate(m_zRot / 16.0f, 0, 0, 1);
    
    QOpenGLVertexArrayObject::Binder vaoBinder(&m_vao);

    if (!m_program->bind()) {
        throw std::runtime_error("Failed to bind scatterer program");
    }
    m_scatterers_vbo.bind();

    m_program->setUniformValue(m_program->uniformLocation("projMatrix"), m_proj);
    m_program->setUniformValue(m_program->uniformLocation("mvMatrix"), m_camera*m_world);
    m_program->setUniformValue(m_program->uniformLocation("normalMatrix"), m_world.normalMatrix());
    
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    // Remember that "vertex"~0 and "normal"~1
    f->glEnableVertexAttribArray(0);
    f->glEnableVertexAttribArray(1);
    f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), 0);
    f->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), reinterpret_cast<void*>(3*sizeof(GLfloat)));
    glDrawArrays(GL_TRIANGLES, 0, m_scatterer_data->vertexCount());
    m_scatterers_vbo.release();
    m_program->release();

    // Draw scan sequence
    if (!m_scanseq_program->bind()) {
        throw std::runtime_error("Failed to bind scanseq program");
    }
    m_scanseq_vbo.bind();

    m_scanseq_program->setUniformValue(m_scanseq_program->uniformLocation("projMatrix"), m_proj);
    m_scanseq_program->setUniformValue(m_scanseq_program->uniformLocation("mvMatrix"), m_camera*m_world);
    m_scanseq_program->setUniformValue(m_scanseq_program->uniformLocation("normalMatrix"), m_world.normalMatrix());
    
    f->glEnableVertexAttribArray(0);
    f->glEnableVertexAttribArray(1);
    f->glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), 0);
    f->glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), reinterpret_cast<void*>(3*sizeof(GLfloat)));
    glDrawArrays(GL_TRIANGLES, 0, m_scanseq_data->vertexCount());
    m_scanseq_vbo.release();
    m_scanseq_program->release();
}

void GLScattererWidget::resizeGL(int w, int h) {
    m_proj.setToIdentity();
    m_proj.perspective(45.0f, GLfloat(w)/h, 0.01f, 100.0f);    
}

void GLScattererWidget::mousePressEvent(QMouseEvent* event) {
    m_lastPos = event->pos();
}

void GLScattererWidget::mouseMoveEvent(QMouseEvent* event) {
    int dx = event->x() - m_lastPos.x();
    int dy = event->y() - m_lastPos.y();
    
    if (event->buttons() & Qt::LeftButton) {
        setXRotation(m_xRot + 8*dy);
        setYRotation(m_yRot + 8*dx);
    } else if (event->buttons() & Qt::RightButton) {
        setXRotation(m_xRot + 8*dy);
        setZRotation(m_zRot + 8*dx);
    }
    m_lastPos = event->pos();
}

void GLScattererWidget::updateFromModel() {
    // Copy updated vertices
    m_scatterers_vbo.bind();
    m_scatterers_vbo.allocate(m_scatterer_data->constData(), m_scatterer_data->count()*sizeof(GLfloat));
    m_scatterers_vbo.release();
    
    // TODO: Should this be here or should the caller do it?
    update();    
}

void GLScattererWidget::setScanSequence(bcsim::ScanSequence::s_ptr scan_seq) {
    m_scanseq_data->setScanSequence(scan_seq);

    // Copy updated vertices
    m_scanseq_vbo.bind();
    m_scanseq_vbo.allocate(m_scanseq_data->constData(), m_scanseq_data->count()*sizeof(GLfloat));
    m_scanseq_vbo.release();

    update();
}

void GLScattererWidget::setScatterers(QSharedPointer<IScattererModel> scatterers) {
    m_scatterer_data = scatterers;
}
