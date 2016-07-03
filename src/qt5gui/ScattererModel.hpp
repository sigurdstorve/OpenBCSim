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
#include <qopengl.h>
#include <QVector>
#include <QVector3D>
#include "SplineCurve.hpp"
#include "../core/LibBCSim.hpp"
#include "trianglemesh3d/TriangleMesh3d.hpp"
#include "IConfig.hpp"

// Interface for something capable of returning data for consumption by OpenGL
class IScattererModel {
public:
    virtual ~IScattererModel() { }

    // Pointer to data [p_x, p_y, p_z, n_x, n_y, n_z, ...] for all vertices
    // and associated normal vectors.
    virtual const GLfloat* constData() const    = 0;

    // Number of floats
    virtual int count() const                   = 0;

    // Number of vertices
    virtual int vertexCount() const             = 0;
};

// Functionality which is common for both SplineScattererModel and FixedScattererModel.
class BaseScattererModel {
public:
    BaseScattererModel(trianglemesh3d::ITriangleMesh3d::u_ptr scatterer_template, IConfig::s_ptr& cfg) {
        m_scatterer_radius = cfg->get_double("scatterer_radius", 1.2e-3);
		precompute_template(std::move(scatterer_template));
    }

protected:
    // Precompute vertices and normals of a template scatterer model.
	void precompute_template(trianglemesh3d::ITriangleMesh3d::u_ptr scatterer_template);

    // Creates random normalized normal vectors.
    QVector<QVector3D> generateRandomNormalVectors(int num_vectors);

protected:
    // Precomputed data for a cube with unit radius
    QVector<GLfloat>                                m_cube_points;
    QVector<GLfloat>                                m_cube_normals;
    double                                          m_scatterer_radius;
};

// Dummy empty model.
class EmptyScattererModel : public IScattererModel {
public:
    virtual const GLfloat* constData() const {
        return nullptr;
    }

    virtual int count() const {
        return 0;
    }

    virtual int vertexCount() const {
        return 0;
    }
};

class SplineScattererModel : public  IScattererModel,
                             private BaseScattererModel {
public:
    SplineScattererModel(trianglemesh3d::ITriangleMesh3d::u_ptr template_model, IConfig::s_ptr& cfg)
		: BaseScattererModel(std::move(template_model), cfg) {  }

    virtual const GLfloat* constData() const {
        return m_data.constData();
    }
    
    virtual int count() const {
        return m_data.size();
    }
    
    virtual int vertexCount() const {
        return m_data.size()/6;
    }

    // Recompute scatterer positions for a new timestamp.
    void setTimestamp(float timestamp);

    // Use a new collection of spline scatterers.
    void setSplines(const std::vector<SplineCurve<float, bcsim::vector3> >& splines);

private:
    // Vertex and normal data that will be passed to VBO
    QVector<GLfloat>                                m_data;
    std::vector<SplineCurve<float, bcsim::vector3> > m_splines;
};

class FixedScattererModel : public  IScattererModel,
                            private BaseScattererModel {
public:
    FixedScattererModel(trianglemesh3d::ITriangleMesh3d::u_ptr template_model, IConfig::s_ptr& cfg)
		: BaseScattererModel(std::move(template_model), cfg)	{ }

    virtual const GLfloat* constData() const {
        return m_data.constData();
    }
    
    virtual int count() const {
        return m_data.size();
    }
    
    virtual int vertexCount() const {
        return m_data.size()/6;
    }

    // Use a new collection of 3D points for visualization.
    void setPoints(const std::vector<bcsim::vector3>& points);

private:
    // Vertex and normal data that will be passed to VBO
    QVector<GLfloat>                                m_data;
};