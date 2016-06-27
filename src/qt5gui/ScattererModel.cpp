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

#include <random>
#include "ScattererModel.hpp"

void BaseScattererModel::precompute_template(trianglemesh3d::ITriangleMesh3d::u_ptr template_model) {
    QVector<GLfloat> ps; // points
    QVector<GLfloat> ns; // normals

	// do data conversion
	const auto num_vertices = template_model->num_vertices();
	for (size_t i = 0; i < 3*num_vertices; i++) {
		m_cube_points.push_back(static_cast<GLfloat>(template_model->vertex_data()[i]));
		m_cube_normals.push_back(static_cast<GLfloat>(template_model->normal_data()[i]));
	}
}


void SplineScattererModel::setTimestamp(float timestamp) {
    size_t num_splines = m_splines.size();
    m_data.clear();
        
    // Each scatterer has one point and one normal vector
    m_data.reserve(num_splines*2*3);
    
    // Evaluate all splines in timestamp
    const float radius = 0.1e-3;
    for (size_t i = 0; i < num_splines; i++) {
        // Evaluate scatterer position
        bcsim::vector3 p = RenderCurve<float, bcsim::vector3>(m_splines[i], timestamp);
        
        // Add a correctly positioned cube in each scatterer position
        for (size_t i = 0; i < m_cube_points.size()/3; i++) {
            m_data.push_back(m_cube_points[3*i]  *radius + p.x);
            m_data.push_back(m_cube_points[3*i+1]*radius + p.y);
            m_data.push_back(m_cube_points[3*i+2]*radius + p.z);
            m_data.push_back(m_cube_normals[3*i]);
            m_data.push_back(m_cube_normals[3*i+1]);
            m_data.push_back(m_cube_normals[3*i+2]);
        }
    }
}

void SplineScattererModel::setSplines(const std::vector<SplineCurve<float, bcsim::vector3> >& splines) {
    size_t num_splines = splines.size();
    m_splines = splines;
}

void FixedScattererModel::setPoints(const std::vector<bcsim::vector3>& points) {
    size_t num_points = points.size();
    m_data.clear();
        
    // Each scatterer has one point and one normal vector
    m_data.reserve(num_points*2*3);
    
    // Evaluate all splines in timestamp
    const float radius = 0.1e-3;
    for (size_t i = 0; i < num_points; i++) {
        auto p = points[i];
        
        // Add a correctly positioned cube in each scatterer position
        for (size_t i = 0; i < m_cube_points.size()/3; i++) {
            m_data.push_back(m_cube_points[3*i]  *radius + p.x);
            m_data.push_back(m_cube_points[3*i+1]*radius + p.y);
            m_data.push_back(m_cube_points[3*i+2]*radius + p.z);
            m_data.push_back(m_cube_normals[3*i]);
            m_data.push_back(m_cube_normals[3*i+1]);
            m_data.push_back(m_cube_normals[3*i+2]);
        }
    }
}
