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

class ScanSeqModel {
public:
    ScanSeqModel() {
    }

    void setScanSequence(bcsim::ScanSequence::s_ptr scan_seq) {
        // This is a HACK which works because the scanlines are ordered
        // TODO: solve in a better way!

        m_data.clear();
        const auto line_length = scan_seq->line_length;
        const auto num_lines = scan_seq->get_num_lines();
        
        for (int line_no = 0; line_no < num_lines-1; line_no++) {
            auto line0 = scan_seq->get_scanline(line_no);
            auto line1 = scan_seq->get_scanline(line_no + 1);

            // HACK: If they start in the same origo, we will draw a triangle
            // If not, we will draw a quad as two triangles.

            const auto o0 = line0.get_origin();
            const auto o1 = line1.get_origin();
            const auto ur0 = line0.get_direction();
            const auto ur1 = line1.get_direction();
            if ((line0.get_origin() - line1.get_origin()).norm() < 1e-6) {
                // assume sector scan
                add_triangle(o0, o0 + ur0*line_length, o0 + ur1*line_length, line0.get_elevational_dir());
            } else {
                // assume linear scan
                add_quad(o0, o0 + ur0*line_length, o1 + ur1*line_length, o1, line0.get_elevational_dir());
            }
        }
    }

    int count() const {
        return m_data.size();
    }

    int vertexCount() const {
        return m_data.size() / 6;
    }

    const GLfloat* constData() const {
        return m_data.constData();
    }
private:
    void add_triangle(const bcsim::vector3& p0,
                      const bcsim::vector3& p1,
                      const bcsim::vector3& p2,
                      const bcsim::vector3& normal) {
        add_vertex(p0, normal);
        add_vertex(p1, normal);
        add_vertex(p2, normal);
    }

    void add_quad(const bcsim::vector3& p0,
                  const bcsim::vector3& p1,
                  const bcsim::vector3& p2,
                  const bcsim::vector3& p3,
                  const bcsim::vector3& normal) {
        add_triangle(p0, p1, p3, normal);
        add_triangle(p3, p1, p2, normal);
    }

    void add_vertex(const bcsim::vector3& point, const bcsim::vector3& normal) {
        m_data.push_back(point.x);
        m_data.push_back(point.y);
        m_data.push_back(point.z);
        m_data.push_back(normal.x);
        m_data.push_back(normal.y);
        m_data.push_back(normal.z);
    }
private:
    QVector<GLfloat>    m_data;
};