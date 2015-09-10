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
        m_data.clear();
        const float line_length = scan_seq->line_length;
        for (int line_no = 0; line_no < scan_seq->get_num_lines(); line_no++) {
            auto scanline = scan_seq->get_scanline(line_no);
            const auto temp_origin = scanline.get_origin();
            const float x0 = temp_origin.x;
            const float y0 = temp_origin.y;
            const float z0 = temp_origin.z;
            const QVector3D start(x0, y0, z0);
            const auto temp_direction = scanline.get_direction();
            const QVector3D end(x0+temp_direction.x*line_length,
                                y0+temp_direction.y*line_length,
                                z0+temp_direction.z*line_length);
            add_line(start, end);
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
    void add_line(const QVector3D& start, const QVector3D& end) {
        add(start);
        add(end);
    }

    void add(const QVector3D& point) {
        m_data.push_back(point.x());
        m_data.push_back(point.y());
        m_data.push_back(point.z());
        // Dummy normal vector
        m_data.push_back(1.0f);
        m_data.push_back(0.0f);
        m_data.push_back(0.0f);
    }
private:
    QVector<GLfloat>    m_data;
};