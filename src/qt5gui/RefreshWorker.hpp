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
#include <stdexcept>
#include <memory>
#include <QObject>
#include <QDebug>
#include <QThread>
#include <QTimer>
#include <QMutex>
#include <QQueue>
#include <QMutexLocker>
#include <QImage>
#include "cartesianator/Cartesianator.hpp"
#include "ScanGeometry.hpp"
#include "BCSimConvenience.hpp"

namespace refresh_worker {

class WorkTask {
public:
    friend class Worker;
    typedef std::shared_ptr<WorkTask> ptr;
    WorkTask() {
    }
    void set_geometry(bcsim::ScanGeometry::ptr geometry) {
        m_scan_geometry = geometry;
    }
    void set_data(const std::vector<std::vector<float>>& data) {
        m_data = data;
    }
    void set_auto_normalize(bool status) {
        m_auto_normalize = status;
    }
    void set_normalize_const(float c) {
        m_normalize_const = c;
    }
    void set_dots_per_meter(float dpm) {
        m_dpm = dpm;
    }
    void set_gain(float gain) {
        m_gain = gain;
    }
    void set_dyn_range(float dyn_range) {
        m_dyn_range = dyn_range;
    }
private:
    std::vector<std::vector<float>>  m_data;
    bcsim::ScanGeometry::ptr            m_scan_geometry;
    bool                                m_auto_normalize;
    float                               m_normalize_const;
    float                               m_dpm;
    float                               m_gain;
    float                               m_dyn_range;
};

class WorkResult {
public:
    friend class Worker;
    typedef std::shared_ptr<WorkResult> ptr;
    WorkResult() {
    }
    QImage  image;
    float   updated_normalization_const;
};

class Worker : public QObject {
Q_OBJECT
public:
    Worker() : QObject() {
        // Create geometry converter
        m_cartesianator = ICartesianator::u_ptr(new CpuCartesianator);
    }

    // enqueue new work item
    Q_SLOT void on_new_data(refresh_worker::WorkTask::ptr msg) {
        QMutexLocker mutex_locker(&m_mutex);
        m_queue.enqueue(msg);
    }
    
private:
    Q_SLOT void on_timeout() {
        QMutexLocker mutex_locker(&m_mutex);
        auto num_elements = m_queue.size();
        if (!m_queue.isEmpty()) {
            // Create output package
            auto work_result = WorkResult::ptr(new WorkResult);
            
            auto work_task = m_queue.dequeue();
            auto& rf_lines = work_task->m_data;

            m_cartesianator->SetGeometry(work_task->m_scan_geometry);


            // update size of final image [pixels]
            float qimage_width_m, qimage_height_m;
            GetCartesianDimensions(work_task->m_scan_geometry, qimage_width_m, qimage_height_m);
            const auto qimage_dpm = work_task->m_dpm;
            const size_t width_pixels  = qimage_width_m*qimage_dpm;
            const size_t height_pixels = qimage_height_m*qimage_dpm;
    
            // As long as the size doesn't change, this call is not expensive.
            m_cartesianator->SetOutputSize(width_pixels, height_pixels);

            const size_t num_beams = rf_lines.size();
            const size_t num_range = rf_lines[0].size();
            if (num_beams <= 0) {
                throw std::runtime_error("No lines were returned");
            }
    

            if (work_task->m_auto_normalize) {
                work_result->updated_normalization_const = bcsim::get_max_value(rf_lines);
            } else {
                work_result->updated_normalization_const = work_task->m_normalize_const;
            }

            // grayscale log-compression
            bcsim::log_compress_frame(rf_lines, work_task->m_dyn_range,
                                      work_result->updated_normalization_const,
                                      work_task->m_gain);
            
            // Copy beamspace data in order to get correct memory layout,
            // i.e. sample index most rapidly varying. Convert to unsigned char.
            std::vector<unsigned char> beamspace_data(num_beams*num_range);
            for (size_t beam_no = 0; beam_no < num_beams; beam_no++) {
                std::transform(std::begin(rf_lines[beam_no]),
                               std::end(rf_lines[beam_no]),
                               beamspace_data.data()+num_range*beam_no,
                               [=](float v) {
                    return static_cast<unsigned char>(v);
                });
            }

            // do geometry transform
            m_cartesianator->Process(beamspace_data.data(), static_cast<int>(num_beams), static_cast<int>(num_range));
    
            // make QImage from output of Cartesianator
            size_t out_x, out_y;
            m_cartesianator->GetOutputSize(out_x, out_y);
            work_result->image = QImage(m_cartesianator->GetOutputBuffer(),
                                        static_cast<int>(out_x),
                                        static_cast<int>(out_y),
                                        static_cast<int>(out_x),
                                        QImage::Format_Indexed8);
            emit finished_processing(work_result);
        }
    }

    // finished processing work item
    Q_SIGNAL void finished_processing(refresh_worker::WorkResult::ptr);

private:
    QMutex                      m_mutex;
    QQueue<WorkTask::ptr>       m_queue;
    ICartesianator::u_ptr       m_cartesianator;
};

class RefreshWorker : public QObject {
Q_OBJECT
public:
    RefreshWorker(int millisec) {
        connect(&m_timer, SIGNAL(timeout()), &m_worker, SLOT(on_timeout()));
        m_timer.start(millisec);
        m_timer.moveToThread(&m_thread);    // not neccessary according to tutorial.
        m_worker.moveToThread(&m_thread);
        m_thread.start();
        connect(&m_worker, SIGNAL(finished_processing(refresh_worker::WorkResult::ptr)),
                this, SIGNAL(processed_data_available(refresh_worker::WorkResult::ptr)));
    }

    // new beam space data for processing
    Q_SLOT void process_data(refresh_worker::WorkTask::ptr message) {
        m_worker.on_new_data(message);
    }

    // processed beam space data is ready
    Q_SIGNAL void processed_data_available(refresh_worker::WorkResult::ptr);

private:
    QThread     m_thread;
    QTimer      m_timer;
    Worker      m_worker;
};

}   // end namespace

Q_DECLARE_METATYPE(refresh_worker::WorkTask::ptr);
Q_DECLARE_METATYPE(refresh_worker::WorkResult::ptr);

