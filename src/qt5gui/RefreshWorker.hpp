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
#include "../utils/cartesianator/Cartesianator.hpp"
#include "../utils/ScanGeometry.hpp"
#include "../utils/BCSimConvenience.hpp"

namespace refresh_worker {

// A safe variant of QImage owning its pixel data.
class SafeQImage {
public:
    SafeQImage()
        : m_pixels(nullptr)
    {
    }

    SafeQImage(const unsigned char* data, int width, int height, int bytes_per_sample, QImage::Format format) {
        const auto num_bytes = width*height*bytes_per_sample;
        m_pixels = new std::vector<unsigned char>;
        m_pixels->reserve(num_bytes);
        for (int i = 0; i < num_bytes; i++) {
            m_pixels->push_back(data[i]);
        }
        m_img = QImage(m_pixels->data(), width, height, width*bytes_per_sample, format);
    }

    QImage get_image() const {
        return m_img;
    }

private:
    QImage                      m_img;
    
    // Must be a pointer so that copying a SafeQImage is safe!
    std::vector<unsigned char>* m_pixels;
};

class WorkTask {
public:
    friend class Worker;
    typedef std::shared_ptr<WorkTask> ptr;

    virtual ~WorkTask() { }

    void set_geometry(bcsim::ScanGeometry::ptr geometry) {
        m_scan_geometry = geometry;
    }

    void set_dots_per_meter(float dpm) {
        m_dpm = dpm;
    }

private:
    bcsim::ScanGeometry::ptr            m_scan_geometry;
    float                               m_dpm;
};

class WorkTask_BMode : public WorkTask {
public:
    friend class Worker;
    typedef std::shared_ptr<WorkTask_BMode> ptr;

    void set_data(const std::vector<std::vector<std::complex<float>>>& data) {
        m_data = data;
    }

    void set_auto_normalize(bool status) {
        m_auto_normalize = status;
    }
    void set_normalize_const(float c) {
        m_normalize_const = c;
    }
    void set_gain(float gain) {
        m_gain = gain;
    }
    void set_dyn_range(float dyn_range) {
        m_dyn_range = dyn_range;
    }

private:
    std::vector<std::vector<std::complex<float>>>  m_data;
    bool                                m_auto_normalize;
    float                               m_normalize_const;
    float                               m_gain;
    float                               m_dyn_range;
};

class WorkTask_ColorDoppler : public WorkTask {
public:
    friend class Worker;
    typedef std::shared_ptr<WorkTask_ColorDoppler> ptr;

    void set_data(const std::vector<std::vector<std::vector<std::complex<float>>>>& data) {
        m_data = data;
    }

private:
    std::vector<std::vector<std::vector<std::complex<float>>>>  m_data;
};

class WorkResult {
public:
    friend class Worker;
    typedef std::shared_ptr<WorkResult> ptr;
    WorkResult() {
    }
    SafeQImage  image;
    float   updated_normalization_const;
};

class Worker : public QObject {
Q_OBJECT
public:
    Worker() : QObject() {
        // Create geometry converters
        m_cartesianator       = ICartesianator<unsigned char>::u_ptr(new CpuCartesianator<unsigned char>);
        m_color_cartesianator = ICartesianator<float>::u_ptr(new CpuCartesianator<float>); 
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
            auto work_task = m_queue.dequeue();
            if (std::dynamic_pointer_cast<WorkTask_BMode>(work_task)) {
                process(std::dynamic_pointer_cast<WorkTask_BMode>(work_task));
            } else if (std::dynamic_pointer_cast<WorkTask_ColorDoppler>(work_task)) {
                process(std::dynamic_pointer_cast<WorkTask_ColorDoppler>(work_task));
            } else {
                qDebug() << "Unable to cast WorkTask";
                throw std::logic_error("Unable to cast WorkTask");
            }
        }
    }

    void process(WorkTask_BMode::ptr work_task) {
        // Create output package
        auto work_result = WorkResult::ptr(new WorkResult);

        const auto& iq_data = work_task->m_data;

        // Transform complex IQ samples to real-valued envelope.
        if (iq_data.size() == 0) throw std::runtime_error("No lines returned");
        const auto num_iq_samples = iq_data[0].size();;
        std::vector<std::vector<float>> env_lines;
        for (size_t line_no = 0; line_no < iq_data.size(); line_no++) {
            std::vector<float> temp;
            temp.reserve(num_iq_samples);
            for (size_t i = 0; i < iq_data[line_no].size(); i++) {
                temp.push_back(std::abs(iq_data[line_no][i]));
            }
            env_lines.push_back(temp);
        }

        m_cartesianator->SetGeometry(work_task->m_scan_geometry);


        // update size of final image [pixels]
        float qimage_width_m, qimage_height_m;
        GetCartesianDimensions(work_task->m_scan_geometry, qimage_width_m, qimage_height_m);
        const auto qimage_dpm = work_task->m_dpm;
        const size_t width_pixels  = qimage_width_m*qimage_dpm;
        const size_t height_pixels = qimage_height_m*qimage_dpm;
    
        // As long as the size doesn't change, this call is not expensive.
        m_cartesianator->SetOutputSize(width_pixels, height_pixels);

        const size_t num_beams = env_lines.size();
        const size_t num_range = env_lines[0].size();
        if (num_beams <= 0) {
            throw std::runtime_error("No lines were returned");
        }
    

        if (work_task->m_auto_normalize) {
            work_result->updated_normalization_const = bcsim::get_max_value(env_lines);
        } else {
            work_result->updated_normalization_const = work_task->m_normalize_const;
        }

        // grayscale log-compression
        bcsim::log_compress_frame(env_lines, work_task->m_dyn_range,
                                    work_result->updated_normalization_const,
                                    work_task->m_gain);
            
        // Copy beamspace data in order to get correct memory layout,
        // i.e. sample index most rapidly varying. Convert to unsigned char.
        std::vector<unsigned char> beamspace_data(num_beams*num_range);
        for (size_t beam_no = 0; beam_no < num_beams; beam_no++) {
            std::transform(std::begin(env_lines[beam_no]),
                            std::end(env_lines[beam_no]),
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

        // copy output buffer
        const auto num_out_pixels = out_x*out_y;
        std::vector<unsigned char> temp_buffer(num_out_pixels);
        for (size_t i = 0; i < num_out_pixels; i++) {
            temp_buffer[i] = m_cartesianator->GetOutputBuffer()[i];
        }

        work_result->image = SafeQImage(m_cartesianator->GetOutputBuffer(),
                                        static_cast<int>(out_x),
                                        static_cast<int>(out_y),
                                        1,
                                        QImage::Format_Indexed8);
        emit finished_processing_bmode(work_result);
    }

    void process(WorkTask_ColorDoppler::ptr work_task) {
        // Create output package
        auto work_result = WorkResult::ptr(new WorkResult);

        const auto& iq_data = work_task->m_data;
        
        // Estimate R0 and R1
        std::vector<std::vector<float>> r0_lines;
        std::vector<std::vector<float>> velocity_lines;

        const auto packet_size    = iq_data.size();
        const auto num_iq_lines   = iq_data[0].size();
        const auto num_iq_samples = iq_data[0][0].size();;
        for (size_t line_no = 0; line_no < num_iq_lines; line_no++) {
            std::vector<float> temp1;
            std::vector<float> temp2;
            temp1.reserve(num_iq_samples);
            temp2.reserve(num_iq_samples);
            for (size_t i = 0; i < num_iq_samples; i++) {
                float r0 = 0.0f;
                std::complex<float> r1 = 0.0f;

                // simple clutter filter
                std::complex<float> mean_val(0.0f);
                for (int packet_no = 0; packet_no < packet_size; packet_no++) {
                    mean_val += iq_data[packet_no][line_no][i];
                }
                mean_val = mean_val/static_cast<float>(packet_size);

                // estimate R0
                for (int packet_no = 0; packet_no < packet_size; packet_no++) {
                    const auto z = iq_data[packet_no][line_no][i] - mean_val;
                    r0 += (z*std::conj(z)).real();
                }

                // estimate R1
                for (int packet_no = 0; packet_no < packet_size-1; packet_no++) {
                    r1 += std::conj(iq_data[packet_no][line_no][i])*iq_data[packet_no+1][line_no][i];
                }
                temp1.push_back(r0);
                temp2.push_back(std::arg(r1));
            }
            r0_lines.push_back(temp1);
            velocity_lines.push_back(temp2);
        }

        const auto max_r0_value = bcsim::get_max_value(r0_lines);
        qDebug() << "Max R0 value is " << max_r0_value;

        m_color_cartesianator->SetGeometry(work_task->m_scan_geometry);

        // update size of final image [pixels]
        float qimage_width_m, qimage_height_m;
        GetCartesianDimensions(work_task->m_scan_geometry, qimage_width_m, qimage_height_m);
        const auto qimage_dpm = work_task->m_dpm;
        const size_t width_pixels  = qimage_width_m*qimage_dpm;
        const size_t height_pixels = qimage_height_m*qimage_dpm;
    
        // As long as the size doesn't change, this call is not expensive.
        m_color_cartesianator->SetOutputSize(width_pixels, height_pixels);

        const size_t num_beams = r0_lines.size();
        const size_t num_range = r0_lines[0].size();
        if (num_beams <= 0) {
            throw std::runtime_error("No lines were returned");
        }
    
        // Copy beamspace data in order to get correct memory layout,
        // i.e. sample index most rapidly varying. Normalize power to [0, 1]
        std::vector<float> beamspace_data(num_beams*num_range);
        for (size_t beam_no = 0; beam_no < num_beams; beam_no++) {
            std::transform(std::begin(r0_lines[beam_no]),
                            std::end(r0_lines[beam_no]),
                            beamspace_data.data()+num_range*beam_no,
                            [=](float v) {
                return v/max_r0_value;
            });
        }

        // do geometry transform
        m_color_cartesianator->Process(beamspace_data.data(), static_cast<int>(num_beams), static_cast<int>(num_range));
    
        // make QImage from output of Cartesianator
        size_t out_x, out_y;
        m_color_cartesianator->GetOutputSize(out_x, out_y);

        // binary thresholding on R0
        const float normalized_threshold = 0.01f;

        const auto num_output_samples = out_x*out_y;
        std::vector<bool> thresholded_samples(num_output_samples);
        const auto out_ptr = m_color_cartesianator->GetOutputBuffer();
        for (size_t i = 0; i < num_output_samples; i++) {
            thresholded_samples[i] = (out_ptr[i] >= normalized_threshold);
        }

        // Reorganize memory layout
        for (size_t beam_no = 0; beam_no < num_beams; beam_no++) {
            std::transform(std::begin(velocity_lines[beam_no]),
                            std::end(velocity_lines[beam_no]),
                            beamspace_data.data()+num_range*beam_no,
                            [=](float v) {
                return v;
            });
        }

        // do geometry transform
        m_color_cartesianator->Process(beamspace_data.data(), static_cast<int>(num_beams), static_cast<int>(num_range));

        std::vector<unsigned char> color_pixels(4*num_output_samples);
        for (size_t i = 0; i < num_output_samples; i++) {
            unsigned char alpha = 0;
            unsigned char red;
            unsigned char green = 0;
            unsigned char blue;

            if (thresholded_samples[i]) {
                const auto phase_angle = m_color_cartesianator->GetOutputBuffer()[i];
                const auto color_value = static_cast<unsigned char>(255.0f*std::abs(phase_angle)/3.14159f); 
                alpha = 255;
                if (phase_angle >= 0.0) {
                    red  = color_value;
                    blue = 0;
                } else {
                    blue = color_value;
                    red  = 0;
                }
            }
            color_pixels[4*i + 3] = alpha;
            color_pixels[4*i + 2] = red;
            color_pixels[4*i + 1] = green;
            color_pixels[4*i + 0] = blue;
        }

        work_result->image = SafeQImage(color_pixels.data(),
                                        static_cast<int>(out_x),
                                        static_cast<int>(out_y),
                                        4,
                                        QImage::Format_ARGB32);
        emit finished_processing_color(work_result);
    }

    // finished processing a B-mode work item
    Q_SIGNAL void finished_processing_bmode(refresh_worker::WorkResult::ptr);

    Q_SIGNAL void finished_processing_color(refresh_worker::WorkResult::ptr);

private:
    QMutex                                  m_mutex;
    QQueue<WorkTask::ptr>                   m_queue;
    ICartesianator<unsigned char>::u_ptr    m_cartesianator;
    ICartesianator<float>::u_ptr            m_color_cartesianator;
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
        connect(&m_worker, SIGNAL(finished_processing_bmode(refresh_worker::WorkResult::ptr)),
                this, SIGNAL(processed_bmode_data_available(refresh_worker::WorkResult::ptr)));
        connect(&m_worker, SIGNAL(finished_processing_color(refresh_worker::WorkResult::ptr)),
                this, SIGNAL(processed_color_data_available(refresh_worker::WorkResult::ptr)));
    }

    // new beam space data for processing
    Q_SLOT void process_data(refresh_worker::WorkTask::ptr message) {
        m_worker.on_new_data(message);
    }

    // processed beam space data is ready
    Q_SIGNAL void processed_bmode_data_available(refresh_worker::WorkResult::ptr);

    Q_SIGNAL void processed_color_data_available(refresh_worker::WorkResult::ptr);

private:
    QThread     m_thread;
    QTimer      m_timer;
    Worker      m_worker;
};

}   // end namespace

Q_DECLARE_METATYPE(refresh_worker::WorkTask::ptr);
Q_DECLARE_METATYPE(refresh_worker::WorkResult::ptr);

