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
#include <memory>
#include <QObject>
#include <QDebug>
#include <QThread>
#include <QTimer>
#include <QMutex>
#include <QQueue>
#include <QMutexLocker>
#include <QPixmap>

namespace refresh_worker {

class WorkTask {
public:
    friend class Worker;
    typedef std::shared_ptr<WorkTask> ptr;
    WorkTask() {
    }
    ~WorkTask() {
        qDebug() << "WorkTask was destroyed";
    }
    void set_geometry(bcsim::ScanGeometry::ptr geometry) {
        m_geometry = geometry;
    }
    void set_data(const std::vector<std::vector<bc_float>>& data) {
        m_data = data;
    }
private:
    std::vector<std::vector<bc_float>>  m_data;
    bcsim::ScanGeometry::ptr            m_geometry;
};

class WorkResult {
public:
    friend class Worker;
    typedef std::shared_ptr<WorkResult> ptr;
    WorkResult() {
    }
    ~WorkResult() {
        qDebug() << "WorkResult was destroyed";
    }
    QPixmap pixmap;
};

Q_DECLARE_METATYPE(WorkTask::ptr);
Q_DECLARE_METATYPE(WorkResult::ptr);

class Worker : public QObject {
Q_OBJECT
public:
    Worker() : QObject() { }

public slots:
    // enqueue new work item
    void on_new_data(WorkTask::ptr msg) {
        QMutexLocker mutex_locker(&m_mutex);
        m_queue.enqueue(msg);
    }
    
private slots:
    void on_timeout() {
        QMutexLocker mutex_locker(&m_mutex);
        auto num_elements = m_queue.size();
        qDebug() << "Refresh timeout @ thread " << QThread::currentThreadId() << ". Number of queued items is" << num_elements;
        if (!m_queue.isEmpty()) {
            auto work_task = m_queue.dequeue();
            qDebug() << "Number of RF lines: " << work_task->m_data.size();
            qDebug() << "Number of samples: " << work_task->m_data[0].size();
            
            // TODO: do work
            
            // Create output package
            auto work_result = WorkResult::ptr(new WorkResult);
            work_result->pixmap = QPixmap::fromImage(QImage("d:/Tux.png"));
            emit finished_processing(work_result);
        }
    }

signals:
    // finished processing work item
    void finished_processing(WorkResult::ptr);

private:
    QMutex                      m_mutex;
    QQueue<WorkTask::ptr>       m_queue;
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
        connect(&m_worker, SIGNAL(finished_processing(WorkResult::ptr)), this, SIGNAL(processed_data_available(WorkResult::ptr)));
    }

public slots:
    // new beam space data for processing
    void process_data(WorkTask::ptr message) {
        m_worker.on_new_data(message);
    }

signals:
    // processed beam space data is ready
    void processed_data_available(WorkResult::ptr);

private:
    QThread     m_thread;
    QTimer      m_timer;
    Worker      m_worker;
};

}   // end namespace
