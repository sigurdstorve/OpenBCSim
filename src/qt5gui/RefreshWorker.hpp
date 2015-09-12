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
#include <QObject>
#include <QDebug>
#include <QThread>
#include <QTimer>
#include <QMutex>
#include <QQueue>
#include <QMutexLocker>

class Worker : public QObject {
Q_OBJECT
public:
    Worker() : QObject() { }

public slots:
    // enqueue new work item
    void on_new_data(int data) {
        QMutexLocker mutex_locker(&m_mutex);
        m_queue.enqueue(data);
    }

private slots:
    void on_timeout() {
        QMutexLocker mutex_locker(&m_mutex);
        auto num_elements = m_queue.size();
        qDebug() << "Refresh timeout @ thread " << QThread::currentThreadId() << ". Number of queued items is" << num_elements;
        if (!m_queue.isEmpty()) {
            auto element = m_queue.dequeue();
            emit finished_processing(42);
        }
    }

signals:
    // finished processing work item
    void finished_processing(int);

private:
    QMutex          m_mutex;
    QQueue<int>     m_queue;
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
        connect(&m_worker, SIGNAL(finished_processing(int)), this, SIGNAL(processed_data_available(int)));
    }

public slots:
    // TODO: accept data and geometry
    void process_data(int item) {
        m_worker.on_new_data(item);
    }

signals:
    // Emitted when a processed frame becomes available.
    void processed_data_available(int);

private:
    QThread     m_thread;
    QTimer      m_timer;
    Worker      m_worker;
};

