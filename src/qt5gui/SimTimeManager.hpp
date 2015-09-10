#pragma once
#include <memory>
#include <QObject>

// Handler for simulation time in interval with wrap-around.
class SimTimeManager : public QObject {
Q_OBJECT
public:
    typedef std::unique_ptr<SimTimeManager> ptr;

    SimTimeManager(double min_time, double max_time)
        : m_min_time(min_time), m_max_time(max_time)
    { }

    // set min time
    void set_min_time(double new_min_time) {
        m_min_time = new_min_time;
        emit min_time_changed(new_min_time);
    }

    // set max time
    void set_max_time(double new_max_time) {
        m_max_time = new_max_time;
        emit max_time_changed(new_max_time);
    }

    // returns current simulation time.
    double get_time() const {
        return m_time;
    }

    // set time increment from frame to frame
    void set_time_delta(double new_dt) {
        m_dt = new_dt;
    }

    // moves one time step forward
    void advance() {
        m_time += m_dt;
        enforce_wrap_around();
        emit time_changed(get_time());
    }

    // set time at start of interval
    void reset() {
        m_time = m_min_time;
        emit time_changed(get_time());
    }

    double get_min_time() const {
        return m_min_time;
    }

    double get_max_time() const {
        return m_max_time;
    }

public slots:
    // set current simulation time
    void set_time(double new_time) {
        m_time = new_time;
        enforce_wrap_around();
        emit time_changed(m_time);
    }

signals:
    // minimum time has changed
    void min_time_changed(double new_value);

    // maximum time has changed
    void max_time_changed(double new_value);

    // current time has changed
    void time_changed(double new_value);

protected:
    void enforce_wrap_around() {
        const auto temp_time = m_time - m_min_time;
        const auto time_len = m_max_time - m_min_time;
        m_time = std::fmod(m_time, time_len) + m_min_time;
    }

private:
    double  m_time;
    double  m_dt;
    double  m_min_time;
    double  m_max_time;
};