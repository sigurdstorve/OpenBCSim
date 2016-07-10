#pragma once
#include <QHBoxLayout>
#include <QWidget>
#include <QSlider>

class SimTimeWidget : public QWidget {
Q_OBJECT
public:
    SimTimeWidget(QWidget* parent=0, Qt::WindowFlags f=0) 
        : QWidget(parent, f)
    {
        // internally using discete steps
        m_num_steps = 1000;

        auto layout = new QHBoxLayout;
        m_slider = new QSlider(Qt::Horizontal);
        m_slider->setMinimum(0);
        m_slider->setMaximum(m_num_steps);
        connect(m_slider, SIGNAL(valueChanged(int)), this, SLOT(on_slider_changed(int)));
        layout->addWidget(m_slider);
        setLayout(layout);
    }

public slots:
    void set_min_time(double new_value) {
        m_min_time = new_value;
    }

    void set_max_time(double new_value) {
        m_max_time = new_value;
    }

    void set_time(double new_time) {
        // only set value if it is a new value
        const auto current_int_value = m_slider->value();
        const auto new_int_value = to_int(new_time);
        if (new_int_value != current_int_value) {
            m_slider->setValue(new_int_value);
        }
    }

signals:
    void time_changed(double new_value);

private:
    double to_float(int slider_value) {
        auto min = m_slider->minimum();
        auto max = m_slider->maximum();
        float norm_pos = static_cast<double>(slider_value)/(max-min);
        float time = m_min_time + norm_pos*(m_max_time-m_min_time);
        return time;
    }

    int to_int(double time_value) {
        return m_num_steps*(time_value-m_min_time)/(m_max_time-m_min_time);
    }

private slots:
    void on_slider_changed(int new_value) {
        const auto new_time_value = to_float(new_value);
        emit time_changed(new_time_value);
    }

private:
    QSlider*    m_slider;
    int         m_num_steps;
    double      m_min_time;
    double      m_max_time;
};


