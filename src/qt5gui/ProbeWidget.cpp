#include <stdexcept>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QComboBox>
#include <QGroupBox>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include "ProbeWidget.hpp"

FixedProbeWidget::FixedProbeWidget(QWidget* parent, Qt::WindowFlags f)
    : QWidget(parent, f)
{
    auto mainlayout = new QVBoxLayout;
    auto groupbox = new QGroupBox("Fixed Probe Position/Orientation");
    auto form_layout = new QFormLayout;
    
    m_origin_x_sb = new QDoubleSpinBox;
    m_origin_x_sb->setRange(-1.0, 1.0);
    m_origin_x_sb->setValue(0.0);
    m_origin_x_sb->setSingleStep(0.001);
    m_origin_x_sb->setDecimals(3);

    m_origin_y_sb = new QDoubleSpinBox;
    m_origin_y_sb->setRange(-1.0, 1.0);
    m_origin_y_sb->setValue(0.0);
    m_origin_y_sb->setSingleStep(0.001);
    m_origin_y_sb->setDecimals(3);

    m_origin_z_sb = new QDoubleSpinBox;
    m_origin_z_sb->setRange(-1.0, 1.0);
    m_origin_z_sb->setValue(0.0);
    m_origin_z_sb->setSingleStep(0.001);
    m_origin_z_sb->setDecimals(3);
    
    m_rot_x_sb = new QDoubleSpinBox;
    m_rot_x_sb->setRange(-10, 10);
    m_rot_x_sb->setValue(0.0);
    m_rot_x_sb->setSingleStep(0.05);
    m_rot_x_sb->setDecimals(3);

    m_rot_y_sb = new QDoubleSpinBox;
    m_rot_y_sb->setRange(-10, 10);
    m_rot_y_sb->setValue(0.0);
    m_rot_y_sb->setSingleStep(0.05);
    m_rot_y_sb->setDecimals(3);

    m_rot_z_sb = new QDoubleSpinBox;
    m_rot_z_sb->setRange(-10, 10);
    m_rot_z_sb->setValue(0.0);
    m_rot_z_sb->setSingleStep(0.05);
    m_rot_z_sb->setDecimals(3);

    form_layout->addRow("Origin.x [m]", m_origin_x_sb);
    form_layout->addRow("Origin.y [m]", m_origin_y_sb);
    form_layout->addRow("Origin.z [m]", m_origin_z_sb);
    form_layout->addRow("Rot.x [rad]",  m_rot_x_sb);
    form_layout->addRow("Rot.y [rad]",  m_rot_y_sb);
    form_layout->addRow("Rot.z [rad]",  m_rot_z_sb);

    groupbox->setLayout(form_layout);
    mainlayout->addWidget(groupbox);
    setLayout(mainlayout);
}

QVector3D FixedProbeWidget::get_origin(float /*time*/) const {
    return QVector3D(m_origin_x_sb->value(),
                     m_origin_y_sb->value(),
                     m_origin_z_sb->value());
}

QVector3D FixedProbeWidget::get_rot_angles(float /*time*/) const {
    return QVector3D(m_rot_x_sb->value(),
                     m_rot_y_sb->value(),
                     m_rot_z_sb->value());
}

DynamicProbeWidget::DynamicProbeWidget(QWidget* parent, Qt::WindowFlags f)
    : QWidget(parent, f)
{
    auto mainlayout = new QVBoxLayout;
    auto groupbox = new QGroupBox("Linear Probe Position/Orientation");
    auto form_layout = new QFormLayout;
    
    m_origin_x1_sb = new QDoubleSpinBox;
    m_origin_x1_sb->setRange(-1.0, 1.0);
    m_origin_x1_sb->setValue(0.0);
    m_origin_x1_sb->setSingleStep(0.001);
    m_origin_x1_sb->setDecimals(3);

    m_origin_y1_sb = new QDoubleSpinBox;
    m_origin_y1_sb->setRange(-1.0, 1.0);
    m_origin_y1_sb->setValue(0.0);
    m_origin_y1_sb->setSingleStep(0.001);
    m_origin_y1_sb->setDecimals(3);

    m_origin_z1_sb = new QDoubleSpinBox;
    m_origin_z1_sb->setRange(-1.0, 1.0);
    m_origin_z1_sb->setValue(0.0);
    m_origin_z1_sb->setSingleStep(0.001);
    m_origin_z1_sb->setDecimals(3);

    m_origin_x2_sb = new QDoubleSpinBox;
    m_origin_x2_sb->setRange(-1.0, 1.0);
    m_origin_x2_sb->setValue(0.0);
    m_origin_x2_sb->setSingleStep(0.001);
    m_origin_x2_sb->setDecimals(3);

    m_origin_y2_sb = new QDoubleSpinBox;
    m_origin_y2_sb->setRange(-1.0, 1.0);
    m_origin_y2_sb->setValue(0.0);
    m_origin_y2_sb->setSingleStep(0.001);
    m_origin_y2_sb->setDecimals(3);

    m_origin_z2_sb = new QDoubleSpinBox;
    m_origin_z2_sb->setRange(-1.0, 1.0);
    m_origin_z2_sb->setValue(0.0);
    m_origin_z2_sb->setSingleStep(0.001);
    m_origin_z2_sb->setDecimals(3);

    m_starttime_sb = new QDoubleSpinBox;
    m_starttime_sb->setRange(-100, 100);
    m_starttime_sb->setValue(0.0f);
    m_starttime_sb->setSingleStep(0.1);
    m_starttime_sb->setDecimals(3);

    m_endtime_sb = new QDoubleSpinBox;
    m_endtime_sb->setRange(-100, 100);
    m_endtime_sb->setValue(1.0f);
    m_endtime_sb->setSingleStep(0.1);
    m_endtime_sb->setDecimals(3);

    m_rotation_x_sb = new QDoubleSpinBox;
    m_rotation_x_sb->setRange(-10, 10);
    m_rotation_x_sb->setValue(0.0);
    m_rotation_x_sb->setSingleStep(0.1);
    m_rotation_x_sb->setDecimals(3);

    m_rotation_y_sb = new QDoubleSpinBox;
    m_rotation_y_sb->setRange(-10, 10);
    m_rotation_y_sb->setValue(0.0);
    m_rotation_y_sb->setSingleStep(0.1);
    m_rotation_y_sb->setDecimals(3);

    m_rotation_z_sb = new QDoubleSpinBox;
    m_rotation_z_sb->setRange(-10, 10);
    m_rotation_z_sb->setValue(0.0);
    m_rotation_z_sb->setSingleStep(0.1);
    m_rotation_z_sb->setDecimals(3);


    form_layout->addRow("Origin1.x [m]", m_origin_x1_sb);
    form_layout->addRow("Origin1.y [m]", m_origin_y1_sb);
    form_layout->addRow("Origin1.z [m]", m_origin_z1_sb);
    form_layout->addRow("Time 1 [s]",    m_starttime_sb);
    form_layout->addRow("Origin2.x [m]", m_origin_x2_sb);
    form_layout->addRow("Origin2.y [m]", m_origin_y2_sb);
    form_layout->addRow("Origin2.z [m]", m_origin_z2_sb);
    form_layout->addRow("Time 2 [s]",    m_endtime_sb);
    form_layout->addRow("Rot.x [rad]",   m_rotation_x_sb);
    form_layout->addRow("Rot.y [rad]",   m_rotation_y_sb);
    form_layout->addRow("Rot.z [rad]",   m_rotation_z_sb);

    groupbox->setLayout(form_layout);
    mainlayout->addWidget(groupbox);
    setLayout(mainlayout);
}

QVector3D DynamicProbeWidget::get_origin(float time) const {
    QVector3D p1(m_origin_x1_sb->value(),
                 m_origin_y1_sb->value(),
                 m_origin_z1_sb->value());
    
    QVector3D p2(m_origin_x2_sb->value(),
                 m_origin_y2_sb->value(),
                 m_origin_z2_sb->value());
    
    const auto start_time = m_starttime_sb->value();
    const auto end_time   = m_endtime_sb->value();

    if (time <= start_time) {
        return p1;
    } else if (time >= end_time) {
        return p2;
    } else {
        // linear interpolation
        const auto k = (time-start_time)/(end_time-start_time);
        qDebug() << "DynamicProbeWidget: interpolation constant is " << k;
        return k*p2 + (1-k)*p1;
    }
};

QVector3D DynamicProbeWidget::get_rot_angles(float /*time*/) const {
    return QVector3D(m_rotation_x_sb->value(),
                     m_rotation_y_sb->value(),
                     m_rotation_z_sb->value());
}

ProbeWidget::ProbeWidget(QWidget* parent, Qt::WindowFlags f)
    : QWidget(parent, f)
{
    auto layout = new QVBoxLayout;

    // stacked editor widgets
    auto fixed_probe_widget = new FixedProbeWidget;
    auto dynamic_probe_widget = new DynamicProbeWidget;
    m_stacked_widget = new QStackedWidget;
    m_stacked_widget->addWidget(fixed_probe_widget);
    m_stacked_widget->addWidget(dynamic_probe_widget);

    auto combo_box = new QComboBox;
    combo_box->addItem(tr("Fixed Probe"));
    combo_box->addItem(tr("Dynamic Probe"));
    connect(combo_box, SIGNAL(activated(int)), m_stacked_widget, SLOT(setCurrentIndex(int)));

    layout->addWidget(m_stacked_widget);
    layout->addWidget(combo_box);
    setLayout(layout);
}

QVector3D ProbeWidget::get_origin(float time) const {
    auto active_widget = m_stacked_widget->currentWidget();
    auto temp = dynamic_cast<detail::IProbeEditor*>(active_widget);
    if (!temp) {
        throw std::logic_error("cast to IProbeEditor failed");
    }
    return temp->get_origin(time);
}

QVector3D ProbeWidget::get_rot_angles(float time) const {
    auto active_widget = m_stacked_widget->currentWidget();
    auto temp = dynamic_cast<detail::IProbeEditor*>(active_widget);
    if (!temp) {
        throw std::logic_error("cast to IProbeEditor failed");
    }
    return temp->get_rot_angles(time);
}
