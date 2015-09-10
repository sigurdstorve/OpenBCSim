#pragma once
#include <QVector3D>
#include <QGenericMatrix>
#include <QWidget>

class QStackedWidget;
class QDoubleSpinBox;

namespace detail {

// Common denominator for obtaining position and orientation.
// It is perhaps better to combine into a single 4x4 matrix?
class IProbeEditor {
public:
    virtual QVector3D get_origin(float time)     const = 0;
    virtual QVector3D get_rot_angles(float time) const = 0;
};

}   // end namespace

// Widget to edit a probe with time-constant position and orientation.
class FixedProbeWidget : public QWidget,
                         public detail::IProbeEditor {
Q_OBJECT
public:
    explicit FixedProbeWidget(QWidget* parent=0, Qt::WindowFlags f=0);

    virtual QVector3D get_origin(float time) const;

    // [rot_x, rot_y, rot_z]
    virtual QVector3D get_rot_angles(float time) const;

private:
    QDoubleSpinBox*     m_origin_x_sb;
    QDoubleSpinBox*     m_origin_y_sb;
    QDoubleSpinBox*     m_origin_z_sb;
    QDoubleSpinBox*     m_rot_x_sb;
    QDoubleSpinBox*     m_rot_y_sb;
    QDoubleSpinBox*     m_rot_z_sb;
};

// Widget to edit a probe with time-varying position and orientation.
// NOTE: Movement is along a straight line with constant velocity.
class DynamicProbeWidget : public QWidget,
                           public detail::IProbeEditor {
Q_OBJECT
public:
    explicit DynamicProbeWidget(QWidget* parent=0, Qt::WindowFlags f=0);

    virtual QVector3D get_origin(float time) const;

    virtual QVector3D get_rot_angles(float time) const;

private:
    QDoubleSpinBox*     m_origin_x1_sb;
    QDoubleSpinBox*     m_origin_y1_sb;
    QDoubleSpinBox*     m_origin_z1_sb;
    QDoubleSpinBox*     m_origin_x2_sb;
    QDoubleSpinBox*     m_origin_y2_sb;
    QDoubleSpinBox*     m_origin_z2_sb;
    QDoubleSpinBox*     m_starttime_sb;
    QDoubleSpinBox*     m_endtime_sb;
};

// Widget to edit probe position and orientation.
class ProbeWidget : public QWidget {
Q_OBJECT
public:
    explicit ProbeWidget(QWidget* parent=0, Qt::WindowFlags f=0);

    virtual QVector3D get_origin(float time) const;

    virtual QVector3D get_rot_angles(float time) const;

private:
    QStackedWidget*     m_stacked_widget;
};
