#pragma once
#include <QWidget>
#include "../../core/ScanSequence.hpp"
#include "ScanseqWidget.hpp"

// Forward decl.
class QDoubleSpinBox;
class QSpinBox;

// Widget for manually creating scan sequences
class LinearScanseqWidget : public QWidget,
                            public detail::IScanseqEditor {
Q_OBJECT
public:
    explicit LinearScanseqWidget(QWidget* parent=0);

    virtual bcsim::ScanGeometry::ptr get_geometry(int& num_lines) const;

private:
    // fields which mirror struct LinearScanGeometry.
    QDoubleSpinBox*     m_width_sb;
    QDoubleSpinBox*     m_range_max_sb;
    // number of scanlines
    QSpinBox*           m_num_lines_sb;     // Number of scanlines
};