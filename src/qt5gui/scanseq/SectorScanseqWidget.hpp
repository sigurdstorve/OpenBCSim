#pragma once
#include <QWidget>
#include "ScanSequence.hpp"
#include "ScanGeometry.hpp"
#include "ScanseqWidget.hpp"

// Forward decl.
class QDoubleSpinBox;
class QSpinBox;

// Widget for manually creating scan sequences
class SectorScanseqWidget : public QWidget,
                            public detail::IScanseqEditor {
Q_OBJECT
public:
    explicit SectorScanseqWidget(QWidget* parent=0);

    virtual bcsim::ScanGeometry::ptr get_geometry(int& num_lines) const;

private:
    // fields which mirror struct ScanSectorGeometry
    QDoubleSpinBox*     m_depth_sb;
    QDoubleSpinBox*     m_width_sb;
    QDoubleSpinBox*     m_tilt_sb;
    // number of scanlines
    QSpinBox*           m_num_lines_sb;
};