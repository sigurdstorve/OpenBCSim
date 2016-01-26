#include "SectorScanseqWidget.hpp"
#include <cmath>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QGroupBox>
#include <QDebug>
#include "../../core/to_string.hpp"

SectorScanseqWidget::SectorScanseqWidget(QWidget* parent)
    : QWidget(parent)
{
    auto mainlayout = new QVBoxLayout;
    auto groupbox = new QGroupBox("Sector Scansequence Editor");
    auto form_layout = new QFormLayout;
    
    m_depth_sb = new QDoubleSpinBox;
    m_depth_sb->setRange(0.001, 0.5);
    m_depth_sb->setValue(0.12);
    m_depth_sb->setSingleStep(1e-3);
    m_depth_sb->setDecimals(3);

    m_width_sb = new QDoubleSpinBox;
    m_width_sb->setRange(0.1, 3.14);
    m_width_sb->setValue(1.0);
    m_width_sb->setSingleStep(0.1);
    m_width_sb->setDecimals(3);
    
    m_tilt_sb = new QDoubleSpinBox;
    m_tilt_sb->setRange(-1.57, 1.57);
    m_tilt_sb->setValue(0.0);
    m_tilt_sb->setSingleStep(0.05);
    m_tilt_sb->setDecimals(3);
    
    m_num_lines_sb = new QSpinBox;
    m_num_lines_sb->setRange(3, 8192);
    m_num_lines_sb->setValue(64);
    
    form_layout->addRow("Depth [m]",        m_depth_sb);
    form_layout->addRow("Width [rad]",      m_width_sb);
    form_layout->addRow("Tilt [rad]",       m_tilt_sb);
    form_layout->addRow("#Lines",           m_num_lines_sb);

    groupbox->setLayout(form_layout);
    mainlayout->addWidget(groupbox);
    setLayout(mainlayout);
}

bcsim::ScanGeometry::ptr SectorScanseqWidget::get_geometry(int& num_lines) const {
    auto geo = new bcsim::SectorScanGeometry;
    
    // get values from input widgets
    geo->depth            = static_cast<float>(m_depth_sb->value());
    geo->tilt             = static_cast<float>(m_tilt_sb->value());
    geo->width            = static_cast<float>(m_width_sb->value());

    num_lines = m_num_lines_sb->value();
    return bcsim::ScanGeometry::ptr(geo);
}
