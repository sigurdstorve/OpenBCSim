#include "LinearScanseqWidget.hpp"
#include <cmath>
#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QGroupBox>
#include "../../core/to_string.hpp"
#include "../../utils/rotation3d.hpp"

LinearScanseqWidget::LinearScanseqWidget(QWidget* parent)
    : QWidget(parent)
{
    auto mainlayout = new QVBoxLayout;
    auto groupbox = new QGroupBox("Linear Scansequence Editor");
    auto form_layout = new QFormLayout;
    
    m_width_sb = new QDoubleSpinBox;
    m_width_sb->setRange(0.001, 0.2);
    m_width_sb->setValue(0.04);
    m_width_sb->setSingleStep(0.0003);
    m_width_sb->setDecimals(3);
    
    m_range_max_sb = new QDoubleSpinBox;
    m_range_max_sb->setRange(0.0, 0.2);
    m_range_max_sb->setValue(0.12);
    m_range_max_sb->setSingleStep(1e-3);
    m_range_max_sb->setDecimals(3);
        
    m_num_lines_sb = new QSpinBox;
    m_num_lines_sb->setRange(3, 8192);
    m_num_lines_sb->setValue(64);
    
    form_layout->addRow("Width [m]",        m_width_sb);
    form_layout->addRow("Max. range [m]",   m_range_max_sb);
    form_layout->addRow("#Lines",           m_num_lines_sb);

    groupbox->setLayout(form_layout);
    mainlayout->addWidget(groupbox);
    setLayout(mainlayout);
}


bcsim::ScanGeometry::ptr LinearScanseqWidget::get_geometry(int& num_lines) const {
    auto geo = new bcsim::LinearScanGeometry;
    
    // get values from input widgets
    geo->range_max = static_cast<float>(m_range_max_sb->value());
    geo->width     = static_cast<float>(m_width_sb->value());

    num_lines = m_num_lines_sb->value();
    return bcsim::ScanGeometry::ptr(geo);
}
