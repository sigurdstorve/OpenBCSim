#include <stdexcept>
#include <QVBoxLayout>
#include <QComboBox>
#include <QStackedWidget>
#include "ScanseqWidget.hpp"
#include "SectorScanseqWidget.hpp"
#include "LinearScanseqWidget.hpp"

ScanseqWidget::ScanseqWidget(QWidget* parent, Qt::WindowFlags f)
    : QWidget(parent, f)
{
    auto layout = new QVBoxLayout;

    // stacked editor widgets
    auto linear_widget = new LinearScanseqWidget;
    auto sector_widget = new SectorScanseqWidget;
    m_stacked_widget = new QStackedWidget;
    m_stacked_widget->addWidget(sector_widget);
    m_stacked_widget->addWidget(linear_widget);

    auto combo_box = new QComboBox;
    combo_box->addItem(tr("Sector Scan"));
    combo_box->addItem(tr("Linear Scan"));
    connect(combo_box, SIGNAL(activated(int)), m_stacked_widget, SLOT(setCurrentIndex(int)));

    layout->addWidget(m_stacked_widget);
    layout->addWidget(combo_box);

    setLayout(layout);
}

bcsim::ScanGeometry::ptr ScanseqWidget::get_geometry(int& num_lines) const {
    auto active_widget = m_stacked_widget->currentWidget();
    auto temp = dynamic_cast<detail::IScanseqEditor*>(active_widget);
    if (!temp) {
        throw std::runtime_error("cast to IScanseqEditor failed");
    }
    return temp->get_geometry(num_lines);
}

