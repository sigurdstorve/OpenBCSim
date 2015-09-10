#pragma once
#include <QWidget>
#include "ScanSequence.hpp"

class QStackedWidget;

namespace detail {

// Common denominator for editors of scansequences.
class IScanseqEditor {
public:
    virtual bcsim::ScanGeometry::ptr get_geometry(int& num_lines) const = 0;
};

}

class ScanseqWidget : public QWidget {
Q_OBJECT
public:
    ScanseqWidget(QWidget* parent=0, Qt::WindowFlags f=0);

    // Returns scansequence and number of lines.
    bcsim::ScanGeometry::ptr get_geometry(int& num_lines) const;

protected:
    QStackedWidget*     m_stacked_widget;
};