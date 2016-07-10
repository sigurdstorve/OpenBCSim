#include <QTextEdit>
#include <QVBoxLayout>
#include <QString>
#include "LogWidget.hpp"

LogWidget::LogWidget(QWidget* parent, Qt::WindowFlags f)
    : QWidget(parent, f)
{
    auto layout = new QVBoxLayout;
    setWindowTitle("Log Window");
    m_text_edit = new QTextEdit;
    m_text_edit->setReadOnly(true);
    layout->addWidget(m_text_edit);
    setLayout(layout);
}

void LogWidget::write(bcsim::ILog::LogType type, const std::string& msg) {
    switch (type) {
    case bcsim::ILog::DEBUG:
        m_text_edit->setTextColor(QColor("grey"));
        m_text_edit->append("[debug] " + QString::fromStdString(msg));
        break;
    case bcsim::ILog::FATAL:
        m_text_edit->setTextColor(QColor("red"));
        m_text_edit->append("[fatal] " + QString::fromStdString(msg));
        break;
    case bcsim::ILog::INFO:
        m_text_edit->setTextColor(QColor("green"));
        m_text_edit->append("[info] " + QString::fromStdString(msg));
        break;
    case bcsim::ILog::WARNING:
        m_text_edit->setTextColor(QColor("black"));
        m_text_edit->append("[warning] " + QString::fromStdString(msg));
        break;
    }
}

void LogWidget::clear_contents() {
    m_text_edit->clear();
}

