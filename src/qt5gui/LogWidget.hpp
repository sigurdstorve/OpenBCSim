#pragma once
#include <QWidget>
#include "../core/BCSimConfig.hpp"

class QTextEdit;

class LogWidget : public QWidget, public bcsim::ILog {
Q_OBJECT
public:
    LogWidget(QWidget* parent=Q_NULLPTR, Qt::WindowFlags f= Qt::WindowFlags());
        
    virtual void write(bcsim::ILog::LogType type, const std::string& msg) override;

    void clear_contents();
        
private:
    QTextEdit*  m_text_edit;
};

