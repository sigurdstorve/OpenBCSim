#pragma once
#include <iostream>

// Temporary: this will become a proper widget.
class ConsoleLog : public bcsim::ILog {
    virtual void write(bcsim::ILog::LogType type, const std::string& msg) {
        std::string prefix;
        switch (type) {
        case bcsim::ILog::DEBUG:
            prefix = "[debug] ";
            break;
        case bcsim::ILog::FATAL:
            prefix = "[fatal] ";
            break;
        case bcsim::ILog::INFO:
            prefix = "[info] ";
            break;
        case bcsim::ILog::WARNING:
            prefix = "[warning] ";
            break;
        }
        std::cout << prefix << msg << std::endl;
    }
};
