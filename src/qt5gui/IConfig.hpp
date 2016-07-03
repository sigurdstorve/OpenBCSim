#pragma once
#include <string>
#include <memory>

// Abstract interface for any configuration object that is capable of
// returning named configuration values
class IConfig {
public:
    typedef std::shared_ptr<IConfig> s_ptr;
    virtual ~IConfig() { }

    virtual bool get_bool(const std::string& par_name) const                                            = 0;
    virtual bool get_bool(const std::string& par_name, bool default_value) const                        = 0;
    virtual std::string get_string(const std::string& par_name) const                                   = 0;
    virtual std::string get_string(const std::string& par_name, const std::string& default_value) const = 0;
    virtual int get_int(const std::string& par_name) const                                              = 0;
    virtual int get_int(const std::string& par_name, int default_value) const                           = 0;
    virtual double get_double(const std::string& par_name) const                                        = 0;
    virtual double get_double(const std::string& par_name, double default_value) const                  = 0;
};
