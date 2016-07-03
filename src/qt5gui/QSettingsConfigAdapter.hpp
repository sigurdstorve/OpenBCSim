#pragma once
#include <string>
#include <memory>
#include <QVariant>
#include "IConfig.hpp"

// forward decl.
class QSettings;

// Adapter from QSettings to IConfig.
class QSettingsConfigAdapter : public IConfig {
public:
    QSettingsConfigAdapter(std::shared_ptr<QSettings>& qsettings);
    virtual ~QSettingsConfigAdapter() { }

    virtual bool get_bool(const std::string& par_name) const override;
    virtual bool get_bool(const std::string& par_name, bool default_value) const override;
    virtual std::string get_string(const std::string& par_name) const;
    virtual std::string get_string(const std::string& par_name, const std::string& default_value) const override;
    virtual int get_int(const std::string& par_name) const override;
    virtual int get_int(const std::string& par_name, int default_value) const override;
    virtual double get_double(const std::string& par_name) const override;
    virtual double get_double(const std::string& par_name, double default_value) const override;

private:
    QVariant get_and_throw_if_not_found(const std::string& par_name) const;

private:
    std::shared_ptr<QSettings> m_qsettings;
};
