#include <stdexcept>
#include <QSettings>
#include "QSettingsConfigAdapter.hpp"

QSettingsConfigAdapter::QSettingsConfigAdapter(std::shared_ptr<QSettings>& qsettings)
    : m_qsettings(qsettings) { }

QVariant QSettingsConfigAdapter::get_and_throw_if_not_found(const std::string& par_name) const {
    const auto value = m_qsettings->value(QString::fromStdString(par_name));
    if (value == QVariant()) {
        throw std::runtime_error("parameter not found");
    }
    return value;
}

bool QSettingsConfigAdapter::get_bool(const std::string& par_name) const {
    return get_and_throw_if_not_found(par_name).toBool();
}

std::string QSettingsConfigAdapter::get_string(const std::string& par_name) const {
    return get_and_throw_if_not_found(par_name).toString().toStdString();
}

int QSettingsConfigAdapter::get_int(const std::string& par_name) const {
    return get_and_throw_if_not_found(par_name).toInt();
}

double QSettingsConfigAdapter::get_double(const std::string& par_name) const {
    return get_and_throw_if_not_found(par_name).toDouble();
}

bool QSettingsConfigAdapter::get_bool(const std::string& par_name, bool default_value) const {
    return m_qsettings->value(QString::fromStdString(par_name), default_value).toBool();
}

std::string QSettingsConfigAdapter::get_string(const std::string& par_name, const std::string& default_value) const {
    return m_qsettings->value(QString::fromStdString(par_name), QString::fromStdString(default_value)).toString().toStdString();
}

int QSettingsConfigAdapter::get_int(const std::string& par_name, int default_value) const {
    return m_qsettings->value(QString::fromStdString(par_name), default_value).toInt();
}

double QSettingsConfigAdapter::get_double(const std::string& par_name, double default_value) const {
    return m_qsettings->value(QString::fromStdString(par_name), default_value).toDouble();
}
