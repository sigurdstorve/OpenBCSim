#include <QFormLayout>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include "GrayscaleTransformWidget.hpp"

GrayscaleTransformWidget::GrayscaleTransformWidget(QWidget* parent, Qt::WindowFlags f)
    : QWidget(parent, f)
{
    auto form_layout = new QFormLayout;

    m_dyn_range_sb = new QDoubleSpinBox;
    m_dyn_range_sb->setMinimum(10.0f);
    m_dyn_range_sb->setMaximum(120.0f);
    m_dyn_range_sb->setValue(50.0f);
    m_dyn_range_sb->setSingleStep(1.0f);
    form_layout->addRow("DynamicRange [db]", m_dyn_range_sb);

    m_gain_sb = new QDoubleSpinBox;
    m_gain_sb->setMinimum(0.0f);
    m_gain_sb->setMaximum(10.f);
    m_gain_sb->setValue(1.0f);
    m_gain_sb->setSingleStep(0.01f);
    form_layout->addRow("Gain", m_gain_sb);

    m_normalization_const_sb = new QDoubleSpinBox;
    m_normalization_const_sb->setMinimum(0.0f);
    m_normalization_const_sb->setMaximum(10e9f);
    m_normalization_const_sb->setValue(1.0f);
    form_layout->addRow("Normalization const.", m_normalization_const_sb);

    m_auto_normalize_cb = new QCheckBox;
    m_auto_normalize_cb->setChecked(true);
    form_layout->addRow("Auto-normalize", m_auto_normalize_cb);

    setLayout(form_layout);
}


GrayscaleTransformSettings GrayscaleTransformWidget::get_values() const {
    GrayscaleTransformSettings res;
    res.auto_normalize      = m_auto_normalize_cb->isChecked();
    res.dyn_range           = m_dyn_range_sb->value();
    res.gain                = m_gain_sb->value();
    res.normalization_const = m_normalization_const_sb->value();
    return res;
}

void GrayscaleTransformWidget::set_normalization_constant(float value) {
    m_normalization_const_sb->setValue(value);
}
