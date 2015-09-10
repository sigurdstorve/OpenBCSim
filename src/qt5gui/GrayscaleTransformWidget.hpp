#include <QWidget>

class QDoubleSpinBox;
class QCheckBox;

struct GrayscaleTransformSettings {
    float dyn_range;
    float gain;
    float normalization_const;
    bool auto_normalize;
};

class GrayscaleTransformWidget : public QWidget {
Q_OBJECT
public:
    explicit GrayscaleTransformWidget(QWidget* parent=0, Qt::WindowFlags f=0);

    GrayscaleTransformSettings get_values() const;

    void set_normalization_constant(float value);

private:
    QDoubleSpinBox*         m_dyn_range_sb;
    QDoubleSpinBox*         m_gain_sb;
    QDoubleSpinBox*         m_normalization_const_sb;
    QCheckBox*              m_auto_normalize_cb;
};