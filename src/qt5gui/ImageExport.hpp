#pragma once
#include <QString>

class QPixmap;

// Class for managing saving of numbered sequence of images, such
// as Image00001.bmp, Image00002.bmp, ...
class ImageSaver {
public:
    explicit ImageSaver(const QString& output_path);

    void set_format_str(const QString& format_str);

    void add(const QPixmap& pixmap);

    void add(const QImage& image);

    void reset_counter();

private:
    const QString construct_cur_basename() const;

    const QString construct_cur_image_path() const;

private:
    size_t  m_counter;
    QString m_output_path;
    QString m_format_str;
    int     m_quality_factor;
};