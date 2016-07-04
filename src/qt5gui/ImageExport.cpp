#include <QPixmap>
#include <QImage>
#include <QDir>
#include "ImageExport.hpp"

ImageSaver::ImageSaver(const QString& output_path)
    : m_output_path(output_path),
      m_quality_factor(100)
{
    set_format_str("Image%1.bmp");
    reset_counter();
}

void ImageSaver::set_format_str(const QString& format_str) {
    m_format_str = format_str;
}

void ImageSaver::reset_counter() {
    m_counter = 0;
}

const QString ImageSaver::add(const QPixmap& pixmap) {
    const auto img_file = construct_cur_image_path();
    pixmap.save(img_file, 0, m_quality_factor);
    m_counter++;
    return img_file;
}

const QString ImageSaver::add(const QImage& image) {
    const auto img_file = construct_cur_image_path();
    image.save(img_file, 0, m_quality_factor);
    m_counter++;
    return img_file;
}

const QString ImageSaver::construct_cur_image_path() const {
    const auto basename = construct_cur_basename();
    return QDir(m_output_path).filePath(basename);
}

const QString ImageSaver::construct_cur_basename() const {
    return QString(m_format_str).arg(m_counter, 6, 10, QChar('0'));
}

const QString ImageSaver::get_output_path() const {
    return m_output_path;
}
