#pragma once
#include <QString>
#include <memory>

class QPixmap;

// Class for managing saving of numbered sequence of images, such
// as Image00001.bmp, Image00002.bmp, ...
class ImageSaver {
public:
    typedef std::unique_ptr<ImageSaver> ptr;

    // create new saver that will put images in output_path.
    explicit ImageSaver(const QString& output_path);

    // set format string to use, eg. "Image%1.bmp"
    void set_format_str(const QString& format_str);

    // returns name of file that was saved
    const QString add(const QPixmap& pixmap);

    // returns name of file that was saved
    const QString add(const QImage& image);

    void reset_counter();

    const QString get_output_path() const;

private:
    const QString construct_cur_basename() const;

    const QString construct_cur_image_path() const;

private:
    size_t  m_counter;
    QString m_output_path;
    QString m_format_str;
    int     m_quality_factor;
};
