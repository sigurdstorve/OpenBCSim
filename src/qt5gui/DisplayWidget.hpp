#pragma once
#include <QWidget>

class QImage;
class QGraphicsView;
class QGraphicsScene;
class QGraphicsPixmapItem;
class QVBoxLayout;
class QStatusBar;

class DisplayWidget : public QWidget {
public:
    DisplayWidget(QWidget* parent = 0, Qt::WindowFlags f=0);

    void fitInView();

    void update_bmode(const QPixmap& pixmap, float x_min, float x_max, float y_min, float y_max);

    void update_colorflow(const QPixmap& pixmap, float x_min, float x_max, float y_min, float y_max);

    void update_status(const QString& msg, int timeout=0);

private:
    QGraphicsView*                  m_view;
    QGraphicsScene*                 m_scene;
    QGraphicsPixmapItem*            m_pixmap_item;
    QGraphicsPixmapItem*            m_colorflow_item;
    QVBoxLayout*                    m_layout;
    QStatusBar*                     m_status_bar;
};