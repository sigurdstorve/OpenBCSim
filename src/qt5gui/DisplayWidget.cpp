#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsPixmapItem>
#include <QWheelEvent>
#include <QHBoxLayout>
#include <QDebug>
#include "DisplayWidget.hpp"

class CustomView : public QGraphicsView {
public:
    CustomView(QWidget* parent = 0)
        : QGraphicsView(parent) { }
    CustomView(QGraphicsScene* scene, QWidget* parent = 0)
        : QGraphicsView(scene, parent) { }

protected:
    virtual void wheelEvent(QWheelEvent* event) override {
        if (event->modifiers() == Qt::ControlModifier) {
            setTransformationAnchor(QGraphicsView::AnchorUnderMouse);
            if (event->delta() > 0) {
                scale(1.1, 1.1);
            } else {
                scale(1.0/1.1, 1.0/1.1);
            }
        } else {
            QGraphicsView::wheelEvent(event);
        }
    }
};

DisplayWidget::DisplayWidget(QWidget* parent, Qt::WindowFlags f)
    : QWidget(parent, f)
{
    // If no inital size is given, the scene bouding rect will be
    // large enough to cover all items that has been added to it
    // since creation.
    m_scene = new QGraphicsScene(this);
    m_view = new CustomView(m_scene);
    m_view->setDragMode(QGraphicsView::ScrollHandDrag);
    m_view->setInteractive(false);
    
    // pixmap for B-mode
    m_pixmap_item = new QGraphicsPixmapItem;
    m_pixmap_item->setTransformationMode(Qt::SmoothTransformation); // enable bilinear filtering
    m_scene->addItem(m_pixmap_item);

    // pixmap for color flow
    m_colorflow_item = new QGraphicsPixmapItem;
    m_colorflow_item->setTransformationMode(Qt::SmoothTransformation);
    m_scene->addItem(m_colorflow_item);

    m_layout = new QHBoxLayout;
    m_layout->addWidget(m_view);
    setLayout(m_layout);
}

void DisplayWidget::update_bmode(const QPixmap& pixmap, float x_min, float x_max, float y_min, float y_max) {
    bool should_autofit = (m_pixmap_item->boundingRect().width()==0) && (m_pixmap_item->boundingRect().height()==0);
    // set pixel data and scale item
    m_pixmap_item->setPixmap(pixmap);
    const auto width_meters = x_max - x_min;
    const auto height_meters = y_max - y_min;
    const auto scale_x = width_meters/pixmap.width();
    const auto scale_y = height_meters/pixmap.height();
    const auto transform_scale = QTransform::fromScale(scale_x, scale_y);
    m_pixmap_item->setTransform(transform_scale);

    // move it
    m_pixmap_item->setPos(x_min, y_min);

    // only do fitInView() first time since it messes up manual zoominal and positioning.
    if (should_autofit) {
        m_view->fitInView(m_pixmap_item, Qt::KeepAspectRatio);
    }
}

void DisplayWidget::update_colorflow(const QPixmap& pixmap, float x_min, float x_max, float y_min, float y_max) {
    m_colorflow_item->setPixmap(pixmap);
    m_colorflow_item->setTransform(QTransform::fromScale(0.05/64, 0.05/64));
    m_colorflow_item->setPos(x_min, y_min);
}