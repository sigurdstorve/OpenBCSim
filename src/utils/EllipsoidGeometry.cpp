#include <cmath>
#include <stdexcept>
#include "EllipsoidGeometry.hpp"

namespace ellipsoid {
Ellipsoid::Ellipsoid(const Region3D& region) {
    compute_coefficients(region);
    compute_centers(region);
}

bool Ellipsoid::is_point_inside(const Point3D& p) const {
    const auto x_sq = std::pow((p.x - m_center.x) / m_coeffs.a, 2);
    const auto y_sq = std::pow((p.y - m_center.y) / m_coeffs.b, 2);
    const auto z_sq = std::pow((p.z - m_center.z) / m_coeffs.c, 2);
    return (x_sq + y_sq + z_sq) <= 1.0f;
}

void Ellipsoid::compute_centers(const Region3D& region) {
    m_center.x = region.x_min + m_coeffs.a;
    m_center.y = region.y_min + m_coeffs.b;
    m_center.z = region.z_min + m_coeffs.c;
}

void Ellipsoid::compute_coefficients(const Region3D& region) {
    m_coeffs.a = 0.5f*(region.x_max - region.x_min);
    m_coeffs.b = 0.5f*(region.y_max - region.y_min);
    m_coeffs.c = 0.5f*(region.z_max - region.z_min);
}

CappedZEllipsoid::CappedZEllipsoid(const Region3D& region, float z_ratio)
    : m_ellipsoid(Ellipsoid(region)),
    m_z_cap(region.z_min + (region.z_max - region.z_min)*z_ratio)
{
    if ((z_ratio < 0.0f) || (z_ratio > 1.0f)) {
        throw std::runtime_error("z-ratio must be in [0.0, 1.0]");
    }
}

bool CappedZEllipsoid::is_point_inside(const Point3D& p) const {
    const auto is_inside_ellipsoid = m_ellipsoid.is_point_inside(p);
    const auto is_above_cap = (p.z <= m_z_cap);
    return is_inside_ellipsoid && is_above_cap;
}

ThickCappedZEllipsoid::ThickCappedZEllipsoid(const Region3D& region, float thickness, float z_ratio)
    : m_inner(CappedZEllipsoid(region, z_ratio)),
    m_outer(CappedZEllipsoid(Region3D(region.x_min - thickness, region.x_max + thickness,
        region.y_min - thickness, region.y_max + thickness,
        region.z_min - thickness, region.z_max + thickness), z_ratio))
{
}

bool ThickCappedZEllipsoid::is_point_inside(const Point3D& p) const {
    const auto is_inside_outer = m_outer.is_point_inside(p);
    const auto is_outside_inner = !m_inner.is_point_inside(p);
    return is_inside_outer && is_outside_inner;
}

}   // end namespace
