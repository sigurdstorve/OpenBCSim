#pragma once

namespace ellipsoid {

struct EllipseCoefficients {
    float a, b, c;
};

struct Point3D {
    Point3D(float x, float y, float z) : x(x), y(y), z(z) { }
    Point3D() : Point3D(0.0f, 0.0f, 0.0f) { }
    float x, y, z;
};

struct Region3D {
    Region3D(float x_min, float x_max, float y_min, float y_max, float z_min, float z_max)
        : x_min(x_min), x_max(x_max), y_min(y_min), y_max(y_max), z_min(z_min), z_max(z_max) { }
    Region3D() : Region3D(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f) { }
    float x_min, x_max;
    float y_min, y_max;
    float z_min, z_max;
};

class Ellipsoid {
public:
    Ellipsoid(const Region3D& region);

    bool is_point_inside(const Point3D& p) const;

private:
    void compute_centers(const Region3D& region);

    void compute_coefficients(const Region3D& region);

private:
    Point3D             m_center;
    EllipseCoefficients m_coeffs;
};

class CappedZEllipsoid {
public:
    CappedZEllipsoid(const Region3D& region, float z_ratio);

    bool is_point_inside(const Point3D& p) const;

private:
    Ellipsoid   m_ellipsoid;
    float       m_z_cap;
};

// Three-dimensional capped ellipsoid model. Dimensions are for inner capped ellipsoid.
// Long-axis is parallel to the z-axis.
class ThickCappedZEllipsoid {
public:
    ThickCappedZEllipsoid(const Region3D& region, float thickness, float z_ratio);

    bool is_point_inside(const Point3D& p) const;

private:
    CappedZEllipsoid    m_inner;
    CappedZEllipsoid    m_outer;
};

}   // end namespace
