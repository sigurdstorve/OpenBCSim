#pragma once
#include <memory>
#include <ostream>

namespace trianglemesh3d {

enum class Mesh3dFileType {
	WAVEFRONT_OBJ
};

// A 3D triangle mesh with normals.
class ITriangleMesh3d {
public:
	typedef std::unique_ptr<ITriangleMesh3d> u_ptr;
	virtual ~ITriangleMesh3d() { }
    
    // the number of vertices in the triangle mesh.
    virtual size_t num_vertices() const = 0;

    // pointer to the vertex data stored as [x0,y0,z0,x1,y1,z1,...]
    virtual const double* vertex_data() const = 0;
	
    // pointer to the normal data stored as [x0,y0,z0,x1,y1,z1,...]
    virtual const double* normal_data() const = 0;
};

std::ostream& operator<<(std::ostream& os, const ITriangleMesh3d& mesh);

// Load model from a file.
ITriangleMesh3d::u_ptr LoadTriangleMesh3d(const std::string& filename, Mesh3dFileType type);

}   // end namespace
