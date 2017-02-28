#pragma once
#include <memory>
#include <ostream>
#include "../../core/export_macros.hpp" // TODO: Better solution?

namespace trianglemesh3d {

enum class Mesh3dFileType {
	WAVEFRONT_OBJ
};

// A 3D triangle mesh with normals.
class DLL_PUBLIC ITriangleMesh3d {
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

DLL_PUBLIC std::ostream& operator<<(std::ostream& os, const ITriangleMesh3d& mesh);

// Load model from a file.
DLL_PUBLIC ITriangleMesh3d::u_ptr LoadTriangleMesh3d(const std::string& filename, Mesh3dFileType type);

// Load model from an input stream. This is useful in case the model file is
// not a file on disk, but e.g. a Qt resource file embedded in an executable.
DLL_PUBLIC ITriangleMesh3d::u_ptr LoadTriangleMesh3d(std::istream& in_stream, Mesh3dFileType type);

}   // end namespace
