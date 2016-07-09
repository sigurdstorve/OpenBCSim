#include <string>
#include <stdexcept>
#include <memory>
#include <fstream>
#include "TriangleMesh3d.hpp"
#include "WavefrontObjLoader.hpp"
namespace trianglemesh3d {

ITriangleMesh3d::u_ptr LoadTriangleMesh3d(const std::string& filename, Mesh3dFileType type) {
	auto stream = std::ifstream(filename, std::ios::in);
	return LoadTriangleMesh3d(stream, type);
}

ITriangleMesh3d::u_ptr LoadTriangleMesh3d(std::istream& in_stream, Mesh3dFileType type) {
	switch (type) {
	case Mesh3dFileType::WAVEFRONT_OBJ:
		return std::make_unique<WavefrontObjLoader>(in_stream);
	default:
		throw std::logic_error(std::string(__FILE__) + " no handler for file type");
	}
}

std::ostream& operator<<(std::ostream& os, const ITriangleMesh3d& mesh) {
    const auto num_verts = mesh.num_vertices();
    os << "[mesh with " << num_verts << " vertices]" << std::endl;
    const auto vp = mesh.vertex_data();
    const auto np = mesh.normal_data();
    for (size_t i = 0; i < num_verts; i++) {
        os << i << " : ";
        os << "vertex=(";
        os << vp[3*i] << ", " << vp[3*i+1] << ", " << vp[3*i+2];
        os << ")";
        os << "normal=(";
        os << np[3*i] << ", " << np[3*i+1] << ", " << np[3*i+2];
        os << ")" << std::endl;
    }
    return os;
}

}   // end namespace
