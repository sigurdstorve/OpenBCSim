#pragma once
#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include "TriangleMesh3d.hpp"

namespace trianglemesh3d {

// Used for representing 3d vertex or normal.
struct Vector3d {
    friend std::istream& operator>>(std::istream& is, Vector3d& v);
    double x, y, z;
};
std::istream& operator>>(std::istream& is, Vector3d& v);

// Used for representing a triangle face
struct TriFace {
    friend std::istream& operator>>(std::istream& is, TriFace& f);
    size_t vertex_inds[3];
    size_t normal_inds[3];
};
std::istream& operator>>(std::istream& is, TriFace& t);

class WavefrontObjLoader : public ITriangleMesh3d {
public:
	// Load 3D mesh from an input stream.
	WavefrontObjLoader(std::istream& obj_stream);

    virtual size_t num_vertices() const override;

    virtual const double* vertex_data() const override;

    virtual const double* normal_data() const override;
        
private:
	void parse_lines(std::istream& obj_stream);
	void process_text_line(const std::string& line);
	void dispatch_command(const std::string& command, std::stringstream& rest);
	void process_vertex_data(std::istream& rest);   // throws
	void process_vertex_normal_data(std::istream& rest);    // throws
	void process_face_data(std::istream& rest);
    void expand_vertices_and_normals();

private:
	const std::string	    m_obj_file;
    std::vector<Vector3d>   m_parsed_vertices;
    std::vector<Vector3d>   m_parsed_normals;
    std::vector<TriFace>    m_parsed_tri_faces;
    
    std::vector<double> m_vertex_data;
	std::vector<double> m_normal_data;
};

} // end namespace
