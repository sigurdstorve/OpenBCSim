#include <fstream>
#include <iostream>
#include <boost/numeric/conversion/cast.hpp>
#include "WavefrontObjLoader.hpp"
namespace trianglemesh3d {

WavefrontObjLoader::WavefrontObjLoader(std::istream& obj_stream) {
	parse_lines(obj_stream);
	expand_vertices_and_normals();

	// clear unused members
	m_parsed_normals.clear();
	m_parsed_tri_faces.clear();
	m_parsed_vertices.clear();
}


void WavefrontObjLoader::parse_lines(std::istream& obj_stream) {
	std::string current_line;
	while (std::getline(obj_stream, current_line)) {
		process_text_line(current_line);
	}
}

void WavefrontObjLoader::process_text_line(const std::string& line) {
	std::stringstream ss;
	ss << line;
	std::string command;
	ss >> command;	
	dispatch_command(command, ss);
}

void WavefrontObjLoader::dispatch_command(const std::string& command, std::stringstream& rest) {
	if (command == "#") {
		// ignore comment lines
	} else if (command == "v") {
		process_vertex_data(rest);
	} else if (command == "vn") {
		process_vertex_normal_data(rest);
	} else if (command == "f") {
		process_face_data(rest);
	} else {
		// ignore unknown commands
	}
}

void WavefrontObjLoader::process_vertex_data(std::istream& rest) {
    Vector3d vertex;
    rest >> vertex;
    if (rest.fail()) {
        throw std::runtime_error("failed to parse vertex");
    }
    m_parsed_vertices.push_back(vertex);
}

void WavefrontObjLoader::process_vertex_normal_data(std::istream& rest) {
    Vector3d normal;
    rest >> normal;
    if (rest.fail()) {
        throw std::runtime_error("failed to parse vertex normal");
    }
    m_parsed_normals.push_back(normal);
}

void WavefrontObjLoader::process_face_data(std::istream& rest) {
    TriFace face;
    rest >> face;
    // TODO: Check that not more stuff to parse left?
    m_parsed_tri_faces.push_back(face);
}

std::istream& operator>>(std::istream& is, Vector3d& v) {
    return (is >> v.x >> v.y >> v.z);
}

std::istream& operator>>(std::istream& is, TriFace& f) {
    std::string part;
    
    for (size_t vertex_no = 0; vertex_no < 3; vertex_no++) {
        is >> part;
        if (is.fail()) {
            throw std::runtime_error("failed reading triangle vertex");
        }
        std::stringstream part_ss(part);
        std::vector<std::string> part_tokens;
        std::string token;
        while (std::getline(part_ss, token, '/')) {
            part_tokens.push_back(token);
        }

        if (part_tokens.size() != 3) {
            throw std::runtime_error("failed parsing triangle vertex");
        }

        try {
            const auto vertex_ind = boost::numeric_cast<size_t>(std::stoi(part_tokens[0]) - 1);
            const auto normal_ind = boost::numeric_cast<size_t>(std::stoi(part_tokens[2]) - 1);
            f.vertex_inds[vertex_no] = vertex_ind;
            f.normal_inds[vertex_no] = normal_ind;
        } catch (boost::numeric::bad_numeric_cast& /*e*/) {
            throw std::runtime_error("bad numerical cast detected");
        } catch (std::runtime_error& /*e*/) {
            throw std::runtime_error("error interpreting triangle face");
        }
    }
    return is;
}

size_t WavefrontObjLoader::num_vertices() const {
    return m_vertex_data.size() / 3;
}

const double* WavefrontObjLoader::vertex_data() const {
    return m_vertex_data.data();
}

const double* WavefrontObjLoader::normal_data() const {
    return m_normal_data.data();
}

void WavefrontObjLoader::expand_vertices_and_normals() {
    const auto num_triangles = m_parsed_tri_faces.size();
    const auto vector_size = num_triangles*3*3;
    m_vertex_data.reserve(vector_size);
    m_normal_data.reserve(vector_size);

    for (const auto& tri : m_parsed_tri_faces) {
        for (size_t i = 0; i < 3; i++) {
            const auto v_ind = tri.vertex_inds[i];
            if (v_ind >= m_parsed_vertices.size()) {
                throw std::runtime_error("invalid vertex index");
            }
            m_vertex_data.push_back(m_parsed_vertices[v_ind].x);
            m_vertex_data.push_back(m_parsed_vertices[v_ind].y);
            m_vertex_data.push_back(m_parsed_vertices[v_ind].z);

            const auto n_ind = tri.normal_inds[i];
            if (n_ind >= m_parsed_normals.size()) {
                throw std::runtime_error("invalid normal index");
            }
            m_normal_data.push_back(m_parsed_normals[n_ind].x);
            m_normal_data.push_back(m_parsed_normals[n_ind].y);
            m_normal_data.push_back(m_parsed_normals[n_ind].z);
        }
    }
}

}   // end namespace
