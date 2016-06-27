set(trianglemesh3d_SRCS
    TriangleMesh3d.cpp
    WavefrontObjLoader.cpp
    )
set(trianglemesh3d_HEADERS
    TriangleMesh3d.hpp
    WavefrontObjLoader.hpp
    )

add_library(trianglemesh3d
            ${trianglemesh3d_SRCS}
            ${trianglemesh3d_HEADERS}
            )
target_link_libraries(trianglemesh3d Boost::boost)
