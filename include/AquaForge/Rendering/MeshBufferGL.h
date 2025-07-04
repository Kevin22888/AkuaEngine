#ifndef AQUAFORGE_MESH_BUFFER_GL_H
#define AQUAFORGE_MESH_BUFFER_GL_H

#include <glad/glad.h>

namespace AquaForge {

/**
 * Note: the vertex count is not how many unique vertices are in the mesh, but
 * how many vertices need to be visited to draw the whole thing. In other words,
 * this is equal to the number of triangle indices.
 */
struct MeshBufferGL {
    GLuint vao;
    GLuint vbo;
    GLsizei vertexCount;
};

} // namespace AquaForge

#endif // AQUAFORGE_MESH_BUFFER_GL_H