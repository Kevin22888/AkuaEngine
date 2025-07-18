#ifndef AKUAENGINE_MESH_BUFFER_GL_H
#define AKUAENGINE_MESH_BUFFER_GL_H

#include <glad/glad.h>

namespace AkuaEngine {

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

} // namespace AkuaEngine

#endif // AKUAENGINE_MESH_BUFFER_GL_H