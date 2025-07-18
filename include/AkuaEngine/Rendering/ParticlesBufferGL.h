#ifndef AKUAENGINE_PARTICLES_BUFFER_GL_H
#define AKUAENGINE_PARTICLES_BUFFER_GL_H

#include <glad/glad.h>

namespace AkuaEngine {

struct ParticlesBufferGL {
    GLuint vao;
    GLuint vbo;
    GLsizei instanceCount;
};

} // namespace AkuaEngine

#endif // AKUAENGINE_PARTICLES_BUFFER_GL_H