#ifndef AQUAFORGE_PARTICLES_BUFFER_GL_H
#define AQUAFORGE_PARTICLES_BUFFER_GL_H

#include <glad/glad.h>

namespace AquaForge {

struct ParticlesBufferGL {
    GLuint vao;
    GLuint vbo;
    GLsizei instanceCount;
};

} // namespace AquaForge

#endif // AQUAFORGE_PARTICLES_BUFFER_GL_H