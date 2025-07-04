#include <AquaForge/Rendering/Renderer.h>
#include <AquaForge/Camera/Camera.h>
#include <AquaForge/Scene/Scene.h>
#include <AquaForge/Scene/SceneObject.h>
#include <AquaForge/Scene/SceneObjectType.h>
#include <AquaForge/Interop/InteropManager.h>
#include <AquaForge/Rendering/MeshBufferGL.h>
#include <AquaForge/Rendering/ParticlesBufferGL.h>
#include <AquaForge/Simulation/ParticleSystem.h>
#include <AquaForge/Simulation/Particle.h>
#include <AquaForge/Shader/ShaderProgram.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <unordered_map>

namespace AquaForge {

void Renderer::bindScene(Scene* scene, InteropManager* interopManager) {
    clearBuffers();
    _activeScene = scene;

    for (SceneObject* object : scene->getObjects()) {
        uploadObject(object, interopManager);
    }
}

void Renderer::render(Camera& camera) {
    glm::mat4 view = camera.getViewMatrix();
    glm::mat4 projection = glm::perspective(glm::radians(camera.getFOV()), _aspectRatio, _nearZ, _farZ);

    for (auto& [object, buffer] : _meshBuffers) {
        ShaderProgram* shader = object->getMaterial()->getShaderProgram();
        shader->bind();
        shader->setUniform<glm::mat4>("projection", projection); // TODO: wrap string literals in enum
        shader->setUniform<glm::mat4>("view", view);
        shader->setUniform<glm::mat4>("model", object->getModelMatrix());

        glBindVertexArray(buffer.vao);
        glDrawArrays(GL_TRIANGLES, 0, buffer.vertexCount);
        glBindVertexArray(0);
    }

    for (auto& [object, buffer] : _particleBuffers) {
        ShaderProgram* shader = object->getMaterial()->getShaderProgram();
        shader->bind();
        shader->setUniform<glm::mat4>("projection", projection);
        shader->setUniform<glm::mat4>("view", view);
        shader->setUniform<glm::mat4>("model", object->getModelMatrix());
        shader->setUniform<glm::vec3>("cameraPosition", camera.getPosition());

        glBindVertexArray(buffer.vao);
        glDrawArraysInstanced(GL_POINTS, 0, 1, buffer.instanceCount);
        glBindVertexArray(0);
    }
}

void Renderer::clearBuffers() {
    for (auto& [_, buffer] : _meshBuffers) {
        glDeleteVertexArrays(1, &buffer.vao);
        glDeleteBuffers(1, &buffer.vbo);
    }
    for (auto& [_, buffer] : _particleBuffers) {
        glDeleteVertexArrays(1, &buffer.vao);
        glDeleteBuffers(1, &buffer.vbo);
    }

    _meshBuffers.clear();
    _particleBuffers.clear();
    _activeScene = nullptr;
}

void Renderer::uploadObject(SceneObject* object, InteropManager* interopManager) {
    if (object->getObjectType() == SceneObjectType::Mesh) {
        Mesh* mesh = object->getMesh();
        MeshBufferGL buffer = uploadMesh(mesh);
        if (object->requiresInterop()) {
            interopManager->registerInteropResource(object, buffer.vbo);
        }
        _meshBuffers[object] = buffer;
    } else if (object->getObjectType() == SceneObjectType::ParticleSystem) {
        ParticleSystem* ps = object->getParticleSystem();
        ParticlesBufferGL buffer = uploadParticles(ps);
        if (object->requiresInterop()) {
            interopManager->registerInteropResource(object, buffer.vbo);
        }
        _particleBuffers[object] = buffer;
    }
}

/**
 * Construct an interleaved mesh data array in the OpenGL convention.
 * Then upload it to GPU.
 */
MeshBufferGL Renderer::uploadMesh(Mesh* mesh) {
    // Creating interleaved array
    const auto& positions = mesh->getVertices();
    const auto& normals = mesh->getNormals();
    const auto& tangents = mesh->getTangents();
    const auto& uvs = mesh->getUVs();
    const auto& indices = mesh->getIndices();

    bool hasNormals = mesh->hasNormals();
    bool hasTangents = mesh->hasTangents();
    bool hasUVs = mesh->hasUVs();

    const int floatsPerVertex = 3 + (hasNormals ? 3 : 0) + (hasTangents ? 3 : 0) + (hasUVs ? 2 : 0);
    std::vector<float> interleaved;
    interleaved.reserve(indices.size() * floatsPerVertex);

    for (uint32_t index : indices) {
        const glm::vec3& pos = positions[index];
        interleaved.push_back(pos.x);
        interleaved.push_back(pos.y);
        interleaved.push_back(pos.z);

        if (hasNormals) {
            const glm::vec3& n = normals[index];
            interleaved.push_back(n.x);
            interleaved.push_back(n.y);
            interleaved.push_back(n.z);
        }

        if (hasTangents) {
            const glm::vec3& t = tangents[index];
            interleaved.push_back(t.x);
            interleaved.push_back(t.y);
            interleaved.push_back(t.z);
        }

        if (hasUVs) {
            const glm::vec2& uv = uvs[index];
            interleaved.push_back(uv.x);
            interleaved.push_back(uv.y);
        }
    }

    // Upload data
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, interleaved.size() * sizeof(float), interleaved.data(), GL_STATIC_DRAW);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "[AquaForge::Renderer::uploadMesh] Error uploading buffer data:\n" << error << std::endl;
    }

    // Attributes are set dynamically based on what's provided in mesh.
    // Position -> Normal -> Tangent -> UV
    // Shader should match that (TODO: generalize this for all shaders)
    int offset = 0;
    GLuint location = 0;

    glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, floatsPerVertex * sizeof(float), (void*)(offset * sizeof(float)));
    glEnableVertexAttribArray(location++);
    offset += 3;

    if (hasNormals) {
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, floatsPerVertex * sizeof(float), (void*)(offset * sizeof(float)));
        glEnableVertexAttribArray(location++);
        offset += 3;
    }

    if (hasTangents) {
        glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE, floatsPerVertex * sizeof(float), (void*)(offset * sizeof(float)));
        glEnableVertexAttribArray(location++);
        offset += 3;
    }

    if (hasUVs) {
        glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE, floatsPerVertex * sizeof(float), (void*)(offset * sizeof(float)));
        glEnableVertexAttribArray(location++);
        offset += 2;
    }

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return MeshBufferGL { vao, vbo, static_cast<GLsizei>(indices.size()) };
}

ParticlesBufferGL Renderer::uploadParticles(ParticleSystem* ps) {
    const std::vector<Particle>& particles = ps->getParticles();

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Particle), particles.data(), GL_STATIC_DRAW); // particles.data() is the safer version of &particles[0]
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "[AquaForge::Renderer::uploadParticles] Error uploading buffer data:\n" << error << std::endl;
    }

    // Position (vec3) at location 0
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, position));
    glEnableVertexAttribArray(0);
    glVertexAttribDivisor(0, 1);

    // Size (float) at location 1
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(offsetof(Particle, size)));
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);
    
    // Color (vec4) at location 2
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(offsetof(Particle, color)));
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return ParticlesBufferGL { vao, vbo, static_cast<GLsizei>(particles.size()) };
}

} // namespace AquaForge