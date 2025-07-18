#ifndef AKUAENGINE_MESH_H
#define AKUAENGINE_MESH_H

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

namespace AkuaEngine {

class Mesh {
public:
    void setVertices(std::vector<glm::vec3>&& vertices);
    void setNormals(std::vector<glm::vec3>&& normals);
    void setTangents(std::vector<glm::vec3>&& tangents);
    void setUVs(std::vector<glm::vec2>&& uvs);
    void setIndices(std::vector<uint32_t>&& indices);

    const std::vector<glm::vec3>& getVertices() const;
    const std::vector<glm::vec3>& getNormals() const;
    const std::vector<glm::vec3>& getTangents() const;
    const std::vector<glm::vec2>& getUVs() const;
    const std::vector<uint32_t>& getIndices() const;

    bool hasNormals() const;
    bool hasTangents() const;
    bool hasUVs() const;

private:
    std::vector<glm::vec3> _vertices;
    std::vector<glm::vec3> _normals;
    std::vector<glm::vec3> _tangents;
    std::vector<glm::vec2> _uvs;
    std::vector<uint32_t> _indices;
};

// ============================= Inline functions ==============================

inline void Mesh::setVertices(std::vector<glm::vec3>&& vertices) {
    _vertices = std::move(vertices);
}

inline void Mesh::setNormals(std::vector<glm::vec3>&& normals) {
    _normals = std::move(normals);
}

inline void Mesh::setTangents(std::vector<glm::vec3>&& tangents) {
    _tangents = std::move(tangents);
}

inline void Mesh::setUVs(std::vector<glm::vec2>&& uvs) {
    _uvs = std::move(uvs);
}

inline void Mesh::setIndices(std::vector<uint32_t>&& indices) {
    _indices = std::move(indices);
}

inline const std::vector<glm::vec3>& Mesh::getVertices() const {
    return _vertices;
}

inline const std::vector<glm::vec3>& Mesh::getNormals() const {
    return _normals;
}

inline const std::vector<glm::vec3>& Mesh::getTangents() const {
    return _tangents;
}

inline const std::vector<glm::vec2>& Mesh::getUVs() const {
    return _uvs;
}

inline const std::vector<uint32_t>& Mesh::getIndices() const {
    return _indices;
}

inline bool Mesh::hasNormals() const { 
    return !_normals.empty(); 
}

inline bool Mesh::hasTangents() const { 
    return !_tangents.empty();
}

inline bool Mesh::hasUVs() const { 
    return !_uvs.empty(); 
}

} // namespace AkuaEngine

#endif // AKUAENGINE_MESH_H