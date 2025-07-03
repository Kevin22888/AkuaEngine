#ifndef AQUAFORGE_SCENE_OBJECT_H
#define AQUAFORGE_SCENE_OBJECT_H

#include <AquaForge/Scene/SceneObjectType.h>
#include <AquaForge/Graphics/Material.h>
#include <AquaForge/Graphics/Mesh.h>
#include <AquaForge/Simulation/ParticleSystem.h>
#include <glm/glm.hpp>

namespace AquaForge {

class SceneObject {
public:
    // Factory functions
    static SceneObject* createMeshObject(Material* material, Mesh* mesh, bool requireInterop);
    static SceneObject* createParticleSystemObject(Material* material, ParticleSystem* ps, bool requireInterop);

    ~SceneObject();

    SceneObject(const SceneObject& other) = delete;
    SceneObject& operator=(const SceneObject& other) = delete;
    SceneObject(SceneObject&& other) noexcept;
    SceneObject& operator=(SceneObject&& other) noexcept;

    SceneObjectType getObjectType() const;
    bool requiresInterop() const;
    glm::mat4 getModelMatrix() const;
    
    Material* getMaterial() const;
    Mesh* getMesh() const;
    ParticleSystem* getParticleSystem() const;

private:
    SceneObjectType _type;
    bool _requireInterop;
    Material* _material;
    Mesh* _mesh;
    ParticleSystem* _particleSystem;
    // TODO: Add transform class

    SceneObject(Material* material, Mesh* mesh, ParticleSystem* ps, SceneObjectType objectType, bool requireInterop);
};

// ============================= Inline functions ==============================

inline SceneObjectType SceneObject::getObjectType() const {
    return _type;
}

inline bool SceneObject::requiresInterop() const {
    return _requireInterop;
}

inline Material* SceneObject::getMaterial() const {
    return _material;
}

inline Mesh* SceneObject::getMesh() const {
    return _mesh;
}

inline ParticleSystem* SceneObject::getParticleSystem() const {
    return _particleSystem;
}

} // namespace AquaForge

#endif // AQUAFORGE_SCENE_OBJECT_H