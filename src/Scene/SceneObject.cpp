#include <AquaForge/Scene/SceneObject.h>
#include <AquaForge/Scene/SceneObjectType.h>
#include <AquaForge/Graphics/Material.h>
#include <AquaForge/Graphics/Mesh.h>
#include <AquaForge/Simulation/ParticleSystem.h>
#include <glm/glm.hpp>

namespace AquaForge {

SceneObject::SceneObject(Material* material, Mesh* mesh, ParticleSystem* ps, SceneObjectType objectType, bool requireInterop) 
    : _material(material),
      _mesh(mesh),
      _particleSystem(ps),
      _type(objectType),
      _requireInterop(requireInterop) {}

SceneObject::~SceneObject() {
    delete _material;
    delete _mesh;
    delete _particleSystem;
}

SceneObject::SceneObject(SceneObject&& other) noexcept 
    : _material(other._material), 
      _mesh(other._mesh), 
      _particleSystem(other._particleSystem) 
{
    other._material = nullptr;
    other._mesh = nullptr;
    other._particleSystem = nullptr;
}

SceneObject& SceneObject::operator=(SceneObject&& other) noexcept {
    if (this != &other) {
        delete _material;
        delete _mesh;
        delete _particleSystem;

        _material = other._material;
        _mesh = other._mesh;
        _particleSystem = other._particleSystem;

        other._material = nullptr;
        other._mesh = nullptr;
        other._particleSystem = nullptr;
    }
    return *this;
}

// Factory functions
SceneObject* SceneObject::createMeshObject(Material* material, Mesh* mesh, bool requireInterop) {
    return new SceneObject(material, mesh, nullptr, SceneObjectType::Mesh, requireInterop);
}

SceneObject* SceneObject::createParticleSystemObject(Material* material, ParticleSystem* ps, bool requireInterop) {
    return new SceneObject(material, nullptr, ps, SceneObjectType::ParticleSystem, requireInterop);
}

// Because there is no transform right now, just return identity.
// Will add more complete geometry control later
glm::mat4 SceneObject::getModelMatrix() const {
    return glm::mat4(1.0f);
}

} // namespace AquaForge