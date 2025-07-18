#ifndef AKUAENGINE_RENDERER_H
#define AKUAENGINE_RENDERER_H

#include <AkuaEngine/Camera/Camera.h>
#include <AkuaEngine/Scene/Scene.h>
#include <AkuaEngine/Scene/SceneObject.h>
#include <AkuaEngine/Interop/InteropManager.h>
#include <AkuaEngine/Rendering/MeshBufferGL.h>
#include <AkuaEngine/Rendering/ParticlesBufferGL.h>
#include <unordered_map>

namespace AkuaEngine {

class Renderer {
public:
    Renderer(float aspectRatio, float nearZ, float farZ);

    void bindScene(Scene* scene, InteropManager* interopManager);
    void render(Camera& camera);
    void clearBuffers();
    void setAspectRatio(float aspectRatio);
    void setNearZ(float nearZ);
    void setFarZ(float farZ);

private:
    float _aspectRatio;
    float _nearZ;
    float _farZ;

    Scene* _activeScene;
    std::unordered_map<SceneObject*, MeshBufferGL> _meshBuffers;
    std::unordered_map<SceneObject*, ParticlesBufferGL> _particleBuffers;

    void uploadObject(SceneObject* object, InteropManager* interopManager);
    MeshBufferGL uploadMesh(Mesh* mesh);
    ParticlesBufferGL uploadParticles(ParticleSystem* ps);
};

// ============================= Inline functions ==============================

inline Renderer::Renderer(float aspectRatio, float nearZ, float farZ) : _aspectRatio(aspectRatio), _nearZ(nearZ), _farZ(farZ) {}

inline void Renderer::setAspectRatio(float aspectRatio) { _aspectRatio = aspectRatio; }

inline void Renderer::setNearZ(float nearZ) { _nearZ = nearZ; }

inline void Renderer::setFarZ(float farZ) { _farZ = farZ; }

} // namespace AkuaEngine

#endif // AKUAENGINE_RENDERER_H