#ifndef AKUAENGINE_INTEROP_MANAGER_H
#define AKUAENGINE_INTEROP_MANAGER_H

#include <AkuaEngine/Interop/InteropResource.h>
#include <AkuaEngine/Scene/SceneObject.h>
#include <glad/glad.h>
#include <unordered_map>

namespace AkuaEngine {

class InteropManager {
public:
    ~InteropManager();

    void registerInteropResource(SceneObject* object, GLuint vbo);
    InteropResource& getInteropResource(SceneObject* object);
    void releaseAll();

private:
    std::unordered_map<SceneObject*, InteropResource> _interopMap;
};

} // namespace AkuaEngine

#endif // AKUAENGINE_INTEROP_MANAGER_H