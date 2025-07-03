#ifndef AQUAFORGE_INTEROP_MANAGER_H
#define AQUAFORGE_INTEROP_MANAGER_H

#include <AquaForge/Interop/InteropResource.h>
#include <AquaForge/Scene/SceneObject.h>
#include <glad/glad.h>
#include <unordered_map>

namespace AquaForge {

class InteropManager {
public:
    ~InteropManager();

    void registerInteropResource(SceneObject* object, GLuint vbo);
    InteropResource& getInteropResource(SceneObject* object);
    void releaseAll();

private:
    std::unordered_map<SceneObject*, InteropResource> _interopMap;
};

} // namespace AquaForge

#endif // AQUAFORGE_INTEROP_MANAGER_H