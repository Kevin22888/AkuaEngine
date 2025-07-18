#include <AkuaEngine/Interop/InteropManager.h>
#include <AkuaEngine/Interop/InteropResource.h>
#include <AkuaEngine/Scene/SceneObject.h>
#include <glad/glad.h>
#include <unordered_map>
#include <iostream>

namespace AkuaEngine {

InteropManager::~InteropManager() {
    releaseAll();
}

void InteropManager::registerInteropResource(SceneObject* object, GLuint vbo) {
    if (!object) return;

    if (_interopMap.find(object) != _interopMap.end()) {
        std::cerr << "[AkuaEngine::InteropManager::registerInteropResource] Object already registered with InteropResource" << std::endl;
        return;
    }

    // Use emplace to avoid copy; InteropResource is move-only
    _interopMap.emplace(object, InteropResource(vbo));
}

InteropResource& InteropManager::getInteropResource(SceneObject* object) {
    return _interopMap.at(object); // throws if object key does not exist
}

void InteropManager::releaseAll() {
    for (auto& [_, resource] : _interopMap) { // bypasses pair.second.release()
        resource.release();
    }
    _interopMap.clear();
}

} // namespace AkuaEngine