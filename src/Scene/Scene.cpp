#include <AkuaEngine/Scene/Scene.h>
#include <AkuaEngine/Scene/SceneObject.h>
#include <vector>
#include <iostream>

namespace AkuaEngine {

Scene::~Scene() {
    clear();
}

// For O(1) insertion, use unordered_set. But I'm keeping the vector in case I need the order
void Scene::addObject(SceneObject* object) {
    if (std::find(_objects.begin(), _objects.end(), object) == _objects.end()) {
        _objects.push_back(object);
    } else {
        std::cerr << "Adding duplicate SceneObject" << std::endl;
    }
}

const std::vector<SceneObject*>& Scene::getObjects() const {
    return _objects;
}

void Scene::clear() {
    for (SceneObject* object : _objects) {
        delete object;
    }
    _objects.clear();
}

} // namespace AkuaEngine