#include <AquaForge/Scene/Scene.h>
#include <AquaForge/Scene/SceneObject.h>
#include <vector>

namespace AquaForge {

Scene::~Scene() {
    for (SceneObject* object : _objects) {
        delete object;
    }
    _objects.clear();
}

void Scene::addObject(SceneObject* object) {
    _objects.push_back(object);
}

const std::vector<SceneObject*>& Scene::getObjects() const {
    return _objects;
}


} // namespace AquaForge