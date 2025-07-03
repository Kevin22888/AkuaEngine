#ifndef AQUAFORGE_SCENE_H
#define AQUAFORGE_SCENE_H

#include <AquaForge/Scene/SceneObject.h>
#include <vector>

namespace AquaForge {

class Scene {
public:
    ~Scene();

    void addObject(SceneObject* object);
    const std::vector<SceneObject*>& getObjects() const;

private:
    std::vector<SceneObject*> _objects;
};

} // namespace AquaForge

#endif // AQUAFORGE_SCENE_H