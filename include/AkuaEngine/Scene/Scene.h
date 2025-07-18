#ifndef AKUAENGINE_SCENE_H
#define AKUAENGINE_SCENE_H

#include <AkuaEngine/Scene/SceneObject.h>
#include <vector>

namespace AkuaEngine {

class Scene {
public:
    ~Scene();

    void addObject(SceneObject* object);
    const std::vector<SceneObject*>& getObjects() const;
    void clear();

private:
    std::vector<SceneObject*> _objects;
};

} // namespace AkuaEngine

#endif // AKUAENGINE_SCENE_H