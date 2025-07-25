#ifndef AKUAENGINE_CAMERA_CONFIG_H
#define AKUAENGINE_CAMERA_CONFIG_H

#include <glm/glm.hpp>

namespace AkuaEngine {

struct CameraConfig {
    float yaw = -90.0f;
    float pitch = 0.0f;
    float movementSpeed = 1.0f;
    float mouseSensitivity = 0.1f;
    float fov = 45.0f;
    glm::vec3 position = {3.0f, 2.0f, 6.0f};
    glm::vec3 worldUp = {0.0f, 1.0f, 0.0f};
};

}

#endif // AKUAENGINE_CAMERA_CONFIG_H