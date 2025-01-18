#ifndef PBF_CAMERA_DEFAULTS_H
#define PBF_CAMERA_DEFAULTS_H

#include <glm/glm.hpp>

namespace CameraDefaults {
    const float YAW = -90.0f;
    const float PITCH = 0.0f;
    const float SPEED = 5.0f;
    const float MOUSE_SENSITIVITY = 0.1f;
    const float FOV = 45.0f;

    const glm::vec3 POSITION(0.0f, 0.0f, 5.0f); //POSITION(0.0f, 0.0f, 0.0f);
    const glm::vec3 WORLD_UP(0.0f, 1.0f, 0.0f);
    const glm::vec3 GAZE(0.0f, 0.0f, -1.0f); //GAZE(0.0f, 0.0f, -1.0f);
}

#endif // PBF_CAMERA_DEFAULTS_H