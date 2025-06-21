#ifndef AQUAFORGE_CAMERA_H
#define AQUAFORGE_CAMERA_H

#include "CameraConfig.h"
#include "CameraMovement.h"
#include <glm/glm.hpp>

namespace AquaForge {

// Camera class
// Handles camera in the 3D scene in OpenGL and computes the view transformation matrix in MVP.
class Camera {
public:
    explicit Camera(const CameraConfig& config);

    float getFOV() const;
    glm::vec3 getPosition() const;
    glm::vec3 getGaze() const;
    glm::vec3 getCameraRight() const;
    glm::vec3 getCameraUp() const;
    glm::mat4 getViewMatrix();

    void updatePosition(CameraMovement direction, float deltaTime);
    void updateOrientation(float xoffset, float yoffset);
    void updateFOV(float yoffset);

private:
    // Camera properties and control attributes
    glm::vec3 _position;
    glm::vec3 _gaze;
    glm::vec3 _cameraRight;
    glm::vec3 _cameraUp;
    glm::vec3 _worldUp;
    float _fov;
    float _yaw;
    float _pitch;
    float _movementSpeed;
    float _mouseSensitivity;

    void updateVectors();
};

// ============================= Inline functions ==============================

inline float Camera::getFOV() const { return _fov; }
inline glm::vec3 Camera::getPosition() const { return _position; }
inline glm::vec3 Camera::getGaze() const { return _gaze; }
inline glm::vec3 Camera::getCameraRight() const { return _cameraRight; }
inline glm::vec3 Camera::getCameraUp() const { return _cameraUp; }

} // namespace AquaForge

#endif // AQUAFORGE_CAMERA_H