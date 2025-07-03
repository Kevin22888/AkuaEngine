#include <AquaForge/Camera/Camera.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cmath>

namespace {

constexpr float MIN_PITCH = -89.0f;
constexpr float MAX_PITCH = 89.0f;
constexpr float MIN_FOV = 1.0f;
constexpr float MAX_FOV = 90.0f;
constexpr float FULL_CIRCLE_DEGREES = 360.0f;
constexpr float ZERO_DEGREES = 0.0f;

}

namespace AquaForge {

Camera::Camera(const CameraConfig& config)
    : _position(config.position),
      _worldUp(config.worldUp),
      _yaw(config.yaw),
      _pitch(config.pitch),
      _movementSpeed(config.movementSpeed),
      _mouseSensitivity(config.mouseSensitivity),
      _fov(config.fov) {
    updateVectors();
}

glm::mat4 Camera::getViewMatrix() {
    return glm::lookAt(_position, _position + _gaze, _cameraUp);
}

void Camera::updatePosition(CameraMovement direction, float deltaTime) {
    float velocity = _movementSpeed * deltaTime;

    switch (direction) {
        case CameraMovement::Forward:
            _position += _gaze * velocity;
            break;
        case CameraMovement::Backward:
            _position -= _gaze * velocity;
            break;
        case CameraMovement::Left:
            _position -= _cameraRight * velocity;
            break;
        case CameraMovement::Right:
            _position += _cameraRight * velocity;
            break;
    }
}

void Camera::updateOrientation(float xoffset, float yoffset) {
    xoffset *= _mouseSensitivity;
    yoffset *= _mouseSensitivity;

    _yaw = std::fmodf(_yaw + xoffset, FULL_CIRCLE_DEGREES);
    if (_yaw < ZERO_DEGREES) _yaw += FULL_CIRCLE_DEGREES;

    _pitch -= yoffset;
    if (_pitch > MAX_PITCH) _pitch = MAX_PITCH;
    if (_pitch < MIN_PITCH) _pitch = MIN_PITCH;

    updateVectors();
}

void Camera::updateFOV(float yoffset) {
    _fov -= yoffset;
    if (_fov < MIN_FOV) _fov = MIN_FOV;
    if (_fov > MAX_FOV) _fov = MAX_FOV;
}

void Camera::updateVectors() {
    // the yaw and pitch are angles, but they affect the cartesian vector of the 
    // camera's gaze direction. Pitch has influence on the xz-plane even though 
    // it appears to be about how y-axis moves.
    glm::vec3 front;
    front.x = cos(glm::radians(_yaw)) * cos(glm::radians(_pitch));
    front.y = sin(glm::radians(_pitch));
    front.z = sin(glm::radians(_yaw)) * cos(glm::radians(_pitch));
    _gaze = glm::normalize(front);

    // re-calculate right and up vectors
    _cameraRight = glm::normalize(glm::cross(_gaze, _worldUp));
    _cameraUp = glm::normalize(glm::cross(_cameraRight, _gaze));
}

} // namespace AquaForge