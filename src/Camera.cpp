#include <Camera.h>

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(glm::vec3 position, glm::vec3 worldUp, float yaw, float pitch)
    : position(position),
      worldUp(worldUp),
      yaw(yaw),
      pitch(pitch),
      gaze(CameraDefaults::GAZE),
      movementSpeed(CameraDefaults::SPEED),
      mouseSensitivity(CameraDefaults::MOUSE_SENSITIVITY),
      fov(CameraDefaults::FOV) {
    updateVectors();
}

glm::mat4 Camera::getViewMatrix() {
    return glm::lookAt(position, position + gaze, cameraUp);
}

float Camera::getFOV() {
    return fov;
}

glm::vec3 Camera::getPosition() const {
    return position;
}

// Handle keyboard input
void Camera::updatePosition(CameraMovement direction, float deltaTime) {
    float velocity = movementSpeed * deltaTime;

    if (direction == FORWARD)
        position += gaze * velocity;
    if (direction == BACKWARD)
        position -= gaze * velocity;
    if (direction == LEFT)
        position -= right * velocity;
    if (direction == RIGHT)
        position += right * velocity;
}

// Handle mouse movement input
void Camera::updateOrientation(float xoffset, float yoffset) {
    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;

    yaw += xoffset;
    pitch -= yoffset;

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    updateVectors();
}

// Handle mouse scroll input (only vertical)
void Camera::updateFOV(float yoffset) {
    fov -= yoffset;
    if (fov < 1.0f) fov = 1.0f;
    if (fov > 45.0f) fov = 45.0f;
}

void Camera::updateVectors() {
    // the yaw and pitch are angles, but they affect the cartesian vector of the 
    // camera's gaze direction. Pitch has influence on the xz-plane even though it appears to
    // be about how y-axis moves.
    glm::vec3 front;
    front.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    front.y = sin(glm::radians(pitch));
    front.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    gaze = glm::normalize(front);

    // re-calculate right and up vectors
    right = glm::normalize(glm::cross(gaze, worldUp));
    cameraUp = glm::normalize(glm::cross(right, gaze));
}