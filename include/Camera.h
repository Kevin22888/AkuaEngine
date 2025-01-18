#ifndef PBF_CAMERA_H
#define PBS_CAMERA_H

#include <Camera_Defaults.h>
#include <glm/glm.hpp>

enum CameraMovement {
    FORWARD, BACKWARD, LEFT, RIGHT
};

// Camera class
// Handles camera in the 3D scene in OpenGL and computes the view transformation matrix in MVP.
class Camera {
private:
    // Camera properties and control attributes
    glm::vec3 position, gaze, right, cameraUp, worldUp;
    float fov, yaw, pitch, movementSpeed, mouseSensitivity;

    void updateVectors();
public:
    Camera(glm::vec3 position = CameraDefaults::POSITION, 
           glm::vec3 worldUp = CameraDefaults::WORLD_UP, 
           float yaw = CameraDefaults::YAW, 
           float pitch = CameraDefaults::PITCH);

    glm::mat4 getViewMatrix();
    float getFOV();
    glm::vec3 getPosition() const;
    void updatePosition(CameraMovement direction, float deltaTime);
    void updateOrientation(float xoffset, float yoffset);
    void updateFOV(float yoffset);
};

#endif // PBS_CAMERA_H