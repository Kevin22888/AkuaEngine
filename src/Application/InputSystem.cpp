#include <AkuaEngine/Application/InputSystem.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace AkuaEngine {

void InputSystem::handleMouseMovement(double xPos, double yPos) {
    float x = static_cast<float>(xPos);
    float y = static_cast<float>(yPos);

    // With this, the first mouse delta will always be 0
    if (_firstMouseInput) {
        _lastX = x;
        _lastY = y;
        _firstMouseInput = false;
    }

    float xOffset = x - _lastX;
    float yOffset = _lastY - y; // y coord goes from bottom to top
    _lastX = x;
    _lastY = y;

    if (_camera) {
        _camera->updateOrientation(xOffset, yOffset);
    }
}

void InputSystem::handleScroll(double xOffset, double yOffset) {
    _camera->updateFOV(static_cast<float>(yOffset));
}

void InputSystem::handleResize(int width, int height) {
    _windowWidth = width;
    _windowHeight = height;
    _aspectRatio = static_cast<float>(width) / height;
}

void InputSystem::processKeyInputs(GLFWwindow* window, float deltaTime) {
    // This is processing inputs, not polling, because we are making meaningful
    // changes to the internal states
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        _camera->updatePosition(CameraMovement::Forward, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        _camera->updatePosition(CameraMovement::Backward, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        _camera->updatePosition(CameraMovement::Left, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        _camera->updatePosition(CameraMovement::Right, deltaTime);
}

} // namespace AkuaEngine
