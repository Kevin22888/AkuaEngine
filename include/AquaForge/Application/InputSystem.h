#ifndef AQUAFORGE_INPUT_SYSTEM_H
#define AQUAFORGE_INPUT_SYSTEM_H

#include <AquaForge/Camera/Camera.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace AquaForge {

class InputSystem {
public:
    InputSystem(int windowWidth, int windowHeight);

    void assignCamera(Camera* camera);
    void handleMouseMovement(double xPos, double yPos);
    void handleScroll(double xOffset, double yOffset);
    void handleResize(int width, int height);

    // This is the only function making InputSystem depend on GLFW, for a more
    // unit-testable InputSystem, we should abstract away the GLFW-dependency
    // maybe using command pattern.
    void processKeyInputs(GLFWwindow* window, float deltaTime);

private:
    Camera* _camera = nullptr;

    // These aren't used now but will be required for cursor ray picking
    int _windowWidth;
    int _windowHeight;
    float _aspectRatio;

    // For mouse delta
    float _lastX;
    float _lastY;

    // Prevent camera jump when the mouse is first captured
    bool _firstMouseInput = true;
};

// ============================= Inline functions ==============================

inline InputSystem::InputSystem(int windowWidth, int windowHeight) : _windowWidth(windowWidth), _windowHeight(windowHeight) {
    _aspectRatio = static_cast<float>(windowWidth) / windowHeight;
    // These actually don't matter because of our firstMouseInput trick
    _lastX = windowWidth / 2.0f;
    _lastY = windowHeight / 2.0f;
}

inline void InputSystem::assignCamera(Camera* camera) {
    _camera = camera;
}


} // namespace AquaForge

#endif // AQUAFORGE_INPUT_SYSTEM_H