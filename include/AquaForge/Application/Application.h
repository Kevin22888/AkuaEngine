#ifndef AQUAFORGE_APPLICATION_H
#define AQUAFORGE_APPLICATION_H

#include <AquaForge/Application/InputSystem.h>
#include <AquaForge/Rendering/Renderer.h>
#include <AquaForge/Scene/Scene.h>
#include <AquaForge/Camera/Camera.h>
#include <AquaForge/Simulation/PBFSolver.h>
#include <AquaForge/Simulation/PBFConfig.h>
#include <AquaForge/Interop/InteropManager.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace AquaForge {

class Application {
public:
    Application();

    int run();

private:
    // Window configs
    int _windowWidth = 1280;
    int _windowHeight = 720;
    GLFWwindow* _window;

    // PBF
    PBFConfig _config;
    LambdaCorrParams _corrParams;
    float _deltaTime = 0.0083f;

    // Subsystems
    Camera _camera;
    InputSystem _input;
    Scene _damBreakScene;
    PBFSolver _solver;
    Renderer _renderer;
    InteropManager _interopManager;

    bool _simulationActive = false;
    SceneObject* _fluidObject = nullptr; // keeping it here for now because we need to query InteropManager

    bool init();
    void registerInputCallbacks();
    void onWindowResize(int width, int height);
    void cleanUp();
    void prepareDamBreak();
};

} // namespace AquaForge

#endif // AQUAFORGE_APPLICATION_H