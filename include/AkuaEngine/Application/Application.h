#ifndef AKUAENGINE_APPLICATION_H
#define AKUAENGINE_APPLICATION_H

#include <AkuaEngine/Application/InputSystem.h>
#include <AkuaEngine/Rendering/Renderer.h>
#include <AkuaEngine/Scene/Scene.h>
#include <AkuaEngine/Camera/Camera.h>
#include <AkuaEngine/Simulation/PBFSolver.h>
#include <AkuaEngine/Simulation/PBFConfig.h>
#include <AkuaEngine/Interop/InteropManager.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace AkuaEngine {

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
    float _deltaTime = 0.016f;

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

} // namespace AkuaEngine

#endif // AKUAENGINE_APPLICATION_H