#include <AkuaEngine/Application/Application.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vector>

namespace {

constexpr int NUM_PARTICLES = 27000;
// TODO: move these scene values to their own system
constexpr glm::vec3 BOX_MIN = {1.5f, 0.0f, 1.5f};
constexpr glm::vec3 BOX_MAX = {4.5f, 4.0f, 4.5f};
constexpr float FLOOR_LEVEL = -0.02f;
constexpr float NEAR_Z = 0.1f;
constexpr float FAR_Z = 160.0f;

}

namespace AkuaEngine {

Application::Application() 
    : _camera(CameraConfig{}),
      _input(_windowWidth, _windowHeight),
      _solver(NUM_PARTICLES, _config, _corrParams),
      _renderer(static_cast<float>(_windowWidth) / _windowHeight, NEAR_Z, FAR_Z)
{
    _input.assignCamera(&_camera);
}

int Application::run() {
    if (!init()) return -1;

    while (!glfwWindowShouldClose(_window)) {
        glfwPollEvents();

        if (glfwGetKey(_window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(_window, true);
        }

        _input.processKeyInputs(_window, _deltaTime);

        if (!_simulationActive && glfwGetKey(_window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            _simulationActive = true;
        }
        if (_simulationActive && glfwGetKey(_window, GLFW_KEY_B) == GLFW_PRESS) {
            _simulationActive = false;
        }

        if (_simulationActive) {
            _solver.step(_interopManager.getInteropResource(_fluidObject), _deltaTime, BOX_MIN, BOX_MAX);
        }

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        _renderer.render(_camera);

        glfwSwapBuffers(_window);
    }

    cleanUp();
    return 0;
}
    
bool Application::init() {
    if (!glfwInit()) {
        std::cerr << "[AkuaEngine::Application::init] Failed to initialize GLFW." << std::endl;
        return false;
    }

    // Request OpenGL 3.3 core profile (to avoid manually enabling GL_POINT_SPRITE)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Init window and set OpenGL context
    _window = glfwCreateWindow(_windowWidth, _windowHeight, "AkuaEngine v0.1", nullptr, nullptr);
    if (!_window) {
        std::cerr << "[AkuaEngine::Application::init] Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(_window); // Make an OpenGL context (_window) the main context on the current thread
    glfwSetWindowTitle(_window, "AkuaEngine");

    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) { // Load OpenGL functions (must come after context)
        std::cerr << "[AkuaEngine::Application::init] Failed to initialize GLAD." << std::endl;
        glfwDestroyWindow(_window);
        glfwTerminate();
        return false;
    }
    glEnable(GL_PROGRAM_POINT_SIZE); // Tell OpenGL to render points as many fragments instead of a single pixel
    glEnable(GL_DEPTH_TEST);

    glfwSetWindowUserPointer(_window, this); // Set window user pointer

    registerInputCallbacks();

    glfwSetInputMode(_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // Capture mouse

    prepareDamBreak();

    return true;
}

void Application::prepareDamBreak() {
    // ============= Floor =============
    Mesh* floorMesh = new Mesh();
    std::vector<glm::vec3> floorVertices = {
        {-2.0f, FLOOR_LEVEL, -2.0f}, // v0
        { 8.0f, FLOOR_LEVEL, -2.0f}, // v1
        { 8.0f, FLOOR_LEVEL,  8.0f}, // v2
        {-2.0f, FLOOR_LEVEL,  8.0f}  // v3
    };
    std::vector<glm::vec2> floorUVs = {
        {0.0f, 0.0f},   // v0
        {1.0f, 0.0f},  // v1
        {1.0f, 1.0f}, // v2
        {0.0f, 1.0f}   // v3
    };
    std::vector<uint32_t> floorIndices = {
        0, 1, 2, // Triangle 1
        0, 2, 3  // Triangle 2
    };
    floorMesh->setVertices(std::move(floorVertices));
    floorMesh->setUVs(std::move(floorUVs));
    floorMesh->setIndices(std::move(floorIndices));

    Material* floorMat = new Material("assets/shaders/floor.vert", "assets/shaders/floor.frag");
    floorMat->getShaderProgram()->bind();
    floorMat->getShaderProgram()->setUniform<glm::vec3>("color1", glm::vec3(0.8f, 0.8f, 0.8f)); // Light gray
    floorMat->getShaderProgram()->setUniform<glm::vec3>("color2", glm::vec3(0.5f, 0.5f, 0.5f)); // Dark gray
    floorMat->getShaderProgram()->setUniform<float>("scale", 20.0f); // smaller number makes larger tiles

    SceneObject* floorObject = SceneObject::createMeshObject(floorMat, floorMesh, false);
    _damBreakScene.addObject(floorObject);
    floorMesh = nullptr;
    floorMat = nullptr;
    floorObject = nullptr;

    // ============= Fluid =============
    glm::vec3 minPos = {2.0f, 1.0f, 2.0f};
    glm::vec3 maxPos = {3.5f, 2.5f, 3.5f};
    glm::vec3 dimensions = maxPos - minPos;

    int numX = static_cast<int>(dimensions.x / _config.particle_spacing);
    int numY = static_cast<int>(dimensions.y / _config.particle_spacing);
    int numZ = static_cast<int>(dimensions.z / _config.particle_spacing);

    std::vector<Particle> particles;
    particles.reserve(numX * numY * numZ);

    for (int x = 0; x < numX; ++x) {
        for (int y = 0; y < numY; ++y) {
            for (int z = 0; z < numZ; ++z) {
                glm::vec3 position = {
                    minPos.x + x * _config.particle_spacing,
                    minPos.y + y * _config.particle_spacing,
                    minPos.z + z * _config.particle_spacing
                };
                
                Particle p;
                p.position = position;
                p.velocity = glm::zero<glm::vec3>();
                p.mass = 1.0f;
                p.color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f); // blue
                p.size = 100.0f; // width in pixels for vertex shader

                particles.push_back(p);
            }
        }
    }

    ParticleSystem* ps = new ParticleSystem(std::move(particles));
    Material* fluidMat = new Material("assets/shaders/fluidParticle.vert", "assets/shaders/fluidParticle.frag");

    // This one we keep for now
    _fluidObject = SceneObject::createParticleSystemObject(fluidMat, ps, true);
    _damBreakScene.addObject(_fluidObject);
    ps = nullptr;
    fluidMat = nullptr;

    _renderer.bindScene(&_damBreakScene, &_interopManager);
}

void Application::registerInputCallbacks() {
    glfwSetFramebufferSizeCallback(_window, [](GLFWwindow* w, int width, int height) {
        auto* app = static_cast<Application*>(glfwGetWindowUserPointer(w));
        app->_input.handleResize(width, height);
        app->onWindowResize(width, height);
    });

    glfwSetCursorPosCallback(_window, [](GLFWwindow* w, double x, double y) {
        auto* app = static_cast<Application*>(glfwGetWindowUserPointer(w));
        app->_input.handleMouseMovement(x, y);
    });

    glfwSetScrollCallback(_window, [](GLFWwindow* w, double xoffset, double yoffset) {
        auto* app = static_cast<Application*>(glfwGetWindowUserPointer(w));
        app->_input.handleScroll(xoffset, yoffset);
    });
}

void Application::onWindowResize(int width, int height) {
    glViewport(0, 0, width, height);
    _windowWidth = width;
    _windowHeight = height;
    _renderer.setAspectRatio(static_cast<float>(_windowWidth) / _windowHeight);
}

void Application::cleanUp() {
    _renderer.clearBuffers();
    _damBreakScene.clear(); // Deletes shader program in each object's material
    _interopManager.releaseAll(); // Unregisters CUDA resource

    glfwTerminate();
}

} // namespace AkuaEngine
