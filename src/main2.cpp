#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <Shader.h>
#include <Camera.h>
#include <Particle.h>
#include <SPH_Estimator.h>


/////////////////////////// Simulation configurations //////////////////////////
std::vector<Particle> particles;
std::vector<uint32_t> neighbourArray;
std::vector<uint32_t> neighbourCount;
std::vector<uint32_t> hashToFirstParticleIndex;

float particleSpacing = 0.25; //4913 particles due to fillParticle logic
glm::vec3 init_box_min(-2.0f, -2.0f, -2.0f);
glm::vec3 init_box_max(2.0f, 2.0f, 2.0f);

int num_particles = pow(static_cast<int>((init_box_max.x - init_box_min.x) / particleSpacing) + 1, 3);

float h = 1.0f; //0.5f;

const int MAX_NEIGHBOURS = 512;
const int TABLE_SIZE = 512 * num_particles;

int selectParticle = num_particles / 2 - 200;
////////////////////////////////////////////////////////////////////////////////


///////////////////////////// Window configurations ////////////////////////////
int windowWidth = 800;
int windowHeight = 600;

// Camera configurations
Camera camera(glm::vec3(0.0f, 0.0f, 5.0f));
bool firstMouseInput = true; // prevent jump when mouse is first captured
float lastX = windowWidth / 2.0f;
float lastY = windowHeight / 2.0f;
float deltaTime = 0.0f;
float timeAtLastFrame = 0.0f;

// Projection Matrix in MVP (re-calculated when mouse scroll or window is resized)
float nearZ = 0.1f;
float farZ = 100.0f;
glm::mat4 projection = glm::perspective(glm::radians(camera.getFOV()), 
                                        static_cast<float>(windowWidth) / static_cast<float>(windowHeight), 
                                        nearZ, farZ); //fov, aspect ratio, near and far planes

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouseInput) {
        lastX = xpos;
        lastY = ypos;
        firstMouseInput = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // y coord goes from bottom to top
    lastX = xpos;
    lastY = ypos;

    camera.updateOrientation(xoffset, yoffset);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.updateFOV(static_cast<float>(yoffset));
    projection = glm::perspective(glm::radians(camera.getFOV()), 
                                  static_cast<float>(windowWidth) / static_cast<float>(windowHeight), 
                                  nearZ, farZ);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    windowWidth = width;
    windowHeight = height;
    projection = glm::perspective(glm::radians(camera.getFOV()), 
                                  static_cast<float>(width) / static_cast<float>(height), 
                                  nearZ, farZ);
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    static bool isMouseCaptured = true;

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    // Toggle mouse capture when Left Alt is pressed and released
    if (key == GLFW_KEY_LEFT_ALT && action == GLFW_PRESS) {
        if (isMouseCaptured) {
            glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
            isMouseCaptured = false;
        }
    } else if (key == GLFW_KEY_LEFT_ALT && action == GLFW_RELEASE) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        isMouseCaptured = true;
    }

    if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
        h += 0.01f;
        std::cout << "h: " << h << std::endl;
    }

    if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
        h -= 0.01f;
        std::cout << "h: " << h << std::endl;
    }

    if (key == GLFW_KEY_3 && action == GLFW_PRESS) {
        selectParticle++;
        std::cout << "Reselecting particle: " << selectParticle << std::endl;
    }

    if (key == GLFW_KEY_4 && action == GLFW_PRESS) {
        selectParticle--;
        std::cout << "Reselecting particle: " << selectParticle << std::endl;
    }
}

void processInput(GLFWwindow* window) {
    // keys for movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.updatePosition(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.updatePosition(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.updatePosition(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.updatePosition(RIGHT, deltaTime);
}

void centerWindow(GLFWwindow* window) {
    // Get primary monitor
    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor) {
        std::cerr << "Failed to get the primary monitor!" << std::endl;
        return;
    }

    // Get monitor work area
    int monitorX, monitorY, monitorWidth, monitorHeight;
    glfwGetMonitorWorkarea(monitor, &monitorX, &monitorY, &monitorWidth, &monitorHeight);

    // Get window size
    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    // Calculate the position to center the window
    int windowPosX = monitorX + (monitorWidth - windowWidth) / 2;
    int windowPosY = monitorY + (monitorHeight - windowHeight) / 2;

    // Set window position
    glfwSetWindowPos(window, windowPosX, windowPosY);
}
////////////////////////////////////////////////////////////////////////////////




void initNeighbourData(int N, const int max_neighbours) {
    neighbourArray.resize(N * max_neighbours, 0);
    neighbourCount.resize(N, 0);
    hashToFirstParticleIndex.resize(TABLE_SIZE, UINT32_MAX);
}

void fillParticles(int N, float spacing, const glm::vec3& init_box_min, const glm::vec3& init_box_max) {
    particles.clear();

    std::cout << "Creating " << N << " particles..." << std::endl;

    for (float x = init_box_min.x; x <= init_box_max.x; x += spacing) {
        for (float y = init_box_min.y; y <= init_box_max.y; y += spacing) {
            for (float z = init_box_min.z; z <= init_box_max.z; z += spacing) {
                Particle p;
                p.position = glm::vec3(x, y, z);
                p.velocity = glm::zero<glm::vec3>();
                p.mass = 1.0f;
                p.color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f); // blue
                p.size = 100.0f; // width in pixels for vertex shader
                particles.push_back(p);
            }
        }
    }

    std::cout << "Created " << particles.size() << " particles." << std::endl;
    if (particles.size() > N) {
        std::cerr << "WARNING: particles have been created outside the intended initialization box." << std::endl;
    }
}


int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW." << std::endl;
        return -1;
    }

    // Request OpenGL 3.3 core profile (to avoid manually enabling GL_POINT_SPRITE)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create window and OpenGL context
    GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "Position-Based Fluids", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window." << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window); // Make the context of window the main context on the current thread
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // a callback for window resizing
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // capture mouse

    centerWindow(window);

    // Load OpenGL functions
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD." << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // Tell OpenGL to render points as many fragments instead of a single pixel
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);

    // Create and load shaders
    Shader shader("src/shaders/vertexShader.glsl", "src/shaders/fragShader.glsl");
    Shader planeShader("src/shaders/planeVertexShader.glsl", "src/shaders/planeFragShader.glsl");
    planeShader.useProgram();
    planeShader.setUniformVec3("color1", glm::vec3(0.8f, 0.8f, 0.8f)); // Light gray
    planeShader.setUniformVec3("color2", glm::vec3(0.5f, 0.5f, 0.5f)); // Dark gray
    planeShader.setUniformFloat("scale", 1.0f); // Adjust for checker size (currently smaller makes larger squares)


    ////////////////////////////// Particles ///////////////////////////////////
    fillParticles(num_particles, particleSpacing, init_box_min, init_box_max);
    initNeighbourData(num_particles, MAX_NEIGHBOURS);

    GLuint particleVAO, particleVBO;

    glGenVertexArrays(1, &particleVAO);
    glGenBuffers(1, &particleVBO);

    glBindVertexArray(particleVAO);
    glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
    glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Particle), &particles[0], GL_DYNAMIC_DRAW);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR) {
        std::cerr << "Error uploading buffer data: " << error << std::endl;
    }

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribDivisor(0, 1);

    // Size attribute
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(offsetof(Particle, size)));
    glEnableVertexAttribArray(1);
    glVertexAttribDivisor(1, 1);
    
    // Color attribute
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)(offsetof(Particle, color)));
    glEnableVertexAttribArray(2);
    glVertexAttribDivisor(2, 1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    ////////////////////////////////////////////////////////////////////////////


    ///////////////////////////// REFERENCE PLANE //////////////////////////////
    // Plane vertices (large quad centered at origin on the xz-plane)
    float halfWidth = 5.0f; //50.0f;
    float planeVertices[] = {
        // Positions            // Texture Coords
        -halfWidth, -5.05f, -halfWidth,        0.0f,      0.0f,
         halfWidth, -5.05f, -halfWidth,   halfWidth,      0.0f,
         halfWidth, -5.05f,  halfWidth,   halfWidth, halfWidth,

        -halfWidth, -5.05f, -halfWidth,        0.0f,      0.0f,
         halfWidth, -5.05f,  halfWidth,   halfWidth, halfWidth,
        -halfWidth, -5.05f,  halfWidth,        0.0f, halfWidth
    };

    GLuint planeVAO, planeVBO;

    glGenVertexArrays(1, &planeVAO);
    glGenBuffers(1, &planeVBO);

    glBindVertexArray(planeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, planeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), planeVertices, GL_STATIC_DRAW);

    // Position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    ////////////////////////////////////////////////////////////////////////////


    ////////////////////////////  Neighbour Data  //////////////////////////////
    GLuint neighbourArrayVBO, neighbourCountVBO, hashTableVBO;
    glGenBuffers(1, &neighbourArrayVBO);
    glGenBuffers(1, &neighbourCountVBO);
    glGenBuffers(1, &hashTableVBO);

    glBindBuffer(GL_ARRAY_BUFFER, neighbourArrayVBO);
    glBufferData(GL_ARRAY_BUFFER, neighbourArray.size() * sizeof(uint32_t), &neighbourArray[0], GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, neighbourCountVBO);
    glBufferData(GL_ARRAY_BUFFER, neighbourCount.size() * sizeof(uint32_t), &neighbourCount[0], GL_DYNAMIC_DRAW);

    glBindBuffer(GL_ARRAY_BUFFER, hashTableVBO);
    glBufferData(GL_ARRAY_BUFFER, hashToFirstParticleIndex.size() * sizeof(uint32_t), &hashToFirstParticleIndex[0], GL_DYNAMIC_DRAW);
    ////////////////////////////////////////////////////////////////////////////


    //////////////////////////// CUDA-OpenGL INTEROP ///////////////////////////
    cudaGraphicsResource* particlesResource = nullptr;
    cudaGraphicsResource* neighbourArrayResource = nullptr;
    cudaGraphicsResource* neighbourCountResource = nullptr;
    cudaGraphicsResource* hashTableResource = nullptr;
    cudaGraphicsGLRegisterBuffer(&particlesResource, particleVBO, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsGLRegisterBuffer(&neighbourArrayResource, neighbourArrayVBO, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsGLRegisterBuffer(&neighbourCountResource, neighbourCountVBO, cudaGraphicsRegisterFlagsNone);
    cudaGraphicsGLRegisterBuffer(&hashTableResource, hashTableVBO, cudaGraphicsRegisterFlagsNone);
    ////////////////////////////////////////////////////////////////////////////

    bool activate = false;
    bool taskComplete = false;

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        float timeAtCurrentFrame = static_cast<float>(glfwGetTime());
        deltaTime = timeAtCurrentFrame - timeAtLastFrame;
        timeAtLastFrame = timeAtCurrentFrame;

        processInput(window);

        ////////////////////////// COMPUTATIONS ////////////////////////////////
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            taskComplete = false;
            activate = false;
        }
        if (!activate && glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
            activate = true;
        }
        if (!taskComplete && activate) {
            std::cout << "Running neighbour search once" << std::endl;
            testCUDAKernels(particlesResource, num_particles, h, neighbourArrayResource, neighbourCountResource, hashTableResource, TABLE_SIZE, MAX_NEIGHBOURS);
            computeDensitiesCUDA(particlesResource, num_particles, h, neighbourArrayResource, neighbourCountResource, MAX_NEIGHBOURS);

            glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
            glGetBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(Particle), &particles[0]);
            glBindBuffer(GL_ARRAY_BUFFER, neighbourArrayVBO);
            glGetBufferSubData(GL_ARRAY_BUFFER, 0, neighbourArray.size() * sizeof(uint32_t), &neighbourArray[0]);
            glBindBuffer(GL_ARRAY_BUFFER, neighbourCountVBO);
            glGetBufferSubData(GL_ARRAY_BUFFER, 0, neighbourCount.size() * sizeof(uint32_t), &neighbourCount[0]);
            glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind the buffer

            particles[selectParticle].color = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
            int start = selectParticle * MAX_NEIGHBOURS;
            for (int offset = 0; offset < neighbourCount[selectParticle]; offset++) {
                int j = neighbourArray[start + offset];
                particles[j].color = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); 
            }

            std::cout << "Particle " << selectParticle << " neighbour count: " << neighbourCount[selectParticle] << std::endl;

            // Re-upload updated particle data to the VBO
            glBindBuffer(GL_ARRAY_BUFFER, particleVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(Particle), &particles[0]);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            
            taskComplete = true;
            std::cout << "All good" << std::endl;
            
            std::cout << "Density of particle " << selectParticle << ": " << particles[selectParticle].density << std::endl;

            std::cout << "A few other densities:" << std::endl;
            for (int k = 0; k < 10; k++) {
                std::cout << particles[k].density << std::endl;    
            }
        }


        ///////////////////////////// RENDERING ////////////////////////////////
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // MVP
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);

        // RENDER REFERENCE PLANE
        planeShader.useProgram();
        planeShader.setUniformMat4("projection", projection);
        planeShader.setUniformMat4("view", view);
        planeShader.setUniformMat4("model", model);

        glBindVertexArray(planeVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);

        // RENDER PARTICLES
        shader.useProgram();
        shader.setUniformMat4("projection", projection);
        shader.setUniformMat4("view", view);
        shader.setUniformMat4("model", model);
        shader.setUniformVec3("cameraPosition", camera.getPosition());

        glBindVertexArray(particleVAO);
        glDrawArraysInstanced(GL_POINTS, 0, 1, particles.size());
        glBindVertexArray(0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    glDeleteVertexArrays(1, &particleVAO);
    glDeleteBuffers(1, &particleVBO);
    glDeleteProgram(shader.programID); // need better way to clean up shader
    glDeleteProgram(planeShader.programID);
    if (particlesResource) {
        cudaGraphicsUnregisterResource(particlesResource);
    }

    glfwTerminate();
    return 0;
}