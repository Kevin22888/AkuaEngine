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
const int NUM_PARTICLES = 35937; //4913;
float particle_spacing = 0.625f;

const int MAX_NEIGHBOURS = 512;

std::vector<Particle> particles;

// Physics
glm::vec3 gravity(0.0f, -9.8f, 0.0f);
float rest_density = 0.5f;

// SPH
float smoothRadius = 2.0f * particle_spacing;

float relaxation = 2000.0f;

// Container
glm::vec3 box_min(-22.0f, -15.0f, -22.0f);
glm::vec3 box_max(22.0f, 22.0f, 22.0f);
float init_box_half_width = 10.0f;
glm::vec3 shift(-11.0f, 0.0f, 0.0f);
glm::vec3 init_box_min = glm::vec3(-init_box_half_width, -init_box_half_width, -init_box_half_width) + shift;
glm::vec3 init_box_max = glm::vec3(init_box_half_width, init_box_half_width, init_box_half_width) + shift;

// Neighbour search
const int TABLE_SIZE = 512 * NUM_PARTICLES; 
uint32_t neighbourArray[NUM_PARTICLES * MAX_NEIGHBOURS];
uint32_t neighbourCount[NUM_PARTICLES];

int solverIterations = 2;

////////////////////////////////////////////////////////////////////////////////


///////////////////////////// Window configurations ////////////////////////////
int windowWidth = 800;
int windowHeight = 600;

// Camera configurations
Camera camera(glm::vec3(30.0f, 20.0f, 40.0f));
bool firstMouseInput = true; // prevent jump when mouse is first captured
float lastX = windowWidth / 2.0f;
float lastY = windowHeight / 2.0f;
float deltaTime = 0.0f;
float timeAtLastFrame = 0.0f;

// Projection Matrix in MVP (re-calculated when mouse scroll or window is resized)
float nearZ = 0.1f;
float farZ = 160.0f;
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

void processInput(GLFWwindow* window) {
    // if the escape key is pressed, let GLFW know we are closing
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, true);
    }

    // keys for movement
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.updatePosition(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.updatePosition(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.updatePosition(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.updatePosition(RIGHT, deltaTime);
    
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
        rest_density += 0.01f;
        std::cout << "Rest density: " << rest_density << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
        rest_density -= 0.01f;
        std::cout << "Rest density: " << rest_density << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
        smoothRadius += 0.001f;
        std::cout << "Smoothing radius: " << smoothRadius << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
        smoothRadius -= 0.001f;
        std::cout << "Smoothing radius: " << smoothRadius << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
        relaxation += 0.01f;
        std::cout << "Relaxation: " << relaxation << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
        relaxation -= 0.01f;
        std::cout << "Relaxation: " << relaxation << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS) {
        gravity.y -= 0.1f;
        std::cout << "Gravity: " << gravity.y << std::endl;
    }
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
        gravity.y += 0.1f;
        std::cout << "Gravity: " << gravity.y << std::endl;
    }
}
////////////////////////////////////////////////////////////////////////////////

/*
void fillParticles(int totalParticles, float spacing, const glm::vec3& box_min, const glm::vec3& box_max) {
    particles.clear();

    // Calculate the dimensions of the bounding box
    glm::vec3 box_size = box_max - box_min;

    // Check if spacing is valid
    if (spacing <= 0.0f) {
        std::cerr << "Error: Spacing must be greater than zero.\n";
        return;
    }

    // Calculate the maximum number of particles per dimension based on spacing
    int maxParticlesX = static_cast<int>(box_size.x / spacing);
    int maxParticlesY = static_cast<int>(box_size.y / spacing);
    int maxParticlesZ = static_cast<int>(box_size.z / spacing);

    if (maxParticlesX <= 0 || maxParticlesY <= 0 || maxParticlesZ <= 0) {
        std::cerr << "Error: Bounding box is too small to fit any particles with the given spacing.\n";
        return;
    }

    // Attempt to create particles
    int createdParticles = 0;
    for (int x = 0; x < maxParticlesX && createdParticles < totalParticles; ++x) {
        for (int y = 0; y < maxParticlesY && createdParticles < totalParticles; ++y) {
            for (int z = 0; z < maxParticlesZ && createdParticles < totalParticles; ++z) {
                glm::vec3 position = box_min + glm::vec3(x * spacing, y * spacing, z * spacing);

                // Check if the position is within bounds (floating point safety)
                if (position.x > box_max.x || position.y > box_max.y || position.z > box_max.z) {
                    continue;
                }

                Particle p;
                p.position = position;
                p.size = 100.0f; // Uniform size
                p.color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f); // Blue
                p.velocity = glm::vec3(0.0f, 0.0f, 0.0f);
                p.mass = 1.0f;

                particles.push_back(p);
                ++createdParticles;
            }
        }
    }

    // Output the result
    std::cout << "Created " << createdParticles << " particles.\n";

    // Check if we couldn't create all requested particles
    if (createdParticles < totalParticles) {
        std::cout << "Could not create " << (totalParticles - createdParticles)
                  << " particles due to bounding box constraints.\n";
    }
}

void initParticles(int numParticlesPerDimension, float spacing) {  // CHECK THE MATH ON THIS
    particles.clear();

    // Calculate the effective box size based on the given number of particles and spacing
    const float totalSize = (numParticlesPerDimension - 1) * spacing;

    // Ensure the particles are confined within the range [-5, 5]
    const glm::vec3 offset(-5.0f, -5.0f, -5.0f); // Start at the lower bound of the box

    if (totalSize > 10.0f) {
        std::cerr << "Error: Particles exceed bounding box [-5, 5]^3 with the current spacing.\n";
        return;
    }

    for (int x = 0; x < numParticlesPerDimension; ++x) {
        for (int y = 0; y < numParticlesPerDimension; ++y) {
            for (int z = 0; z < numParticlesPerDimension; ++z) {
                Particle p;
                p.position = glm::vec3(x * spacing, y * spacing, z * spacing) + offset;
                p.size = 100.0f; // Uniform size
                p.color = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f); // Blue

                p.velocity = glm::vec3(0.0f, 0.0f, 0.0f);
                p.mass = 1.0f;

                particles.push_back(p);
            }
        }
    }

    std::cout << "Initialized " << particles.size() << " particles in a lattice layout." << std::endl;

    // for (const auto& p : particles) {
    // std::cout << "Position: (" << p.position.x << ", " << p.position.y << ", " << p.position.z
    //           << ") | Size: " << p.size
    //           << " | Color: (" << p.color.r << ", " << p.color.g << ", " << p.color.b << ")" 
    //           << std::endl;
    // }
}
*/

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
                p.size = 2000.0f; // width in pixels for vertex shader
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
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // capture mouse

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

    //initParticles(20, spacing);
    fillParticles(NUM_PARTICLES, particle_spacing, init_box_min, init_box_max);

    GLuint VAO, VBO;

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Particle), &particles[0], GL_STATIC_DRAW);
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

    // REFERENCE PLANE //
    // Plane vertices (large quad centered at origin on the xz-plane)
    float height = -15.5f;
    float planeVertices[] = {
        // Positions            // Texture Coords
        -80.0f, height, -80.0f,    0.0f,  0.0f,
         80.0f, height, -80.0f,   20.0f,  0.0f,
         80.0f, height,  80.0f,   20.0f, 20.0f,

        -80.0f, height, -80.0f,    0.0f,  0.0f,
         80.0f, height,  80.0f,   20.0f, 20.0f,
        -80.0f, height,  80.0f,    0.0f, 20.0f
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
    /////////////////////

    /// CUDA-OpenGL INTEROP ///
    cudaGraphicsResource* particlesResource = nullptr;

    cudaGraphicsGLRegisterBuffer(&particlesResource, VBO, cudaGraphicsRegisterFlagsNone);
    std::cout << "VBO registered with CUDA." << std::endl;

    std::cout << "particles.size() = " << particles.size() << std::endl;

    const float fixedDeltaTime = 0.02f;  // Fixed time step for integration
    const float renderDeltaTime = 1.0f/30.0f; //0.016f;  // Target time for rendering (60 FPS)
    float accumulatedTime = 0.0f;
    float accumulatedRenderTime = 0.0f;

    bool activate = false;
    // Render loop
    while (!glfwWindowShouldClose(window)) {
        float timeAtCurrentFrame = static_cast<float>(glfwGetTime());
        deltaTime = timeAtCurrentFrame - timeAtLastFrame;
        timeAtLastFrame = timeAtCurrentFrame;

        //accumulatedTime += deltaTime;
        //accumulatedRenderTime += deltaTime;

        //std::cout << "dt = " << deltaTime << " s" << std::endl;

        processInput(window);

        ////////////////////////// COMPUTATIONS ////////////////////////////////

        //while (accumulatedTime >= fixedDeltaTime) {
            if (!activate && glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
                activate = true;
            }
            if (activate && glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS) {
                activate = false;
            }

            if (activate) {
                predictNewPositionCUDA(particlesResource, NUM_PARTICLES, gravity, deltaTime);

                findParticleNeighboursCUDA(particlesResource, NUM_PARTICLES, smoothRadius, neighbourArray, neighbourCount, TABLE_SIZE, MAX_NEIGHBOURS);
                runConstraintSolverCUDA(particlesResource, NUM_PARTICLES, smoothRadius, solverIterations, neighbourArray, neighbourCount, 
                                        MAX_NEIGHBOURS, rest_density, relaxation, box_min, box_max);

                updatePositionAndVelocityCUDA(particlesResource, NUM_PARTICLES, deltaTime);
            }

            //accumulatedTime -= fixedDeltaTime;
        //}
        ///////////////////////////// RENDERING ////////////////////////////////

        //if (accumulatedRenderTime >= renderDeltaTime) {
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

            glBindVertexArray(VAO);
            glDrawArraysInstanced(GL_POINTS, 0, 1, particles.size());
            glBindVertexArray(0);

            glfwSwapBuffers(window);
            //accumulatedRenderTime -= renderDeltaTime;
        //}
        glfwPollEvents();
    }

    // Clean up
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shader.programID); // need better way to clean up shader
    glDeleteProgram(planeShader.programID);
    if (particlesResource) {
        cudaGraphicsUnregisterResource(particlesResource);
        //std::cout << "VBO unregistered." << std::endl;
    }

    glfwTerminate();
    return 0;
}