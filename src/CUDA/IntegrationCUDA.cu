#include <AquaForge/CUDA/IntegrationCUDA.h>
#include <AquaForge/Simulation/Particle.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

namespace AquaForge {

namespace IntegrationCUDA {

// =================================== CUDA ====================================

__global__ void kernel_predict_position(Particle* particles, int numParticles, float dt, glm::vec3 force) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;

    Particle* p = &particles[i];

    // The simplest Euler integration
    p->new_velocity = p->velocity + force * dt;
    p->new_position = p->position + p->new_velocity * dt;
}

__global__ void kernel_update_position_and_velocity(Particle* sortedParticles, int numParticles, float dt) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;

    Particle* p = &sortedParticles[i];

    // Update velocity using the corrected position: v_i = (new_position - position) / dt
    p->new_velocity = (p->new_position - p->position) / dt;

    p->position = p->new_position;
    p->velocity = p->new_velocity;
}

// ================================= Wrappers ==================================

void predictNewPositionCUDA(cudaGraphicsResource* particlesResource, int numParticles, glm::vec3 force, float deltaTime) {
    // Map resource data to CUDA, retrieve dvice pointer
    cudaGraphicsMapResources(1, &particlesResource, 0);
    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    // Make initial prediction
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    kernel_predict_position<<<gridSize, blockSize>>>(d_particles, numParticles, deltaTime, force);
    cudaDeviceSynchronize();

    // Let OpenGL regain control over the resource
    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}

void updatePositionAndVelocityCUDA(cudaGraphicsResource* particlesResource, int numParticles, float deltaTime) {
    // Map resource data to CUDA, retrieve dvice pointer
    cudaGraphicsMapResources(1, &particlesResource, 0);
    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    // Update final position (after constraint solver)
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    kernel_update_position_and_velocity<<<gridSize, blockSize>>>(d_particles, numParticles, deltaTime);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}

} // namespace IntegrationCUDA

} // namespace AquaForge