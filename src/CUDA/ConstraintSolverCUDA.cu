#include <AkuaEngine/CUDA/ConstraintSolverCUDA.h>
#include <AkuaEngine/CUDA/SmoothingKernelsCUDA.h>
#include <AkuaEngine/Simulation/Particle.h>
#include <AkuaEngine/Simulation/PBFConfig.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <glm/glm.hpp>

namespace AkuaEngine {

namespace ConstraintSolverCUDA {

// =================================== CUDA ====================================

__global__ void kernel_calculate_densities(
    Particle* sortedParticles, 
    int numParticles, 
    uint32_t* neighbourArray, 
    uint32_t* neighbourCount, 
    float smoothRadius, 
    const int maxNeighbours
) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;

    Particle* p_i = &sortedParticles[i];

    // This particle itself contributes to the density too
    p_i->density = p_i->mass * SmoothingKernels::device_poly6(0, smoothRadius);

    // The rest of the SPH density estimation
    int start = i * maxNeighbours;
    for (int offset = 0; offset < neighbourCount[i]; offset++) {
        uint32_t j = neighbourArray[start + offset];
        Particle* p_j = &sortedParticles[j];

        glm::vec3 separation = p_i->new_position - p_j->new_position;
        float distanceSquared = glm::dot(separation, separation);
        p_i->density += p_j->mass * SmoothingKernels::device_poly6(distanceSquared, smoothRadius);
    }
}

/*
Caclulating the scaling factor, lambda, which is used to find a position correction term for the given particle.
This involves computing the gradient of the constraint function, which has different cases. And after that, we 
need to sum the squared magnitudes of the gradients. For readability, I've separated the computation steps into 
two loops instead of one compact loop. 
As mentioned, the equations can be found here: https://mmacklin.com/pbf_sig_preprint.pdf.
*/
__global__ void kernel_calculate_lambdas(
    Particle* sortedParticles, 
    int numParticles,
    uint32_t* neighbourArray, 
    uint32_t* neighbourCount,
    float smoothRadius, 
    const int maxNeighbours, 
    const float inverseRestDensity,
    const float relaxation
) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;

    Particle* p_i = &sortedParticles[i];

    // Gradient of constraint function with respect to this particle itself.
    glm::vec3 grad_pi_Ci(0.0f);

    int start = i * maxNeighbours;
    for (int offset = 0; offset < neighbourCount[i]; offset++) {
        uint32_t j = neighbourArray[start + offset];
        Particle* p_j = &sortedParticles[j];

        glm::vec3 separation = p_i->new_position - p_j->new_position;
        grad_pi_Ci += p_j->mass * SmoothingKernels::device_gradient_spiky(separation, smoothRadius);
    }
    grad_pi_Ci *= inverseRestDensity;

    // Gradient of constraint function with respect to each neighbour particle.
    // Calculating the summation directly.
    float sum = 0;

    for (int offset = 0; offset < neighbourCount[i]; offset++) {
        uint32_t j = neighbourArray[start + offset];
        Particle* p_j = &sortedParticles[j];

        glm::vec3 separation = p_i->new_position - p_j->new_position;
        glm::vec3 grad_pj_Ci = - inverseRestDensity * p_j->mass * SmoothingKernels::device_gradient_spiky(separation, smoothRadius);
        float magnitudeSquared = glm::dot(grad_pj_Ci, grad_pj_Ci);
        sum += magnitudeSquared;
    }

    float C_i = p_i->density * inverseRestDensity - 1;

    // Relaxation softens the constraint and reduces instability when particles are too close.
    p_i->lambda = - C_i / (sum + glm::dot(grad_pi_Ci, grad_pi_Ci) + relaxation);
}

__global__ void kernel_calculate_position_delta(
    Particle* sortedParticles, 
    int numParticles,
    uint32_t* neighbourArray, 
    uint32_t* neighbourCount,
    float smoothRadius, 
    const int maxNeighbours, 
    const float inverseRestDensity,
    LambdaCorrParams corrParams
) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;

    Particle* p_i = &sortedParticles[i];

    glm::vec3 position_delta(0.0f);

    int start = i * maxNeighbours;
    for (int offset = 0; offset < neighbourCount[i]; offset++) {
        uint32_t j = neighbourArray[start + offset];
        Particle* p_j = &sortedParticles[j];

        glm::vec3 separation = p_i->new_position - p_j->new_position;
        float distanceSquared = glm::dot(separation, separation);
        float deltaQSquared = corrParams.delta_q * corrParams.delta_q;
        float lambda_corr = - corrParams.k * powf(SmoothingKernels::device_poly6(distanceSquared, smoothRadius) / SmoothingKernels::device_poly6(deltaQSquared, smoothRadius), corrParams.n);

        position_delta += (p_i->lambda + p_j->lambda + lambda_corr) * p_j->mass * SmoothingKernels::device_gradient_spiky(separation, smoothRadius);
    }

    p_i->position_delta = position_delta * inverseRestDensity;
}

/*
This is just here to temporarily act as a collision detection.
Later we will use virtual particles.
*/
__device__ void handle_particle_collision(Particle* p, glm::vec3 boxMin, glm::vec3 boxMax) {
    float minDist = 0.02f;
    float stiffness = 0.5f;
     glm::vec3 correction(0.0f);

    if (p->new_position.x < boxMin.x + minDist)
        correction.x += stiffness * (boxMin.x + minDist - p->new_position.x);
    if (p->new_position.x > boxMax.x - minDist)
        correction.x += stiffness * (boxMax.x - minDist - p->new_position.x);

    if (p->new_position.y < boxMin.y + minDist)
        correction.y += stiffness * (boxMin.y + minDist - p->new_position.y);
    if (p->new_position.y > boxMax.y - minDist)
        correction.y += stiffness * (boxMax.y - minDist - p->new_position.y);

    if (p->new_position.z < boxMin.z + minDist)
        correction.z += stiffness * (boxMin.z + minDist - p->new_position.z);
    if (p->new_position.z > boxMax.z - minDist)
        correction.z += stiffness * (boxMax.z - minDist - p->new_position.z);

    p->new_position += correction;
}

__global__ void kernel_correct_position(Particle* sortedParticles, int numParticles, glm::vec3 boxMin, glm::vec3 boxMax) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;
    
    Particle* p = &sortedParticles[i];

    // Apply correction to the predicted position
    p->new_position += p->position_delta;
    
    handle_particle_collision(p, boxMin, boxMax);
}

// ================================= Wrappers ==================================

void runConstraintSolverCUDA(
    cudaGraphicsResource* particlesResource, 
    int numParticles, 
    int solverIterations, 
    uint32_t* neighbourArray, 
    uint32_t* neighbourCount, 
    float smoothRadius, 
    const int maxNeighbours, 
    const float restDensity, 
    const float relaxation, 
    glm::vec3 boxMin, 
    glm::vec3 boxMax,
    LambdaCorrParams corrParams
) {
    // Map the relevant graphics resource to CUDA and retrieve a device pointer for that resource
    cudaGraphicsMapResources(1, &particlesResource, 0);
    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    // For neighbour data, let thrust take care of the memory copying task, we can get the device vector easily.
    // These vectors will be freed automatically when they go out of scope.
    thrust::device_vector<uint32_t> d_neighbourArrayVector(neighbourArray, neighbourArray + numParticles * maxNeighbours);
    thrust::device_vector<uint32_t> d_neighbourCountVector(neighbourCount, neighbourCount + numParticles);
    uint32_t* d_neighbourArray = thrust::raw_pointer_cast(d_neighbourArrayVector.data());
    uint32_t* d_neighbourCount = thrust::raw_pointer_cast(d_neighbourCountVector.data());

    // Prepare to launch CUDA kernels
    float inverseRestDensity = 1.0f / restDensity;

    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;

    while (solverIterations-- > 0) {
        kernel_calculate_densities<<<gridSize, blockSize>>>(d_particles, numParticles, d_neighbourArray, d_neighbourCount, smoothRadius, maxNeighbours);
        cudaDeviceSynchronize();

        kernel_calculate_lambdas<<<gridSize, blockSize>>>(d_particles, numParticles, d_neighbourArray, d_neighbourCount, smoothRadius, maxNeighbours, inverseRestDensity, relaxation);
        cudaDeviceSynchronize();

        kernel_calculate_position_delta<<<gridSize, blockSize>>>(d_particles, numParticles, d_neighbourArray, d_neighbourCount, smoothRadius, maxNeighbours, inverseRestDensity, corrParams);
        cudaDeviceSynchronize();

        kernel_correct_position<<<gridSize, blockSize>>>(d_particles, numParticles, boxMin, boxMax);
        cudaDeviceSynchronize();
    }

    // Particle data stay in the VRAM as they need to be rendered later. We just need to unmap this resource to let OpenGL regain access.
    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}

} // namespace ConstraintSolverCUDA

} // namespace AkuaEngine
