#include <AkuaEngine/CUDA/IntegrationCUDA.h>
#include <AkuaEngine/CUDA/SmoothingKernelsCUDA.h>
#include <AkuaEngine/CUDA/MathUtilsCUDA.h>
#include <AkuaEngine/Simulation/Particle.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <glm/glm.hpp>

namespace {

// TODO: move this to a larger collision detection module
// and replace glm::vec3 with float3.
struct CollisionPlane {
    glm::vec3 point;
    glm::vec3 normal;
};

}

namespace AkuaEngine {

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

__device__ void resolve_collision(
    glm::vec3& position,
    glm::vec3& velocity,
    const glm::vec3& planePoint,
    const glm::vec3& normal,
    float minDist,
    float restitution,
    float friction
) {
    float distance = glm::dot(position - planePoint, normal);
    float approaching = glm::dot(velocity, normal);

    if (distance < minDist) {
        glm::vec3 v_n = approaching * normal;
        glm::vec3 v_t = velocity - v_n;
        
        if (approaching < 0.0f) {    
            velocity = -restitution * v_n + (1.0f - friction) * v_t;
        } else if (fabs(approaching) < 1e-5f) {
            velocity = (1.0f - friction) * v_t;
        }
    }
}

__global__ void kernel_apply_boundary_velocity_damping(
    Particle* sortedParticles,
    int numParticles,
    glm::vec3 boxMin,
    glm::vec3 boxMax,
    float restitution,
    float friction
) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;

    Particle* p = &sortedParticles[i];

    const float minDist = 0.025f;

    CollisionPlane planes[6] = {
        { glm::vec3(boxMin.x, 0, 0),  glm::vec3( 1,  0,  0) },
        { glm::vec3(boxMax.x, 0, 0),  glm::vec3(-1,  0,  0) },
        { glm::vec3(0, boxMin.y, 0),  glm::vec3( 0,  1,  0) },
        { glm::vec3(0, boxMax.y, 0),  glm::vec3( 0, -1,  0) },
        { glm::vec3(0, 0, boxMin.z),  glm::vec3( 0,  0,  1) },
        { glm::vec3(0, 0, boxMax.z),  glm::vec3( 0,  0, -1) }
    };

    for (const auto& plane : planes) {
        resolve_collision(p->position, p->velocity, plane.point, plane.normal, minDist, restitution, friction);
    }
}

__global__ void kernel_compute_vorticities(
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

    p_i->vorticity = glm::vec3(0.0f);

    int start = i * maxNeighbours;
    for (int offset = 0; offset < neighbourCount[i]; offset++) {
        uint32_t j = neighbourArray[start + offset];
        Particle* p_j = &sortedParticles[j];

        glm::vec3 v_ij = p_j->velocity - p_i->velocity;
        glm::vec3 separation = p_i->new_position - p_j->new_position;
        p_i->vorticity += - p_j->mass * MathUtilsCUDA::cross(v_ij, SmoothingKernels::device_gradient_spiky(separation, smoothRadius));
    }
}

__global__ void kernel_apply_vorticity_confinement(
    Particle* sortedParticles,
    int numParticles,
    uint32_t* neighbourArray,
    uint32_t* neighbourCount,
    int maxNeighbours,
    float smoothRadius,
    float deltaTime,
    float vorticityEpsilon
) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;

    Particle* p_i = &sortedParticles[i];

    float inverseDensity = 1 / p_i->density;
    glm::vec3 eta_i = glm::vec3(0.0f);
    int start = i * maxNeighbours;
    for (int offset = 0; offset < neighbourCount[i]; offset++) {
        uint32_t j = neighbourArray[start + offset];
        Particle* p_j = &sortedParticles[j];

        float vorticityDiff = glm::length(p_i->vorticity) - glm::length(p_j->vorticity);
        glm::vec3 separation = p_i->new_position - p_j->new_position;
        eta_i += p_j->mass * vorticityDiff * SmoothingKernels::device_gradient_spiky(separation, smoothRadius);
    }
    eta_i *= inverseDensity;

    float etaLen = glm::length(eta_i);
    if (etaLen < 1e-5f) return;
    glm::vec3 N = eta_i / etaLen;

    glm::vec3 vorticity_force = vorticityEpsilon * MathUtilsCUDA::cross(N, p_i->vorticity);

    p_i->velocity = p_i->velocity + deltaTime * vorticity_force;
}

__global__ void kernel_apply_xsph_viscosity(
    Particle* sortedParticles,
    int numParticles,
    uint32_t* neighbourArray,
    uint32_t* neighbourCount,
    int maxNeighbours,
    float smoothRadius,
    float xsphCoefficient
) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;

    Particle* p_i = &sortedParticles[i];

    glm::vec3 velocity_delta(0.0f);
    int start = i * maxNeighbours;
    for (int offset = 0; offset < neighbourCount[i]; offset++) {
        uint32_t j = neighbourArray[start + offset];
        Particle* p_j = &sortedParticles[j];

        glm::vec3 v_ij = p_j->velocity - p_i->velocity;
        glm::vec3 separation = p_i->new_position - p_j->new_position;
        float w = SmoothingKernels::device_poly6(glm::dot(separation, separation), smoothRadius);

        velocity_delta += (p_j->mass / p_j->density) * v_ij * w;
    }

    p_i->velocity += xsphCoefficient * velocity_delta;
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

void applyBoundaryVelocityDampingCUDA(
    cudaGraphicsResource* particlesResource,
    int numParticles,
    glm::vec3 boxMin,
    glm::vec3 boxMax,
    float restitution,
    float friction
) {
    cudaGraphicsMapResources(1, &particlesResource, 0);
    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    kernel_apply_boundary_velocity_damping<<<gridSize, blockSize>>>(d_particles, numParticles, boxMin, boxMax, restitution, friction);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}

void applyVorticityAndViscosityCUDA(
    cudaGraphicsResource* particlesResource, 
    int numParticles,
    uint32_t* neighbourArray, 
    uint32_t* neighbourCount,
    float smoothRadius,
    float deltaTime,
    float vorticityEpsilon,
    float viscosity,
    const int maxNeighbours
) {
    cudaGraphicsMapResources(1, &particlesResource, 0);
    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    thrust::device_vector<uint32_t> d_neighbourArrayVector(neighbourArray, neighbourArray + numParticles * maxNeighbours);
    thrust::device_vector<uint32_t> d_neighbourCountVector(neighbourCount, neighbourCount + numParticles);
    uint32_t* d_neighbourArray = thrust::raw_pointer_cast(d_neighbourArrayVector.data());
    uint32_t* d_neighbourCount = thrust::raw_pointer_cast(d_neighbourCountVector.data());

    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    kernel_compute_vorticities<<<gridSize, blockSize>>>(d_particles, numParticles, d_neighbourArray, d_neighbourCount, smoothRadius, maxNeighbours);
    cudaDeviceSynchronize();

    kernel_apply_vorticity_confinement<<<gridSize, blockSize>>>(d_particles, numParticles, d_neighbourArray, d_neighbourCount, maxNeighbours, smoothRadius, deltaTime, vorticityEpsilon);
    cudaDeviceSynchronize();

    kernel_apply_xsph_viscosity<<<gridSize, blockSize>>>(d_particles, numParticles, d_neighbourArray, d_neighbourCount, maxNeighbours, smoothRadius, viscosity);
    cudaDeviceSynchronize();
    
    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}

} // namespace IntegrationCUDA

} // namespace AkuaEngine