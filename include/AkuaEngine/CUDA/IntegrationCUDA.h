#ifndef AKUAENGINE_INTEGRATION_H
#define AKUAENGINE_INTEGRATION_H

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

namespace AkuaEngine {

namespace IntegrationCUDA {

void predictNewPositionCUDA(cudaGraphicsResource* particlesResource, int numParticles, glm::vec3 force, float deltaTime);
void updatePositionAndVelocityCUDA(cudaGraphicsResource* particlesResource, int numParticles, float deltaTime);
void applyVelocityAdjustmentsCUDA(
    cudaGraphicsResource* particlesResource, 
    int numParticles,
    uint32_t* neighbourArray, 
    uint32_t* neighbourCount,
    float smoothRadius,
    float deltaTime,
    float vorticityEpsilon,
    float viscosity,
    const int maxNeighbours
);

} // namespace IntegrationCUDA

} // namespace AkuaEngine

#endif // AKUAENGINE_INTEGRATION_H