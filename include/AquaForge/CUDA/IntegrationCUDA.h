#ifndef AQUAFORGE_INTEGRATION_H
#define AQUAFORGE_INTEGRATION_H

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

namespace AquaForge {

namespace IntegrationCUDA {

void predictNewPositionCUDA(cudaGraphicsResource* particlesResource, int numParticles, glm::vec3 force, float deltaTime);
void updatePositionAndVelocityCUDA(cudaGraphicsResource* particlesResource, int numParticles, float deltaTime);

} // namespace IntegrationCUDA

} // namespace AquaForge

#endif // AQUAFORGE_INTEGRATION_H