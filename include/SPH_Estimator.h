#ifndef PBF_SPH_ESTIMATOR_H
#define PBF_SPH_ESTIMATOR_H

#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include <Particle.h>

// void testCUDAKernels(cudaGraphicsResource* particlesResource, 
//                      int numParticles, float smoothRadius, 
//                      cudaGraphicsResource* neighbourArrayResource, 
//                      cudaGraphicsResource* neighbourCountResource,
//                      cudaGraphicsResource* hashTableResource,
//                      const int TABLE_SIZE, const int MAX_NEIGHBOURS);

// void computeDensitiesCUDA(cudaGraphicsResource* particlesResource, 
//                           int numParticles, float smoothRadius, 
//                           cudaGraphicsResource* neighbourArrayResource, 
//                           cudaGraphicsResource* neighbourCountResource, 
//                           const int MAX_NEIGHBOURS);


void predictNewPositionCUDA(cudaGraphicsResource* particlesResource, int numParticles, glm::vec3 force, float deltaTime);
void findParticleNeighboursCUDA(cudaGraphicsResource* particlesResource, int numParticles, float smoothRadius, uint32_t* neighbourArray, uint32_t* neighbourCount, const int TABLE_SIZE, const int MAX_NEIGHBOURS);
void runConstraintSolverCUDA(cudaGraphicsResource* particlesResource, int numParticles, float smoothRadius, int solverIterations, uint32_t* neighbourArray, uint32_t* neighbourCount, const int MAX_NEIGHBOURS, const float REST_DENSITY, const float EPSILON, glm::vec3 box_min, glm::vec3 box_max);
void correctPositionCUDA(cudaGraphicsResource* particlesResource, int numParticles, glm::vec3 box_min, glm::vec3 box_max);
void updatePositionAndVelocityCUDA(cudaGraphicsResource* particlesResource, int numParticles, float deltaTime);

#endif // PBF_SPH_ESTIMATOR_H