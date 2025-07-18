#ifndef AKUAENGINE_CONSTRAINT_SOLVER_H
#define AKUAENGINE_CONSTRAINT_SOLVER_H

#include <AkuaEngine/Simulation/PBFConfig.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <cstdint>

namespace AkuaEngine {

namespace ConstraintSolverCUDA {

void runConstraintSolverCUDA(
    cudaGraphicsResource* particlesResource, 
    int numParticles, 
    int solverIterations, 
    uint32_t* neighbourArray, 
    uint32_t* neighbourCount, 
    float smoothRadius, 
    const int maxNeighbours, 
    const float restDensity, 
    const float epsilon, 
    glm::vec3 boxMin, 
    glm::vec3 boxMax,
    LambdaCorrParams corrParams
);

} // namespace ConstraintSolverCUDA

} // namespace AkuaEngine

#endif // AKUAENGINE_CONSTRAINT_SOLVER_H