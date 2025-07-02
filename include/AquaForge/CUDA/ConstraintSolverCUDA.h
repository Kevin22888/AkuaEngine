#ifndef AQUAFORGE_CONSTRAINT_SOLVER_H
#define AQUAFORGE_CONSTRAINT_SOLVER_H

#include <AquaForge/Simulation/PBFConfig.h>
#include <glm/glm.hpp>
#include <cuda_gl_interop.h>
#include <cstdint>

namespace AquaForge {

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

} // namespace AquaForge

#endif // AQUAFORGE_CONSTRAINT_SOLVER_H