#include <AkuaEngine/Simulation/PBFSolver.h>
#include <AkuaEngine/Simulation/PBFConfig.h>
#include <AkuaEngine/Interop/InteropResource.h>
#include <AkuaEngine/CUDA/IntegrationCUDA.h>
#include <AkuaEngine/CUDA/NeighbourSearchCUDA.h>
#include <AkuaEngine/CUDA/ConstraintSolverCUDA.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <cstdint>

namespace AkuaEngine {

PBFSolver::PBFSolver(int numParticles, const PBFConfig& config, const LambdaCorrParams& corrParams) 
    : _numParticles(numParticles),
      _tableSize(config.maxNeighbours * numParticles), // table size scales with max neighbours
      _neighbourArray(numParticles * config.maxNeighbours),
      _neighbourCount(numParticles),
      _config(config),
      _corrParams(corrParams) 
{}

void PBFSolver::step(const InteropResource& particleInterop, float deltaTime, glm::vec3 boxMin, glm::vec3 boxMax) {
    using namespace IntegrationCUDA;
    using namespace NeighbourSearchCUDA;
    using namespace ConstraintSolverCUDA;
    
    cudaGraphicsResource* particlesResource = particleInterop.getGraphicsResource();

    // The PBF algorithm: 1. Predict positions
    predictNewPositionCUDA(particlesResource, _numParticles, _config.gravity, deltaTime);

    // 2. Find neighbours
    findParticleNeighboursCUDA(
        particlesResource, 
        _numParticles, 
        _neighbourArray.data(), 
        _neighbourCount.data(),
        _config.smoothRadius,
        _config.spatialHashCellSize,
        _tableSize,
        _config.maxNeighbours
    );

    // 3. Iterations of density constraint solving (position corrections)
    runConstraintSolverCUDA(
        particlesResource,
        _numParticles,
        _config.solverIterations,
        _neighbourArray.data(),
        _neighbourCount.data(),
        _config.smoothRadius,
        _config.maxNeighbours,
        _config.restDensity,
        _config.relaxation,
        boxMin,
        boxMax,
        _corrParams
    );

    // 4. True update to the particle states
    updatePositionAndVelocityCUDA(particlesResource, _numParticles, deltaTime);

    // 5. Vorticity Confinement
    applyVelocityAdjustmentsCUDA(
        particlesResource, 
        _numParticles, 
        _neighbourArray.data(), 
        _neighbourCount.data(), 
        _config.smoothRadius,
        deltaTime,
        _config.vorticityEpsilon,
        _config.viscosity,
        _config.maxNeighbours
    );
}

} // namespace AkuaEngine