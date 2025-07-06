#ifndef AQUAFORGE_PBF_SOLVER_H
#define AQUAFORGE_PBF_SOLVER_H

#include <AquaForge/Simulation/PBFConfig.h>
#include <AquaForge/Interop/InteropResource.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

namespace AquaForge {

class PBFSolver {
public:
    PBFSolver(int numParticles, const PBFConfig& config, const LambdaCorrParams& corrParams);

    // Calls wrapper functions that launch CUDA kernels
    void step(const InteropResource& particleInterop, float deltaTime, glm::vec3 boxMin, glm::vec3 boxMax);

private:
    int _numParticles;
    int _tableSize;
    std::vector<uint32_t> _neighbourArray;
    std::vector<uint32_t> _neighbourCount;
    PBFConfig _config;
    LambdaCorrParams _corrParams;
};

} // namespace AquaForge

#endif // AQUAFORGE_PBF_SOLVER_H