#ifndef AKUAENGINE_PBF_SOLVER_H
#define AKUAENGINE_PBF_SOLVER_H

#include <AkuaEngine/Simulation/PBFConfig.h>
#include <AkuaEngine/Interop/InteropResource.h>
#include <glm/glm.hpp>
#include <vector>
#include <cstdint>

namespace AkuaEngine {

class PBFSolver {
public:
    PBFSolver(int numParticles, const PBFConfig& config, const LambdaCorrParams& corrParams);

    // Calls wrapper functions that launch CUDA kernels
    void step(const InteropResource& particleInterop, float deltaTime, glm::vec3 boxMin, glm::vec3 boxMax);
    void setGravity(glm::vec3 gravity);

private:
    int _numParticles;
    int _tableSize;
    std::vector<uint32_t> _neighbourArray;
    std::vector<uint32_t> _neighbourCount;
    PBFConfig _config;
    LambdaCorrParams _corrParams;
};

inline void PBFSolver::setGravity(glm::vec3 gravity) {
    _config.gravity = gravity;
}

} // namespace AkuaEngine

#endif // AKUAENGINE_PBF_SOLVER_H