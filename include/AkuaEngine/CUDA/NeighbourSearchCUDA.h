#ifndef AKUAENGINE_NEIGHBOUR_SEARCH_H
#define AKUAENGINE_NEIGHBOUR_SEARCH_H

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <cstdint>

namespace AkuaEngine {

namespace NeighbourSearchCUDA {

void findParticleNeighboursCUDA(
    cudaGraphicsResource* particlesResource,
    int numParticles,
    uint32_t* neighbourArray,
    uint32_t* neighbourCount,
    float smoothRadius,
    float cellSize,
    const int tableSize,
    const int maxNeighbours
);

} // namespace NeighbourSearchCUDA

} // namespace AkuaEngine

#endif // AKUAENGINE_NEIGHBOUR_SEARCH_H