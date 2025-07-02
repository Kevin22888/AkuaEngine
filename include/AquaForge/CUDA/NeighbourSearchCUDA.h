#ifndef AQUAFORGE_NEIGHBOUR_SEARCH_H
#define AQUAFORGE_NEIGHBOUR_SEARCH_H

#include <cuda_gl_interop.h>
#include <cstdint>

namespace AquaForge {

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

} // namespace AquaForge

#endif // AQUAFORGE_NEIGHBOUR_SEARCH_H