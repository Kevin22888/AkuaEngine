#include <AkuaEngine/CUDA/NeighbourSearchCUDA.h>
#include <AkuaEngine/Simulation/Particle.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <glm/glm.hpp>

namespace AkuaEngine {

namespace NeighbourSearchCUDA {

// =================================== CUDA ====================================

__device__ glm::ivec3 device_discretize_position(Particle* particle, float cellSize) {
    return {
        static_cast<int>(floorf(particle->new_position.x / cellSize)),
        static_cast<int>(floorf(particle->new_position.y / cellSize)),
        static_cast<int>(floorf(particle->new_position.z / cellSize))
    };
}

__device__ uint32_t device_get_hash(glm::ivec3 position, const int tableSize) {
    return (static_cast<uint32_t>((position.x * 73856093)) ^ 
            static_cast<uint32_t>((position.y * 19349663)) ^ 
            static_cast<uint32_t>((position.z * 83492791))) % tableSize;
}

/*
Compute a hash for each particle depending on its position and our choice of the 
size of the spatial partitioning cell.

This is Spatial Hashing: assign each particle to a grid cell, then compute a hash for that cell.
The hash for a particle maps its 3D discrete cell coordinates to a 1D table index.
*/
__global__ void kernel_compute_hashes(Particle* particles, int numParticles, float cellSize, const int tableSize) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;
    
    Particle* p = &particles[i];

    glm::ivec3 cell = device_discretize_position(p, cellSize);
    p->hash = device_get_hash(cell, tableSize);;
}

/*
Build a hash table by identifying where each hash bucket starts in the sorted particles array. 
This assumes the particles are sorted by hash!
For each unique hash, the lookup table stores its first occurrence in the sorted particles array.
That is, it maps a hash to the first particle in the segment of particles with that hash.
*/
__global__ void kernel_build_hash_table(Particle* sortedParticles, int numParticles, uint32_t* hashToFirstParticleIndex) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;
    
    uint32_t hash = sortedParticles[i].hash;

    if (i == 0) {
        hashToFirstParticleIndex[hash] = i;
    } else {
        if (hash != sortedParticles[i - 1].hash) {
            hashToFirstParticleIndex[hash] = i;
        }
    }
}

/*
For a given particle, find the cell it belongs to, then find all the neighbouring cells (a total of 3x3x3 cells).
For each cell, compute the hash and look up those particles using the hash table. 
Each of those particles is a neighbouring candidate.
*/
__global__ void kernel_find_neighbours(
    Particle* sortedParticles,
    int numParticles,
    uint32_t* neighbourArray,
    uint32_t* neighbourCount,
    uint32_t* hashToFirstParticleIndex,
    float cellSize,
    float smoothRadius,
    const int tableSize,
    const int maxNeighbours
) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= numParticles) return;

    Particle* p_i = &sortedParticles[i];

    // Identify the particle's cell and precompute squared smoothing radius
    glm::ivec3 cell = device_discretize_position(p_i, cellSize);
    float smoothRadiusSquared = smoothRadius * smoothRadius;

    // Search this particle's cell and all adjacent cells
    int count = 0;
    int start = i * maxNeighbours;
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                uint32_t hash = device_get_hash(cell + glm::ivec3(dx, dy, dz), tableSize);

                int candidateIndex = hashToFirstParticleIndex[hash];
                if (candidateIndex == UINT32_MAX) continue; // No neighbours here

                while (candidateIndex < numParticles && count < maxNeighbours) {
                    // Skip if candidate is self
                    if (candidateIndex == i) {
                        candidateIndex++;
                        continue;
                    }

                    Particle* candidate = &sortedParticles[candidateIndex];

                    // Stop if we are outside the current cell
                    if (candidate->hash != hash) break;

                    // Record if this is a neighbour
                    glm::vec3 separation = p_i->new_position - candidate->new_position;
                    float distanceSquared = glm::dot(separation, separation);
                    if (distanceSquared < smoothRadiusSquared) {
                        neighbourArray[start + count] = candidateIndex;
                        count++;
                    }

                    candidateIndex++;
                }
            }
        }
    }

    neighbourCount[i] = count;
}

// ================================= Wrappers ==================================

void findParticleNeighboursCUDA(
    cudaGraphicsResource* particlesResource,
    int numParticles,
    uint32_t* neighbourArray,
    uint32_t* neighbourCount,
    float smoothRadius,
    float cellSize,
    const int tableSize,
    const int maxNeighbours
) {
    // Map resource data to CUDA, retrieve device pointer
    cudaGraphicsMapResources(1, &particlesResource, 0);
    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    // Let thrust handle the copy of neighbour array data from host to device. Same as in ConstraintSolverCUDA
    thrust::device_vector<uint32_t> d_neighbourArrayVector(neighbourArray, neighbourArray + numParticles * maxNeighbours);
    thrust::device_vector<uint32_t> d_neighbourCountVector(neighbourCount, neighbourCount + numParticles);
    uint32_t* d_neighbourArray = thrust::raw_pointer_cast(d_neighbourArrayVector.data());
    uint32_t* d_neighbourCount = thrust::raw_pointer_cast(d_neighbourCountVector.data());

    // Create a hash table on the GPU and retrieve pointer
    thrust::device_vector<uint32_t> d_hashToFirstParticleIndexVector(tableSize, UINT32_MAX);
    uint32_t* d_hashToFirstParticleIndex = thrust::raw_pointer_cast(d_hashToFirstParticleIndexVector.data());

    // Compute hash
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    kernel_compute_hashes<<<gridSize, blockSize>>>(d_particles, numParticles, smoothRadius, tableSize);
    cudaDeviceSynchronize();

    // Sort by hash
    thrust::device_ptr<Particle> d_particlePtr = thrust::device_pointer_cast(d_particles);
    thrust::sort(d_particlePtr, d_particlePtr + numParticles, [] __device__(const Particle& a, const Particle& b) {
        return a.hash < b.hash;
    });

    // Construct hash table (now d_particles still points to the first location of the array, but it will have been sorted)
    kernel_build_hash_table<<<gridSize, blockSize>>>(d_particles, numParticles, d_hashToFirstParticleIndex);
    cudaDeviceSynchronize();

    // Find neighbours (using the smoothing radius as the cell size)
    kernel_find_neighbours<<<gridSize, blockSize>>>(d_particles, numParticles, d_neighbourArray, d_neighbourCount, d_hashToFirstParticleIndex, 
                                                    cellSize, smoothRadius, tableSize, maxNeighbours);
    cudaDeviceSynchronize();

    // The neighbour arrays need to be saved, the particle resource doesn't need this, and hash table is no longer needed
    cudaMemcpy(neighbourArray, d_neighbourArray, numParticles * maxNeighbours * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(neighbourCount, d_neighbourCount, numParticles * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Let OpenGL regain control to the particle resource
    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}

} // namespace NeighbourSearchCUDA

} // namespace AkuaEngine
