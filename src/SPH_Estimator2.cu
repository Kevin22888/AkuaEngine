#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>
#include <SPH_Estimator.h>

////////////////////////////////////////////////////////////////////////////////
//                            Smoothing Kernels                               //
////////////////////////////////////////////////////////////////////////////////
__device__ float device_poly6(glm::vec3 r_vector, float h) {
    float distanceSquared = glm::length2(r_vector);
    float h2 = h * h;
    if (distanceSquared > h2) return 0.0f;

    return 315.0f / (64.0f * 3.1415926535f * glm::pow(h, 9.0f)) * glm::pow(h2 - distanceSquared, 3.0f);
}

__device__ glm::vec3 device_gradient_spiky(glm::vec3 r_vector, float h) { // consider switching to CUDA float3
    float r = glm::length(r_vector);
    if (r >= h) return glm::vec3(0.0f, 0.0f, 0.0f);

    return - 45.0f / (3.1415926535f * powf(h, 6)) * powf(h - r, 2.0f) * glm::normalize(r_vector);
}
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//                            Spatial Hashing                                 //
////////////////////////////////////////////////////////////////////////////////
__device__ glm::ivec3 device_discretize_position(Particle* particle, float cellSize) {
    return {
        static_cast<int>(floorf(particle->position.x / cellSize)),
        static_cast<int>(floorf(particle->position.y / cellSize)),
        static_cast<int>(floorf(particle->position.z / cellSize))
    };
}

__device__ uint32_t device_get_hash(glm::ivec3 position, const int TABLE_SIZE) {
    return (static_cast<uint32_t>((position.x * 73856093)) ^ 
            static_cast<uint32_t>((position.y * 19349663)) ^ 
            static_cast<uint32_t>((position.z * 83492791))) % TABLE_SIZE;
}

__global__ void kernel_reset_hash_table(uint32_t* hashTable, const int TABLE_SIZE) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < TABLE_SIZE) {
        hashTable[i] = UINT32_MAX;
    }
}

__global__ void kernel_compute_hashes(Particle* particles, int numParticles, float cellSize, const int TABLE_SIZE) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numParticles) {
        Particle* particle = &particles[i];

        glm::ivec3 cell = device_discretize_position(particle, cellSize);
        particle->hash = device_get_hash(cell, TABLE_SIZE);;
    }
}

__global__ void kernel_build_hash_table(Particle* sortedParticles, int numParticles, uint32_t* hashToFirstParticleIndex) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numParticles) {
        uint32_t hash = sortedParticles[i].hash;

        if (i == 0) {
            hashToFirstParticleIndex[hash] = i;
        } else {
            if (hash != sortedParticles[i - 1].hash) {
                hashToFirstParticleIndex[hash] = i;
            }
        }
    }
}

__global__ void kernel_find_neighbours(Particle* sortedParticles, int numParticles, 
                                       uint32_t* hashToFirstParticleIndex, float cellSize, float smoothRadius, 
                                       uint32_t* neighbourArray, uint32_t* neighbourCount,
                                       const int TABLE_SIZE, const int MAX_NEIGHBOURS) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numParticles) {
        Particle* p = &sortedParticles[i];
        glm::ivec3 cell = device_discretize_position(p, cellSize);
        float smoothRadiusSquared = smoothRadius * smoothRadius;
        int start = i * MAX_NEIGHBOURS;
        int count = 0;
        
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                for (int dz = -1; dz <= 1; dz++) {
                    uint32_t hash = device_get_hash(cell + glm::ivec3(dx, dy, dz), TABLE_SIZE);

                    int candidateIndex = hashToFirstParticleIndex[hash];
                    if (candidateIndex == UINT32_MAX) continue; // no neighbours here

                    while (candidateIndex < numParticles && count < MAX_NEIGHBOURS) {
                        // skip if candidate is self
                        if (candidateIndex == i) {
                            candidateIndex++;
                            continue;
                        }

                        Particle* candidate = &sortedParticles[candidateIndex];

                        // stop if we are outside the current cell
                        if (candidate->hash != hash) break;

                        // record if this is a neighbour
                        if (glm::length2(candidate->position - p->position) < smoothRadiusSquared) {
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
}
////////////////////////////////////////////////////////////////////////////////


__global__ void kernel_compute_density(Particle* sortedParticles, int numParticles, int smoothRadius, uint32_t* neighbourArray, uint32_t* neighbourCount, int MAX_NEIGHBOURS) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= numParticles) return;

    Particle* p = &sortedParticles[i];

    p->density = p->mass * device_poly6(glm::zero<glm::vec3>(), smoothRadius);

    int neightbourStart = i * MAX_NEIGHBOURS;
    for (int offset = 0; offset < neighbourCount[i]; offset++) {
        uint16_t j = neightbourStart + offset;
        Particle* p_j = &sortedParticles[j];
        p->density += p_j->mass * device_poly6(p->position - p_j->position, smoothRadius);
    }
}


struct CompareByHash {
    __host__ __device__
    bool operator()(const Particle& a, const Particle& b) const {
        return a.hash < b.hash;
    }
};


void testCUDAKernels(cudaGraphicsResource* particlesResource, int numParticles, float smoothRadius, cudaGraphicsResource* neighbourArrayResource, cudaGraphicsResource* neighbourCountResource, cudaGraphicsResource* hashTableResource, const int TABLE_SIZE, const int MAX_NEIGHBOURS) {
    printf("Mapping resources\n");
    
    // Map particle data, neighbour data and retrieve pointers
    cudaGraphicsMapResources(1, &particlesResource, 0);
    cudaGraphicsMapResources(1, &neighbourArrayResource, 0);
    cudaGraphicsMapResources(1, &neighbourCountResource, 0);
    cudaGraphicsMapResources(1, &hashTableResource, 0);
    Particle* d_particles;
    uint32_t* d_neighbourArray;
    uint32_t* d_neighbourCount;
    uint32_t* d_hashToFirstParticleIndex;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);
    cudaGraphicsResourceGetMappedPointer((void**)&d_neighbourArray, &numBytes, neighbourArrayResource);
    cudaGraphicsResourceGetMappedPointer((void**)&d_neighbourCount, &numBytes, neighbourCountResource);
    cudaGraphicsResourceGetMappedPointer((void**)&d_hashToFirstParticleIndex, &numBytes, hashTableResource);

    printf("Launching reset hash table\n");
    
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    kernel_reset_hash_table<<<gridSize, blockSize>>>(d_hashToFirstParticleIndex, TABLE_SIZE);
    cudaDeviceSynchronize();

    printf("Launching compute hash\n");

    kernel_compute_hashes<<<gridSize, blockSize>>>(d_particles, numParticles, smoothRadius, TABLE_SIZE);
    cudaDeviceSynchronize();

    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("Free memory: %zu / %zu bytes\n", freeMem, totalMem);

    printf("Casting device pointer\n");

    thrust::device_ptr<Particle> d_particlePtr = thrust::device_pointer_cast(d_particles);

    printf("Sorting hash\n");

    thrust::sort(d_particlePtr, d_particlePtr + numParticles, CompareByHash());

    printf("Launching build hash table\n");

    // Construct hash table (now d_particles still points to the first location of the array, but it will have been sorted)
    kernel_build_hash_table<<<gridSize, blockSize>>>(d_particles, numParticles, d_hashToFirstParticleIndex);
    cudaDeviceSynchronize();

    printf("Launching neighbour search\n");

    // Find neighbours
    kernel_find_neighbours<<<gridSize, blockSize>>>(d_particles, numParticles, d_hashToFirstParticleIndex, 
                                                    smoothRadius, smoothRadius, d_neighbourArray, d_neighbourCount, 
                                                    TABLE_SIZE, MAX_NEIGHBOURS);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &particlesResource, 0);
    cudaGraphicsUnmapResources(1, &neighbourArrayResource, 0);
    cudaGraphicsUnmapResources(1, &neighbourCountResource, 0);
    cudaGraphicsUnmapResources(1, &hashTableResource, 0);
}


void computeDensitiesCUDA(cudaGraphicsResource* particlesResource, int numParticles, float smoothRadius, cudaGraphicsResource* neighbourArrayResource, cudaGraphicsResource* neighbourCountResource, const int MAX_NEIGHBOURS) {
    // Again, pointers for particles and neighbour arrays
    cudaGraphicsMapResources(1, &particlesResource, 0);
    cudaGraphicsMapResources(1, &neighbourArrayResource, 0);
    cudaGraphicsMapResources(1, &neighbourCountResource, 0);
    Particle* d_particles;
    uint32_t* d_neighbourArray;
    uint32_t* d_neighbourCount;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);
    cudaGraphicsResourceGetMappedPointer((void**)&d_neighbourArray, &numBytes, neighbourArrayResource);
    cudaGraphicsResourceGetMappedPointer((void**)&d_neighbourCount, &numBytes, neighbourCountResource);

    printf("Launching compute density\n");

    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    kernel_compute_density<<<gridSize, blockSize>>>(d_particles, numParticles, smoothRadius, d_neighbourArray, d_neighbourCount, MAX_NEIGHBOURS);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &particlesResource, 0);
    cudaGraphicsUnmapResources(1, &neighbourArrayResource, 0);
    cudaGraphicsUnmapResources(1, &neighbourCountResource, 0);
}