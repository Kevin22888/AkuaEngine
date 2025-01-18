#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

#include <Smoothing_Kernel.h>
#include <SPH_Estimator.h>


__global__ void kernel_predict_position(Particle* particles, int numParticles, float dt, glm::vec3 force) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numParticles) {
        Particle* p = &particles[i];

        // The simplest Euler integration
        p->new_velocity = p->velocity + force * dt;
        p->new_position = p->position + p->new_velocity * dt;
    }
}

__device__ glm::ivec3 device_discretize_position(Particle* particle, float cellSize) {
    return {
        static_cast<int>(floorf(particle->new_position.x / cellSize)),
        static_cast<int>(floorf(particle->new_position.y / cellSize)),
        static_cast<int>(floorf(particle->new_position.z / cellSize))
    };
}

__device__ uint32_t device_get_hash(glm::ivec3 position, const int TABLE_SIZE) {
    return (static_cast<uint32_t>((position.x * 73856093)) ^ 
            static_cast<uint32_t>((position.y * 19349663)) ^ 
            static_cast<uint32_t>((position.z * 83492791))) % TABLE_SIZE;
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
                        if (glm::length2(candidate->new_position - p->new_position) < smoothRadiusSquared) {
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

__global__ void kernel_calculate_densities(Particle* sortedParticles, int numParticles,
                                           uint32_t* neighbourArray, uint32_t* neighbourCount,
                                           float smoothRadius, const int MAX_NEIGHBOURS) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numParticles) {
        Particle* p = &sortedParticles[i];

        // p itself contributes to the density as well
        p->density = p->mass * device_poly6(0, smoothRadius);            

        int start = i * MAX_NEIGHBOURS;
        for (int offset = 0; offset < neighbourCount[i]; offset++) {
            uint32_t j = neighbourArray[start + offset];
            Particle* p_j = &sortedParticles[j];

            float distanceSquared = glm::length2(p->new_position - p_j->new_position);
            p->density += p_j->mass * device_poly6(distanceSquared, smoothRadius);
        }

        // if (i == 200)
        //     printf("Neighbours of p_200: %d.\nDensity found: %f\n", neighbourCount[i], p->density);
    }
}

// Calculate the lambda parameter required to find delta position
// Must be called AFTER densities are found
__global__ void kernel_calculate_lambdas(Particle* sortedParticles, int numParticles,
                                         uint32_t* neighbourArray, uint32_t* neighbourCount,
                                         float smoothRadius, const int MAX_NEIGHBOURS, 
                                         const float REST_DENSITY, const float EPSILON) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numParticles) {
        Particle* p = &sortedParticles[i];

        // Find C_i
        float C_i = p->density / REST_DENSITY - 1;

        // Find sum of square magnitude of grad_pk(C_i), where k steps through all of i and neighbours j's
        // This has two cases, when k = i there is a summation over just all the neighbours
        // When k = j it is a single gradient (see derivation details in supplementary files)
        
        // k = i
        glm::vec3 grad_pi_Ci(0.0f, 0.0f, 0.0f);

        int start = i * MAX_NEIGHBOURS;
        for (int offset = 0; offset < neighbourCount[i]; offset++) {
            uint32_t j = neighbourArray[start + offset];
            Particle* p_j = &sortedParticles[j];

            grad_pi_Ci += p_j->mass * device_gradient_spiky(p->new_position - p_j->new_position, smoothRadius);
        }
        grad_pi_Ci /= REST_DENSITY;

        //printf("grad_pi_Ci: (%f, %f, %f)\n", grad_pi_Ci.x, grad_pi_Ci.y, grad_pi_Ci.z);

        // k = j
        // Accumulator for summation of all grad_pk_Ci
        float totalSum = 0;

        for (int offset = 0; offset < neighbourCount[i]; offset++) {
            uint32_t j = neighbourArray[start + offset];
            Particle* p_j = &sortedParticles[j];

            glm::vec3 grad_pj_Ci = - 1 / REST_DENSITY * p_j->mass * device_gradient_spiky(p->new_position - p_j->new_position, smoothRadius);
            totalSum += glm::length2(grad_pj_Ci);
        }

        p->lambda = - C_i / (totalSum + glm::length2(grad_pi_Ci) + EPSILON);
    }
}

__global__ void kernel_calculate_position_delta(Particle* sortedParticles, int numParticles,
                                                uint32_t* neighbourArray, uint32_t* neighbourCount,
                                                float smoothRadius, const int MAX_NEIGHBOURS, 
                                                const float REST_DENSITY) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numParticles) {
        Particle* p = &sortedParticles[i];

        glm::vec3 position_delta(0.0f, 0.0f, 0.0f);

        // Temporarily adding lambda correction here just for testing
        float k = 0.01f;
        float delta_q = 0.0f;
        float n = 5.0f;
        
        int start = i * MAX_NEIGHBOURS;
        for (int offset = 0; offset < neighbourCount[i]; offset++) {
            uint32_t j = neighbourArray[start + offset];
            Particle* p_j = &sortedParticles[j];

            // calculating lambda correction
            float r2 = glm::length2(p->new_position - p_j->new_position);
            float lambda_corr = - k * powf(device_poly6(r2, smoothRadius) / device_poly6(delta_q, smoothRadius), n);

            position_delta += (p->lambda + p_j->lambda + lambda_corr) * p_j->mass * device_gradient_spiky(p->new_position - p_j->new_position, smoothRadius);
        }

        p->position_delta = position_delta / REST_DENSITY;

        // if (i == 200)
        //     printf("P_delta: (%f, %f, %f)\nposition_delta mag: %f\n", 
        //     p->position_delta.x, p->position_delta.y, p->position_delta.z, glm::length(position_delta));
    }
}

__device__ void handle_particle_collision(Particle* p, glm::vec3 box_min, glm::vec3 box_max) { //, float elasticity, float offset) {
    float elasticity = 1.0f;
    float offset = 0.01f;
    
    // Check collisions with each boundary
    if (p->new_position.x <= box_min.x) {
        p->new_position.x = box_min.x + offset;
        p->velocity.x = -p->velocity.x * elasticity;
    } else if (p->new_position.x >= box_max.x) {
        p->new_position.x = box_max.x - offset;
        p->velocity.x = -p->velocity.x * elasticity;
    }

    if (p->new_position.y <= box_min.y) {
        p->new_position.y = box_min.y + offset;
        p->velocity.y = -p->velocity.y * elasticity;
    } else if (p->new_position.y >= box_max.y) {
        p->new_position.y = box_max.y - offset;
        p->velocity.y = -p->velocity.y * elasticity;
    }

    if (p->new_position.z <= box_min.z) {
        p->new_position.z = box_min.z + offset;
        p->velocity.z = -p->velocity.z * elasticity;
    } else if (p->new_position.z >= box_max.z) {
        p->new_position.z = box_max.z - offset;
        p->velocity.z = -p->velocity.z * elasticity;
    }

    p->new_position.x = glm::clamp(p->new_position.x, box_min.x, box_max.x);
    p->new_position.y = glm::clamp(p->new_position.y, box_min.y, box_max.y);
    p->new_position.z = glm::clamp(p->new_position.z, box_min.z, box_max.z);
}

__global__ void kernel_correct_position(Particle* sortedParticles, int numParticles, glm::vec3 box_min, glm::vec3 box_max) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numParticles) {
        Particle* p = &sortedParticles[i];

        // Apply correction to the predicted position
        p->new_position += p->position_delta;

        handle_particle_collision(p, box_min, box_max);
    }
}

__global__ void kernel_update_position_and_velocity(Particle* sortedParticles, int numParticles, float dt) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < numParticles) {
        Particle* p = &sortedParticles[i];

        // Update velocity: v_i = (new_position - position) / dt
        p->new_velocity = (p->new_position - p->position) / dt;

        // float MAX_VELOCITY = 50.0f;
        // if (glm::length(p->new_velocity) > MAX_VELOCITY) {
        //     p->new_velocity = glm::normalize(p->new_velocity) * MAX_VELOCITY;
        // }

        // if (i == 200) {
        //     printf("Old position: (%f, %f, %f)\nnew position: (%f, %f, %f)\nOld velocity: (%f, %f, %f)\nNew velocity: (%f, %f, %f)\np* - p: (%f, %f, %f)\n", 
        //     p->position.x, p->position.y, p->position.z, p->new_position.x, p->new_position.y, p->new_position.z, 
        //     p->velocity.x, p->velocity.y, p->velocity.z, p->new_velocity.x, p->new_velocity.y, p->new_velocity.z,
        //     p->new_position.x - p->position.x, p->new_position.y - p->position.y, p->new_position.z - p->position.z);
        // }

        p->position = p->new_position;
        p->velocity = p->new_velocity;
    }
}



void predictNewPositionCUDA(cudaGraphicsResource* particlesResource, int numParticles, glm::vec3 force, float deltaTime) {
    // First map the resource to CUDA
    cudaGraphicsMapResources(1, &particlesResource, 0);

    // Then get a pointer for it
    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    // Launch kernels with this mapped resource's pointer
    int threadsPerBlock = 256;
    int blocksPerGrid = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    kernel_predict_position<<<blocksPerGrid, threadsPerBlock>>>(d_particles, numParticles, deltaTime, force);
    cudaDeviceSynchronize();

    // Finally upmap the resource
    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}

void findParticleNeighboursCUDA(cudaGraphicsResource* particlesResource, int numParticles, float smoothRadius, 
                                uint32_t* neighbourArray, uint32_t* neighbourCount, const int TABLE_SIZE, const int MAX_NEIGHBOURS) {
    // Map particle data and retrieve pointer
    cudaGraphicsMapResources(1, &particlesResource, 0);
    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    // Copy neighbour arrays to GPU and retrieve pointers
    thrust::device_vector<uint32_t> d_neighbourArrayVector(neighbourArray, neighbourArray + numParticles * MAX_NEIGHBOURS);
    thrust::device_vector<uint32_t> d_neighbourCountVector(neighbourCount, neighbourCount + numParticles);
    uint32_t* d_neighbourArray = thrust::raw_pointer_cast(d_neighbourArrayVector.data());
    uint32_t* d_neighbourCount = thrust::raw_pointer_cast(d_neighbourCountVector.data());

    // Create a hash table on the GPU and retrieve pointer
    thrust::device_vector<uint32_t> d_hashToFirstParticleIndexVector(TABLE_SIZE, UINT32_MAX);
    uint32_t* d_hashToFirstParticleIndex = thrust::raw_pointer_cast(d_hashToFirstParticleIndexVector.data());

    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    
    // Compute hash and stored in each particle's hash field
    kernel_compute_hashes<<<gridSize, blockSize>>>(d_particles, numParticles, smoothRadius, TABLE_SIZE);
    cudaDeviceSynchronize();

    // Sort by hash
    thrust::device_ptr<Particle> d_particlePtr = thrust::device_pointer_cast(d_particles);
    thrust::sort(d_particlePtr, d_particlePtr + numParticles, [] __device__(const Particle& a, const Particle& b) {
        return a.hash < b.hash;
    });

    // Construct hash table (now d_particles still points to the first location of the array, but it will have been sorted)
    kernel_build_hash_table<<<gridSize, blockSize>>>(d_particles, numParticles, d_hashToFirstParticleIndex);
    cudaDeviceSynchronize();

    // Find neighbours
    kernel_find_neighbours<<<gridSize, blockSize>>>(d_particles, numParticles, d_hashToFirstParticleIndex, 
                                                    smoothRadius, smoothRadius, d_neighbourArray, d_neighbourCount, 
                                                    TABLE_SIZE, MAX_NEIGHBOURS);
    cudaDeviceSynchronize();

    // The neighbour arrays need to be saved, the particle resource doesn't need this, and hash table is no longer needed
    cudaMemcpy(neighbourArray, d_neighbourArray, numParticles * MAX_NEIGHBOURS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(neighbourCount, d_neighbourCount, numParticles * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}

void runConstraintSolverCUDA(cudaGraphicsResource* particlesResource, int numParticles, float smoothRadius, int solverIterations,
                             uint32_t* neighbourArray, uint32_t* neighbourCount, const int MAX_NEIGHBOURS, const float REST_DENSITY, const float EPSILON, glm::vec3 box_min, glm::vec3 box_max) {
    // Again, pointers for particles and neighbour arrays
    cudaGraphicsMapResources(1, &particlesResource, 0);
    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    thrust::device_vector<uint32_t> d_neighbourArrayVector(neighbourArray, neighbourArray + numParticles * MAX_NEIGHBOURS);
    thrust::device_vector<uint32_t> d_neighbourCountVector(neighbourCount, neighbourCount + numParticles);
    uint32_t* d_neighbourArray = thrust::raw_pointer_cast(d_neighbourArrayVector.data());
    uint32_t* d_neighbourCount = thrust::raw_pointer_cast(d_neighbourCountVector.data());

    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;

    //printf("Running solver once\n");
    while (solverIterations-- > 0) {  // TODO: these calculations should be using the new positions in each iteration, right now they use the original
        //printf("-----Solver iteration----\n");
        kernel_calculate_densities<<<gridSize, blockSize>>>(d_particles, numParticles, d_neighbourArray, d_neighbourCount, smoothRadius, MAX_NEIGHBOURS);
        cudaDeviceSynchronize();

        kernel_calculate_lambdas<<<gridSize, blockSize>>>(d_particles, numParticles, d_neighbourArray, d_neighbourCount, smoothRadius, MAX_NEIGHBOURS, REST_DENSITY, EPSILON);
        cudaDeviceSynchronize();

        kernel_calculate_position_delta<<<gridSize, blockSize>>>(d_particles, numParticles, d_neighbourArray, d_neighbourCount, smoothRadius, MAX_NEIGHBOURS, REST_DENSITY);
        cudaDeviceSynchronize();

        kernel_correct_position<<<gridSize, blockSize>>>(d_particles, numParticles, box_min, box_max);
        cudaDeviceSynchronize();
    }

    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}

void correctPositionCUDA(cudaGraphicsResource* particlesResource, int numParticles, glm::vec3 box_min, glm::vec3 box_max) {
    cudaGraphicsMapResources(1, &particlesResource, 0);

    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    // Apply position delta to correct the predicted position
    // NOTE: collision detection is handled within this with a simple projection to surface
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    kernel_correct_position<<<gridSize, blockSize>>>(d_particles, numParticles, box_min, box_max);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}

void updatePositionAndVelocityCUDA(cudaGraphicsResource* particlesResource, int numParticles, float deltaTime) {
    cudaGraphicsMapResources(1, &particlesResource, 0);

    Particle* d_particles;
    size_t numBytes;
    cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &numBytes, particlesResource);

    // Apply position delta to correct the predicted position
    // NOTE: collision detection is handled within this with a simple projection to surface
    int blockSize = 256;
    int gridSize = (numParticles + blockSize - 1) / blockSize;
    kernel_update_position_and_velocity<<<gridSize, blockSize>>>(d_particles, numParticles, deltaTime);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &particlesResource, 0);
}
