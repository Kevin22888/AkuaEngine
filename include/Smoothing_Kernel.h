#ifndef PBF_SMOOTHING_KERNEL_H
#define PBF_SMOOTHING_KERNEL_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/norm.hpp>

// If they aren't inlined, using them in another .cu file won't work, linkage is different (?)
inline __device__ float device_poly6(float distanceSquared, float h) {
    float h2 = h * h;
    if (distanceSquared > h2) return 0.0f;

    return 315.0f / (64.0f * 3.14f * glm::pow(h, 9.0f)) * glm::pow(h2 - distanceSquared, 3.0f);
}

inline __device__ glm::vec3 device_gradient_spiky(glm::vec3 r_vector, float h) { // consider switching to CUDA float3
    float r = glm::length(r_vector);
    if (r > h) return glm::vec3(0.0f, 0.0f, 0.0f);

    return - 45.0f / (3.14f * powf(h, 6)) * glm::pow(h - r, 2.0f) * r_vector / r;
}

#endif // PBF_SMOOTHING_KERNEL_H