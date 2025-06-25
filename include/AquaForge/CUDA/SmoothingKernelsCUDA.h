#ifndef AQUAFORGE_SMOOTHING_KERNEL_H
#define AQUAFORGE_SMOOTHING_KERNEL_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace AquaForge {

namespace SmoothingKernels {

// Mark these functions inline, so they can appear in multiple translation units without violating the One Definition Rule.
// Without inline, each file including this header will define its own version of the same function, causing a linker error.
// Alternatively, no inlining but enabling Relocatable Device Code (RDC) can allow cross-file __device__ function definitions,
// but be careful with misusing it and causing overhead from too many function calls.

__device__ inline float device_poly6(float distanceSquared, float h) {
    float h2 = h * h;
    if (distanceSquared > h2) return 0.0f;

    return 315.0f / (64.0f * 3.14f * powf(h, 9.0f)) * powf(h2 - distanceSquared, 3.0f);
}

__device__ inline glm::vec3 device_gradient_spiky(glm::vec3 r_vector, float h) {
    float r_magnitude = glm::length(r_vector);
    if (r_magnitude > h || r_magnitude < 1e-5f) return glm::vec3(0.0f); // Avoid instability by discarding super small distances

    return - 45.0f / (3.14f * powf(h, 6.0f)) * powf(h - r_magnitude, 2.0f) * r_vector / r_magnitude;
}

} // namespace SmoothingKernels

} // namespace AquaForge

#endif // AQUAFORGE_SMOOTHING_KERNEL_H