#ifndef AKUAENGINE_MATH_UTILS_H
#define AKUAENGINE_MATH_UTILS_H

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace AkuaEngine {

namespace MathUtilsCUDA {

// Later we will remove glm vectors from all the CUDA logic
__device__ inline glm::vec3 cross(const glm::vec3& a, const glm::vec3& b) {
    return glm::vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

} // namespace MathUtilsCUDA

} // namespace AkuaEngine

#endif // AKUAENGINE_MATH_UTILS_H