#ifndef AKUAENGINE_PARTICLE_H
#define AKUAENGINE_PARTICLE_H

#include <glm/glm.hpp>

namespace AkuaEngine {

struct Particle {
    // Primary physical states
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 new_position;
    glm::vec3 new_velocity;
    
    // Density Solver adjustment
    glm::vec3 position_delta;

    glm::vec3 vorticity;

    // SPH fields
    float mass;
    float density;
    float lambda;
    
    // Spatial hashing
    uint32_t hash;
    
    // Rendering
    glm::vec4 color;
    float size;
};

} // namespace AkuaEngine

#endif // AKUAENGINE_PARTICLE_H