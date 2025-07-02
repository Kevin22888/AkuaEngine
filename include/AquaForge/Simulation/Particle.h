#ifndef AQUAFORGE_PARTICLE_H
#define AQUAFORGE_PARTICLE_H

#include <glm/glm.hpp>

namespace AquaForge {

struct Particle {
    // Primary physical states
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 new_position;
    glm::vec3 new_velocity;
    
    // Density Solver adjustment
    glm::vec3 position_delta;

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

} // namespace AquaForge

#endif // AQUAFORGE_PARTICLE_H