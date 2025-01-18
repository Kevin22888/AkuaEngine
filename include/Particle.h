#ifndef PBF_PARTICLE_H
#define PBF_PARTICLE_H

#include <glm/glm.hpp>

struct Particle {
    glm::vec3 position;
    glm::vec3 velocity;
    glm::vec3 new_position;
    glm::vec3 new_velocity;
    glm::vec3 position_delta;
    float mass;
    float density;
    uint32_t hash;
    float lambda;
    glm::vec4 color;
    float size;
};

#endif // PBF_PARTICLE_H